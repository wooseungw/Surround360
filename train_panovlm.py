# 파일 경로: train_llava.py

import os
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    CLIPVisionConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoImageProcessor, AutoTokenizer, LlavaNextProcessor, Blip2Config
)
import wandb

# 데이터셋 및 모델 임포트
from dataset import QuIC360Dataset, data_collator
from src.models.panorama_llava import PanoramaLLaVA, PanoramaLLaVAConfig

# --- 1. 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- 2. 설정 파일 로드 및 인자 파싱 ---
def parse_args() -> argparse.Namespace:
    """스크립트 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description="Finetune a Panorama-LLaVA model.")
    parser.add_argument("--config", type=str, required=True, help="Path to the training configuration YAML file.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training.")
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """YAML 설정 파일을 로드합니다."""
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded successfully.")
    return config

# --- 3. 메인 학습 로직 ---
def main():
    args = parse_args()
    config = load_config(args.config)

    # --- 3.1. WandB 초기화 (DDP 환경에서는 랭크 0에서만 초기화) ---
    wandb_cfg = config.get('wandb', {})
    if args.local_rank <= 0:  # 랭크 0 또는 비분산 환경
        wandb.init(
            project=wandb_cfg.get('project', 'PanoramaLLaVA'),
            name=wandb_cfg.get('name', 'finetune-run'),
            config=config  # 전체 설정을 wandb에 기록
        )
        logger.info("WandB initialized.")

    # --- 3.2. 프로세서 로드 ---
    model_cfg = config['model']
    vision_encoder_name = model_cfg['vision_encoder_name']
    language_model_name = model_cfg['language_model_name']

    # 1. Vision Encoder에 맞는 Image Processor 로드
    image_processor = AutoImageProcessor.from_pretrained(vision_encoder_name)
    
    # 2. Language Model에 맞는 Tokenizer 로드
    tokenizer = AutoTokenizer.from_pretrained(language_model_name)
    
    # 3. Blip2Processor를 사용하여 두 컴포넌트를 하나로 합칩니다.
    #    (이름은 Blip2Processor지만, 범용적으로 사용 가능합니다)
    processor = LlavaNextProcessor(image_processor=image_processor, tokenizer=tokenizer)
    # LLaMA, Gemma 등 BOS/EOS 기반 모델을 위한 PAD 토큰 설정
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    logger.info("Vision-Language Processor created successfully.")
    
    # --- 3.3. 모델 구성 및 가중치 로드 ---
    logger.info("Building PanoramaLLaVA model...")
    vision_config = AutoConfig.from_pretrained(vision_encoder_name)
    language_config = AutoConfig.from_pretrained(language_model_name)
    
    # 2. PanoramaLLaVAConfig를 생성할 때, 로드한 config 객체를 그대로 전달합니다.
    model_config = PanoramaLLaVAConfig(
        vision_config=vision_config,       # hf_config.vision_config -> vision_config 로 수정
        language_config=language_config,
        mm_hidden_size=model_cfg.get('mm_hidden_size', 512)
    )

    model = PanoramaLLaVA(model_config)
    model.vision_tower = AutoModel.from_pretrained(vision_encoder_name)
    model.language_model = AutoModelForCausalLM.from_pretrained(language_model_name)
    
    
    # 2) 각 부분에 사전학습된 가중치 채워넣기
    logger.info(f"Loading weights from vision_tower: {model_cfg['vision_encoder_name']}")
    model.vision_tower = AutoModel.from_pretrained(model_cfg['vision_encoder_name'])
    
    logger.info(f"Loading weights from language_model: {model_cfg['language_model_name']}")
    model.language_model = AutoModelForCausalLM.from_pretrained(model_cfg['language_model_name'])
    
    # --- 3.4. 모델 파라미터 동결 설정 ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = 0
    
    logger.info("Setting parameter requires_grad...")
    for name, param in model.named_parameters():
        if "vision_tower" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            trainable_params += param.numel()
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({(100 * trainable_params / total_params):.2f}%)")

    # --- 3.5. 데이터셋 준비 ---
    data_cfg = config['data']
    data_dir = Path(data_cfg['dir'])
    
    logger.info(f"Loading train dataset from: {data_dir/data_cfg['train_file']}")
    train_dataset = QuIC360Dataset(
        data_dir / data_cfg['train_file'], 
        processor, 
        max_length=data_cfg['max_length'],
        split="train",
        image_size=data_cfg['image_size'],
        do_crop=data_cfg['do_crop'],
        fov=data_cfg['fov'],
        overlap_ratio=data_cfg['overlap_ratio'],
        use_augmentation=data_cfg.get('use_augmentation', False) 
    )
    eval_dataset = QuIC360Dataset(
        data_dir / data_cfg['valid_file'], 
        processor, 
        max_length=data_cfg['max_length'],
        split="valid",
        image_size=data_cfg['image_size'],
        do_crop=data_cfg['do_crop'],
        fov=data_cfg['fov'],
        overlap_ratio=data_cfg['overlap_ratio'],
        use_augmentation=False # 평가 시에는 항상 증강을 끔
    )
    logger.info(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(eval_dataset)}")

    # --- 3.6. TrainingArguments 설정 ---
    training_cfg = config['training']
    
    # DeepSpeed 설정 준비
    deepspeed_cfg = config.get('deepspeed', {})
    deepspeed_config = None
    if deepspeed_cfg.get('enabled', False) and 'config' in deepspeed_cfg:
        deepspeed_config = deepspeed_cfg['config']
        logger.info(f"DeepSpeed enabled with config: {deepspeed_config}")
    
    training_args = TrainingArguments(
        output_dir=training_cfg['output_dir'],
        num_train_epochs=training_cfg['num_epochs'],
        per_device_train_batch_size=training_cfg['per_device_train_batch_size'],
        per_device_eval_batch_size=training_cfg['per_device_eval_batch_size'],
        gradient_accumulation_steps=training_cfg.get('gradient_accumulation_steps', 1),
        learning_rate=training_cfg['learning_rate'],
        warmup_ratio=training_cfg['warmup_ratio'],
        weight_decay=training_cfg['weight_decay'],
        logging_dir=training_cfg['logging_dir'],
        logging_steps=training_cfg['logging_steps'],
        eval_strategy=training_cfg.get('eval_strategy', 'steps'),
        eval_steps=training_cfg.get('eval_steps', 500),
        save_strategy=training_cfg.get('save_strategy', 'steps'),
        save_steps=training_cfg.get('save_steps', 500),
        save_total_limit=training_cfg.get('save_total_limit', 3),
        
        # 추가 설정
        run_name=training_cfg.get('run_name', 'llava-finetune-run'),
        gradient_checkpointing=training_cfg.get('gradient_checkpointing', True),
        fp16=training_cfg.get('fp16', False),
        load_best_model_at_end=training_cfg.get('load_best_model_at_end', True),
        metric_for_best_model=training_cfg.get('metric_for_best_model', 'eval_loss'),
        greater_is_better=training_cfg.get('greater_is_better', False),
        dataloader_num_workers=training_cfg.get('dataloader_num_workers', 4),
        
        local_rank=args.local_rank,
        deepspeed = deepspeed_config if deepspeed_config else None  # DeepSpeed 지원 추가
    )

    # --- 3.7. Trainer 초기화 및 학습 실행 ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training finished.")
    
    # --- 3.8. 최종 모델 저장 ---
    save_dir = Path(training_cfg['output_dir'])
    
    # 분산 학습 환경에서는 랭크 0에서만 최종 저장
    if args.local_rank <= 0:
        logger.info(f"Saving final model to {save_dir}...")
        trainer.save_model(str(save_dir))
        processor.save_pretrained(save_dir)
        logger.info("Model and processor saved successfully.")
        wandb.finish()

if __name__ == "__main__":
    main()