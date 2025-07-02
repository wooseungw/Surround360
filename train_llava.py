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
    AutoModelForCausalLM
)
import wandb

# [가정] 아래 파일들이 올바른 경로에 위치해 있다고 가정합니다.
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

    # --- 3.1. WandB 초기화 ---
    wandb_cfg = config.get('wandb', {})
    wandb.init(
        project=wandb_cfg.get('project', 'PanoramaLLaVA'),
        name=wandb_cfg.get('name', 'finetune-run'),
        config=config  # 전체 설정을 wandb에 기록
    )
    logger.info("WandB initialized.")

    # --- 3.2. 프로세서 로드 ---
    model_cfg = config['model']
    processor = AutoProcessor.from_pretrained(model_cfg['language_model_name'], use_fast=False)
    if processor.pad_token is None:
        processor.pad_token = processor.eos_token
        logger.info("Processor pad_token is set to eos_token.")

    # --- 3.3. 모델 구성 및 가중치 로드 ---
    logger.info("Building PanoramaLLaVA model...")
    vision_config = CLIPVisionConfig.from_pretrained(model_cfg['vision_encoder_name'])
    language_config = AutoConfig.from_pretrained(model_cfg['language_model_name'])
    
    model_config = PanoramaLLaVAConfig(
        vision_config=vision_config,
        language_config=language_config,
        mm_hidden_size=model_cfg.get('mm_hidden_size', 512)
    )

    # 1) 아키텍처(뼈대) 생성
    model = PanoramaLLaVA(model_config)
    
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
    training_args = TrainingArguments(
        utput_dir=training_cfg['output_dir'],
        run_name=f"{training_cfg.get('run_name', 'run')}_stage{args.stage}",
        num_train_epochs=training_cfg['num_epochs'],
        per_device_train_batch_size=training_cfg['batch_size']['train'],
        per_device_eval_batch_size=training_cfg['batch_size']['eval'],
        gradient_accumulation_steps=training_cfg.get('gradient_accumulation_steps', 1),
        gradient_checkpointing=training_cfg.get('gradient_checkpointing', True),
        learning_rate=float(training_cfg.get('learning_rate', 5e-5)),
        warmup_ratio=training_cfg.get('warmup_ratio', 0.03),
        weight_decay=training_cfg.get('weight_decay', 0.0),
        max_grad_norm=training_cfg.get('max_grad_norm', 1.0),
        dataloader_num_workers=training_cfg.get('dataloader_num_workers', 4),
        logging_dir=training_cfg.get('logging_dir', './logs'),
        logging_steps=training_cfg.get('logging_steps', 10),
        eval_strategy=training_cfg.get('eval_strategy', 'steps'),
        eval_steps=training_cfg.get('eval_steps', 500),
        save_strategy=training_cfg.get('save_strategy', 'steps'),
        save_steps=training_cfg.get('save_steps', 500),
        save_total_limit=training_cfg.get('save_total_limit', 2),
        load_best_model_at_end=training_cfg.get('load_best_model_at_end', True),
        metric_for_best_model=training_cfg.get('metric_for_best_model', 'eval_loss'),
        greater_is_better=training_cfg.get('greater_is_better', False),
        fp16=training_cfg.get('fp16', True),
        deepspeed=config.get('deepspeed', {}).get('config') if config.get('deepspeed', {}).get('enabled') else None,
        report_to=training_cfg.get('report_to', 'wandb'),
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
    logger.info(f"Saving final model to {save_dir}...")
    trainer.save_model(str(save_dir))
    processor.save_pretrained(save_dir)
    logger.info("Model and processor saved successfully.")
    
    wandb.finish()

if __name__ == "__main__":
    main()