import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from transformers import (
    Blip2Processor,
    Blip2Config,
    TrainingArguments,
    Trainer
)
from torch.nn import CrossEntropyLoss

from PIL import Image
Image.MAX_IMAGE_PIXELS = None # 대용량 파노라마 이미지 로드를 위한 설정

import wandb

# [가정] 아래 파일들이 src/ 디렉토리 등에 올바르게 위치해 있다고 가정합니다.
from dataset import QuIC360Dataset, data_collator
from src.models.surroundblip import SurroundBlip 


class Stage1Trainer(Trainer):
    """
    1단계 Vision Pre-training을 위한 커스텀 Trainer.
    모델의 forward 함수에 `pretrain_vision_only=True` 인자를 자동으로 전달합니다.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs): # **kwargs 추가
        """
        Hugging Face Trainer의 training_step에서 전달하는 추가 인자들(예: num_items_in_batch)을
        **kwargs로 받아 에러를 방지합니다.
        """
        # 모델의 forward 함수를 호출할 때, 1단계 학습임을 알리는 플래그를 전달합니다.
        # 데이터셋이 텍스트 관련 입력을 포함하더라도, 모델은 이를 무시하고 overlap_loss만 계산합니다.
        outputs = model(**inputs, pretrain_vision_only=True)
        loss = outputs.get("loss")
        return (loss, outputs) if return_outputs else loss



def parse_args():
    """커맨드 라인 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description="Train SurroundBlip model with 2-stage strategy")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    # --- 학습 단계를 지정하는 인자 ---
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2], help="Training stage: 1 for vision pre-training, 2 for fine-tuning")
    return parser.parse_args()


def load_config(config_path):
    """YAML 설정 파일을 로드합니다."""
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print("Loaded config:", config)
    return config


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # WandB 초기화
    wandb_config = config.get('wandb', {})
    wandb.init(
        project=wandb_config.get('project', 'Surround360'),
        name=f"{wandb_config.get('name', 'run')}_stage{args.stage}" # 실행 이름에 stage 명시
    )
    
    # BLIP-2 모델 및 프로세서 로드
    pretrain_name = config['model']['pretrain_name']
    processor = Blip2Processor.from_pretrained(pretrain_name, use_fast=False)
    
    # SurroundBlip 모델 로드
    hf_config = Blip2Config.from_pretrained(pretrain_name)
    if 'num_query_tokens' in config['model']:
        hf_config.num_query_tokens = config['model']['num_query_tokens']
    if 'qformer' in config['model']:
        for key, value in config['model']['qformer'].items():
            if hasattr(hf_config.qformer_config, key):
                setattr(hf_config.qformer_config, key, value)
            
    model = SurroundBlip.from_pretrained(
        pretrain_name,
        config=hf_config,
        ignore_mismatched_sizes=True
    )
    
    # --- [핵심 2] 단계별 파라미터 동결/해제 로직 수정 ---
    if args.stage == 1:
        print("--- CONFIGURING FOR STAGE 1: VISION PRE-TRAINING ---")
        # Vision Model만 학습하도록 설정
        for name, param in model.named_parameters():
            if "vision_model" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("Training only vision_model parameters.")
    
    elif args.stage == 2:
        print("--- CONFIGURING FOR STAGE 2: INSTRUCTION FINE-TUNING ---")
        # Vision Model은 동결하고, 나머지(Q-Former, LM)는 학습
        for name, param in model.named_parameters():
            if "vision_model" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        print("Froze vision_model. Training Q-Former, Language Projection, and Language Model.")

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 데이터셋 및 데이터로더 초기화
    data_cfg = config['data']
    data_dir = Path(data_cfg['dir'])
    
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
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Eval dataset length: {len(eval_dataset)}")

    # 학습 인자 설정
    training_cfg = config['training']
    training_args = TrainingArguments(
        output_dir=training_cfg['output_dir'],
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
        save_only_model=True
    )

    # --- [핵심 3] 단계에 맞는 트레이너 선택 및 초기화 ---
    if args.stage == 1:
        print("Initializing Stage1Trainer for vision pre-training.")
        trainer = Stage1Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
    else: # stage == 2
        print("Initializing standard Trainer for fine-tuning.")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
  
    # 학습 시작
    trainer.train()
    
    # 모델 및 프로세서 저장
    save_dir = Path(config['model']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir)) # trainer.save_model 사용 권장
    processor.save_pretrained(save_dir)
    print(f"Model and processor saved to {save_dir}")
    
    wandb.finish()

if __name__ == "__main__":
    main()