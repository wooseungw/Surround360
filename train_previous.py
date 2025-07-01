import os

import torch
from torch.utils.data import DataLoader, Dataset
from transformers.training_args import TrainingArguments
from src.models.surroundblip_buffer import SurroundBlip

from transformers import Blip2Processor, Blip2ForConditionalGeneration, Blip2Config, Trainer
import pandas as pd
from PIL import Image
# 최대 픽셀 수 제한 해제 (None으로 설정)
Image.MAX_IMAGE_PIXELS = None

import wandb
from pathlib import Path

import yaml
import argparse
from typing import Dict, List, Optional, Union, Any

from py360convert import e2p
import numpy as np
import torch.nn.functional as F

from dataset import QuIC360Dataset, data_collator

PAD_TOKEN_ID = 1
IGNORE_INDEX = -100

def parse_args():
    parser = argparse.ArgumentParser(description="Train BLIP-2 model with parameters from a YAML file")
    parser.add_argument("--config", type=str, default="config/train.yaml", help="Path to the config file")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print("Loaded config:", config)
    return config


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # wandb 설정
    wandb.init(project=config['wandb']['project'], name=config['wandb']['name'])
    
    # BLIP-2 모델 및 프로세서 로드
    name = config['model']['name']
    print("Model name:", name)
    pretrain_name = config['model']['pretrain_name']
    processor = Blip2Processor.from_pretrained(pretrain_name, use_fast=False)
    # instantiate SurroundBlip with modified config
    if name == "surround":
        print("Loading SurroundBlip model")
        # load Hugging Face BLIP-2 config and override Q-Former settings if provided
        hf_config = Blip2Config.from_pretrained(pretrain_name)
        # override top-level num_query_tokens if present
        if 'num_query_tokens' in config['model']:
            hf_config.num_query_tokens = config['model']['num_query_tokens']
        # override nested qformer_config fields if present
        if 'qformer' in config['model']:
            for key, value in config['model']['qformer'].items():
                if hasattr(hf_config.qformer_config, key):
                    setattr(hf_config.qformer_config, key, value)
        model = SurroundBlip.from_pretrained(
            pretrain_name,
            config=hf_config,
            ignore_mismatched_sizes=True
        )
    else:
        print("Loading BLIP-2 model")
        model = Blip2ForConditionalGeneration.from_pretrained(pretrain_name)
    # Freeze vision encoder parameters
    # for param in model.vision_model.parameters():
    #     param.requires_grad = False
    # print("Vision model parameters have been frozen.")
    # Freeze language model parameters
    for param in model.language_model.parameters():
        param.requires_grad = False
    print("Language model parameters have been frozen.")
    
    if config['training']['train_itm']:
        from torch import nn
        model.tau       = nn.Parameter(torch.tensor(1.0))                           # 대조 온도
        hidden_size    = model.qformer.config.hidden_size
        model.itm_head = nn.Linear(hidden_size, 2)                                 # 이미지-텍스트 매칭 헤드
        # 3) 언어모델 파라미터는 동결 (ITC/ITM 단계에서는 generator는 건드리지 않음)
        for p in model.language_model.parameters():
            p.requires_grad = False

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 데이터셋 경로 설정
    data_dir = Path(config['data']['dir'])
    
    # 데이터셋 및 데이터로더 초기화
    
    print("train_file:", data_dir/config['data']['train_file'])
    print("valid_file:", data_dir/config['data']['valid_file'])
    train_dataset = QuIC360Dataset(
        data_dir/config['data']['train_file'], 
        processor, 
        max_length=config['data']['max_length'],
        split="train",
        image_size=config['data']['image_size'],
        do_crop=config['data']['do_crop'],
        fov=config['data']['fov'],
        overlap_ratio=config['data']['overlap_ratio']
    )
    print(f"Dataset length: {len(train_dataset)}")
    eval_dataset = QuIC360Dataset(
        data_dir/config['data']['valid_file'], 
        processor, 
        max_length=config['data']['max_length'],
        split="valid",
        image_size=config['data']['image_size'],
        do_crop=config['data']['do_crop'],
        fov=config['data']['fov'],
        overlap_ratio=config['data']['overlap_ratio']
    )
    print(f"Dataset length: {len(eval_dataset)}")

    # 학습 인자 설정 - YAML 설정을 더 정확히 반영
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        run_name=config['training']['run_name'],
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size']['train'],
        per_device_eval_batch_size=config['training']['batch_size']['eval'],
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 4),
        gradient_checkpointing=config['training'].get('gradient_checkpointing', True),
        learning_rate=float(config['training'].get('learning_rate', 2e-5)),
        warmup_ratio=config['training'].get('warmup_ratio', 0.1),
        weight_decay=config['training'].get('weight_decay', 0.01),
        max_grad_norm=config['training'].get('max_grad_norm', 1.0),
        dataloader_num_workers=config['training'].get('dataloader_num_workers', 0),
        logging_dir=config['training']['logging_dir'],
        logging_steps=config['training']['logging_steps'],
        eval_strategy=config['training']['eval_strategy'],
        eval_steps=config['training'].get('eval_steps', 500),
        save_strategy=config['training']['save_strategy'],
        save_steps=config['training'].get('save_steps', 500),
        save_total_limit=config['training'].get('save_total_limit', 3),
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training'].get('metric_for_best_model', 'eval_loss'),
        greater_is_better=config['training'].get('greater_is_better', False),
        fp16=True,  # DeepSpeed config에서 관리
        deepspeed=config['deepspeed']['config'] if config['deepspeed']['enabled'] else None,
        report_to=config['training']['report_to'],
        save_only_model=True
    )

    # 트레이너 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        # compute_metrics=compute_metrics_wrapper,
    )
  
    trainer.train()
    
    # 모델 저장
    save_dir = Path(config['model']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    
    # wandb 종료
    wandb.finish()

if __name__ == "__main__":
    main()