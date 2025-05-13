import os

import torch
from torch.utils.data import DataLoader, Dataset
from transformers.training_args import TrainingArguments
from src.models.surroundblip import SurroundBlip
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

PAD_TOKEN_ID = 1

def parse_args():
    parser = argparse.ArgumentParser(description="Train BLIP-2 model with parameters from a YAML file")
    parser.add_argument("--config", type=str, default="config/train.yaml", help="Path to the config file")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print("Loaded config:", config)
    return config

class QuIC360Dataset(Dataset):
    def __init__(self, 
                 csv_file: str,
                 processor: Blip2Processor,
                 image_size: list = [224,224],
                 max_length: Optional[int] = None,
                 split: str = "train",
                 do_crop: bool = False,
                 fov: Optional[float] = None,
                 overlap_ratio: Optional[float] = None,
                 transform: bool = False):
        super().__init__()
        
        self.df = pd.read_csv(csv_file)
        self.processor = processor
        
        self.max_length = max_length
        self.split = split
        self.do_crop = do_crop
        if self.do_crop:
            self.image_size = (int(image_size[0] * 2), int(image_size[1] * 4))
            self.fov = fov
            self.overlap_ratio = overlap_ratio
            print(f"Do Crop, Image size: {self.image_size}")
        else:
            self.image_size = tuple(image_size)
            print(f"Do not Crop, Image size: {self.image_size}")
        self.transform = transform
        
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        # 이미지 경로와 질문, 정답을 가져옵니다.
        image_path = self.df.iloc[idx]["url"]
        question = str(self.df.iloc[idx]["query"])
        answer = str(self.df.iloc[idx]["annotation"])
        
        # 이미지를 로드합니다.
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
                text=question,
                images=image,
                size=self.image_size,
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )
        # qtext = f"Question: {question} Answer:"
        # 질문과 정답을 전처리합니다.
        if self.do_crop:
            inputs["pixel_values"] = self.crop_equirectangular_tensor(inputs["pixel_values"])
        
        # 정답을 전처리합니다.
        answers = self.processor(
            text=answer,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        
        # Hugging Face Trainer가 기대하는 평평한 구조로 반환
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),  # (Num Crops ,C, H, W)
            "input_ids": inputs["input_ids"].squeeze(0),        # (L1)
            "attention_mask": inputs["attention_mask"].squeeze(0),  # (L1)
            "labels": answers["input_ids"].squeeze(0),          # (L2)
            "image_path": image_path,
            "question": question,
            "answer": answer
        }

    def crop_equirectangular_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        B, C, H2, W4 = img_tensor.shape
        assert B == 1
        H, W = H2 // 2, W4 // 4

        # 1) stride 각도
        step = self.fov * (1.0 - self.overlap_ratio)

        # 2) 필요한 패치 개수
        num_patches = int(np.ceil(360.0 / step))

        # 3) 0도부터 시작해 step 간격으로 중심 각 생성
        yaw_centers = (np.arange(num_patches) * step) % 360.0

        # 4) e2p u_deg 인자용으로 -180~180 범위로 매핑
        yaw_centers = np.where(yaw_centers > 180.0, yaw_centers - 360.0, yaw_centers)

        # 5) numpy array 변환
        img_np = img_tensor[0].permute(1, 2, 0).numpy()

        patches = []
        for u_deg in yaw_centers:
            pers = e2p(
                img_np,
                fov_deg=self.fov,
                u_deg=float(u_deg),
                v_deg=0.0,
                out_hw=(H, W),
                in_rot_deg=0.0,
                mode="bilinear",
            )  # (H, W, C)
            t = torch.from_numpy(pers).permute(2, 0, 1)  # (C, H, W)
            patches.append(t)

        # (N, C, H, W) → (1, N, C, H, W)
        return torch.stack(patches, dim=0).unsqueeze(0)

def data_collator(features):
    """Simple data collator for BLIP2"""
    # 입력 검증
    if not features:
        raise ValueError("Features list is empty!")
    
    # 첫 번째 feature 확인
    first = features[0]
    if not isinstance(first, dict):
        raise ValueError(f"Feature is not a dict, got {type(first)}")
    
    batch = {}
    
    # 텐서 필드들은 stack
    if "pixel_values" in first:
        batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
    if "input_ids" in first:
        batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
    if "attention_mask" in first:
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
    if "labels" in first:
        # Stack labels and create a mask to ignore padding tokens
        labels = torch.stack([f["labels"] for f in features])
        # Create attention mask where pad tokens (token_id=1) are masked out with -100
        labels_mask = labels.clone()
        labels_mask[labels == PAD_TOKEN_ID] = -100  # Set pad tokens to -100 so they're ignored in loss calculation
        batch["labels"] = labels_mask
    
    # 문자열 필드들은 리스트로
    if "image_path" in first:
        batch["image_path"] = [f["image_path"] for f in features]
    if "question" in first:
        batch["question"] = [f["question"] for f in features]
    if "answer" in first:
        batch["answer"] = [f["answer"] for f in features]
    
    return batch

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
        save_optimizer=False,      # skip saving optimizer state to reduce checkpoint size
        save_scheduler=False,      # skip saving scheduler state to reduce checkpoint size
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