import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Blip2Processor, Blip2ForConditionalGeneration, TrainingArguments, Trainer
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
import wandb
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
import yaml
import argparse
from typing import Dict, List, Optional, Union


def parse_args():
    parser = argparse.ArgumentParser(description="Train BLIP-2 model with parameters from a YAML file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# 데이터셋 클래스 정의
class QuIC360Dataset(Dataset):
    def __init__(self, 
                 csv_file: str,
                 processor: Blip2Processor,
                 max_length: int = 128,
                 split: str = "train",
                 image_size: tuple = (224, 224)
                ):
        self.df = pd.read_csv(csv_file)
        self.processor = processor
        self.max_length = max_length
        self.split = split
        self.image_size = image_size
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        url = row['url']
        query = row['query']
        caption = row['annotation']  # 두 번째 열에 캡션이 있는 것으로 가정
        
        # 이미지 다운로드 및 로드
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        
        # 이미지와 텍스트 인코딩
        inputs = self.processor(
            images=image, 
            text=query,
            size=self.image_size,
            text=caption, 
            do_crop=True,
            overlap_ratio=0.5,
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        # 배치 차원 제거
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs
        

# 데이터 콜레이터 정의
def data_collator(batch):
    input_ids = torch.stack([example["input_ids"] for example in batch])
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    attention_mask = torch.stack([example["attention_mask"] for example in batch])
    
    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "attention_mask": attention_mask,
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # wandb 설정
    wandb.init(project=config['wandb']['project'], name=config['wandb']['name'])
    
    # BLIP-2 모델 및 프로세서 로드
    model_name = config['model']['name']
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.float16 if config['model']['fp16'] else torch.float32
    )
    # Freeze vision encoder parameters
    for param in model.vision_model.parameters():
        param.requires_grad = False
    print("Vision model parameters have been frozen.")
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
    image_size = tuple(config['data']['image_size'])
    max_length = config['data']['max_length']
    
    train_dataset = QuIC360Dataset(
        data_dir/config['data']['train_file'], 
        processor, 
        max_length=max_length, 
        split="train",
        image_size=image_size
    )
    
    eval_dataset = QuIC360Dataset(
        data_dir/config['data']['valid_file'], 
        processor, 
        max_length=max_length, 
        split="valid",
        image_size=image_size
    )
    
    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size']['train'],
        per_device_eval_batch_size=config['training']['batch_size']['eval'],
        warmup_steps=config['training']['warmup_steps'],
        weight_decay=config['training']['weight_decay'],
        logging_dir=config['training']['logging_dir'],
        logging_steps=config['training']['logging_steps'],
        evaluation_strategy=config['training']['evaluation_strategy'],
        save_strategy=config['training']['save_strategy'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        report_to=config['training']['report_to'],
        fp16=config['training']['fp16'],
    )
    
    # 트레이너 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # 모델 학습
    trainer.train()
    
    # 모델 저장
    save_dir = Path(config['model']['save_dir'])
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    
    # wandb 종료
    wandb.finish()


if __name__ == "__main__":
    main()