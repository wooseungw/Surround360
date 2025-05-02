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

# wandb 설정
wandb.init(project="surround360-blip2", name="blip2-quic360")

# BLIP-2 모델 및 프로세서 로드
model_name = "Salesforce/blip2-opt-2.7b"  # 또는 다른 BLIP-2 모델 선택 가능
processor = Blip2Processor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_name, 
    torch_dtype=torch.float16  # 메모리 사용량 감소를 위해 fp16 사용
)

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 데이터셋 경로 설정
data_dir = Path("data/raw/QuIC360")

# 커스텀 데이터셋 클래스 정의
class QuIC360Dataset(Dataset):
    def __init__(self, csv_file, processor, max_length=30, split="train"):
        self.df = pd.read_csv(csv_file)
        self.processor = processor
        self.max_length = max_length
        self.split = split
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        url = row['url']
        caption = row[1]  # 두 번째 열에 캡션이 있는 것으로 가정
        
        # 이미지 다운로드 및 로드
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # 이미지와 텍스트 인코딩
            inputs = self.processor(
                images=image, 
                text=caption, 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors="pt"
            )
            
            # 배치 차원 제거
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            return inputs
            
        except Exception as e:
            print(f"Error loading image from {url}: {e}")
            # 에러 발생시 임의의 검은색 이미지와 빈 캡션 반환
            dummy_image = Image.new('RGB', (224, 224), color='black')
            inputs = self.processor(
                images=dummy_image, 
                text="", 
                padding="max_length", 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors="pt"
            )
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
        "labels": input_ids.clone(),  # 자기회귀 학습을 위해 입력을 레이블로 사용
    }

# 데이터셋 및 데이터로더 초기화
train_dataset = QuIC360Dataset(data_dir/"train.csv", processor, max_length=30, split="train")
eval_dataset = QuIC360Dataset(data_dir/"valid.csv", processor, max_length=30, split="valid")

# 학습 인자 설정
training_args = TrainingArguments(
    output_dir="./results/blip2",
    num_train_epochs=3,
    per_device_train_batch_size=4,  # 메모리에 따라 조정
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",  # wandb로 로깅
    fp16=True,  # 메모리 사용량 감소를 위해 fp16 사용
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
model.save_pretrained("./saved_model/blip2")
processor.save_pretrained("./saved_model/blip2")

# wandb 종료
wandb.finish()