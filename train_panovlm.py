"""
PanoVLM 모델 학습 스크립트
파노라마 이미지를 처리하는 Vision-Language 모델을 학습하기 위한 코드입니다.
"""
import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.training_args import TrainingArguments
from src.models.panovlm import PanoVLM
from src.models.panovlm_config import PanoVLMConfig

# AutoTokenizer와 AutoProcessor 사용
from transformers import AutoTokenizer, AutoProcessor, Trainer
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
IGNORE_INDEX = -100

def parse_args():
    parser = argparse.ArgumentParser(description="Train PanoVLM model with parameters from a YAML file")
    parser.add_argument("--config", type=str, default="config/panovlm_train.yaml", help="Path to the config file")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print("Loaded config:", config)
    return config

class PanoVLMDataset(Dataset):
    def __init__(self, 
                 csv_file: str,
                 tokenizer: AutoTokenizer,
                 image_processor: Optional[AutoProcessor] = None,
                 image_size: list = [224, 224],
                 max_length: Optional[int] = None,
                 split: str = "train",
                 do_crop: bool = False,
                 fov: Optional[float] = None,
                 overlap_ratio: Optional[float] = None,
                 transform: bool = False):
        super().__init__()
        
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
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
        
        prompt = f"Query: {question}"
        full_text = prompt + " <image> " + "Answer: " + answer
        
        # 이미지 로드 및 처리
        image = Image.open(image_path).convert("RGB")
        
        # 이미지 처리
        if self.image_processor:
            # 이미지 프로세서가 있는 경우
            pixel_values = self.image_processor(
                image, 
                return_tensors="pt", 
                size=self.image_size
            ).pixel_values
        else:
            # 간단한 기본 전처리
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            pixel_values = transform(image).unsqueeze(0)
        
        # 텍스트 처리
        text_inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        
        # Equirectangular 이미지 크롭
        if self.do_crop:
            pixel_values = self.crop_equirectangular_tensor(pixel_values)
        
        # 라벨 생성
        labels = text_inputs.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX
        
        # 디버깅 (첫 번째 샘플에 대해서만)
        if idx == 0:
            print("==Input sequence==")
            print(text_inputs.input_ids[0])
            print(self.tokenizer.decode(text_inputs.input_ids[0], skip_special_tokens=False))
            print("==Attention mask==")
            print(text_inputs.attention_mask[0])
            print("==Labels==")
            print(labels[0])
            
        # Hugging Face Trainer가 기대하는 형태로 반환
        return {
            "pixel_values": pixel_values.squeeze(0),  # (Num Crops, C, H, W)
            "input_ids": text_inputs.input_ids.squeeze(0),  # (L1)
            "attention_mask": text_inputs.attention_mask.squeeze(0),  # (L1)
            "labels": labels.squeeze(0),  # (L2)
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
    """PanoVLM을 위한 데이터 콜레이터"""
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
        # Mask가 이미 적용된 상태로 가정
        batch["labels"] = labels
    
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
    wandb.init(project=config.get('wandb', {}).get('project', "PanoVLM"), 
               name=config.get('wandb', {}).get('name', "panovlm_training"))
    
    # PanoVLM Config 생성
    print("Creating PanoVLM config...")
    pano_config = PanoVLMConfig(
        vision_model_name_or_path=config["vision_model_name_or_path"],
        language_model_name_or_path=config["language_model_name_or_path"],
    )
    
    # 필요에 따라 projector 구성 설정
    if "projector_type" in config:
        print(f"Using projector type: {config['projector_type']}")
        from src.models.panovlm_config import ProjectorConfig
        
        # 프로젝터 구성 생성
        pano_config.projector_config = ProjectorConfig(
            type=config["projector_type"],
            in_features=config.get("projector_dim_in", 768),
            out_features=config.get("projector_dim_out", 768)
        )
    
    # PanoVLM 모델 초기화
    print("Initializing PanoVLM model...")
    model = PanoVLM(pano_config)
    
    # 프로세서 (토크나이저 및 이미지 프로세서) 초기화
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["language_model_name_or_path"], 
        trust_remote_code=True
    )
    
    # 패딩 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 이미지 프로세서 (선택적)
    image_processor = None
    if "vision_processor" in config and config["vision_processor"]:
        print(f"Loading image processor: {config['vision_processor']}")
        try:
            image_processor = AutoProcessor.from_pretrained(config["vision_processor"])
        except:
            print("Failed to load image processor, using default preprocessing")
    
    # 모델 파라미터 동결 관리 함수
    def freeze_model_parameters(model_part, model_name, freeze=True, freeze_except_layers=None):
        """
        모델 파라미터를 동결하거나 특정 레이어만 학습 가능하게 설정
        
        Args:
            model_part: 동결할 모델 부분
            model_name: 로그 출력용 모델 이름
            freeze: 동결 여부
            freeze_except_layers: 동결하지 않을 레이어 이름 목록 (전체 경로)
        """
        if not freeze:
            print(f"{model_name} 전체 파라미터가 학습 가능한 상태로 설정되었습니다.")
            return

        # 모든 파라미터 동결
        for name, param in model_part.named_parameters():
            param.requires_grad = False
            
        # 특정 레이어는 학습 가능하게 설정
        if freeze_except_layers:
            unfrozen_count = 0
            for name, param in model_part.named_parameters():
                if any(except_layer in name for except_layer in freeze_except_layers):
                    param.requires_grad = True
                    unfrozen_count += 1
            
            if unfrozen_count > 0:
                print(f"{model_name} 모델에서 {unfrozen_count}개 파라미터가 학습 가능한 상태로 설정되었습니다.")
        
        print(f"{model_name} 모델 파라미터가 동결되었습니다.")
        
        # 학습 가능한 파라미터 수 계산
        trainable_params = sum(p.numel() for p in model_part.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model_part.parameters())
        print(f"{model_name} 학습 가능한 파라미터: {trainable_params:,}/{total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # Vision 모델 파라미터 동결 여부 설정
    vision_freeze = config.get("freeze_vision", False)
    vision_freeze_except = config.get("freeze_vision_except", [])
    freeze_model_parameters(model.vision_model, "Vision Encoder", vision_freeze, vision_freeze_except)
    
    # LLM 모델 파라미터 동결 여부 설정
    language_freeze = config.get("freeze_language", True)
    language_freeze_except = config.get("freeze_language_except", [])
    freeze_model_parameters(model.language_model, "Language Model", language_freeze, language_freeze_except)
    
    # Projector 파라미터 동결 여부 설정 (옵션)
    if model.projector is not None:
        projector_freeze = config.get("freeze_projector", False)
        if projector_freeze:
            for param in model.projector.parameters():
                param.requires_grad = False
            print("Projector 파라미터가 동결되었습니다.")
        else:
            print("Projector 파라미터가 학습 가능합니다.")
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 데이터셋 경로 설정
    data_dir = Path(config.get('data', {}).get('dir', 'data'))
    
    # 데이터셋 및 데이터로더 초기화
    train_file = config.get('data', {}).get('train_file', 'train.csv')
    valid_file = config.get('data', {}).get('valid_file', 'valid.csv')
    
    print(f"Train file: {data_dir/train_file}")
    print(f"Valid file: {data_dir/valid_file}")
    
    train_dataset = PanoVLMDataset(
        data_dir/train_file,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=config.get('data', {}).get('max_length', 512),
        split="train",
        image_size=config.get('data', {}).get('image_size', [224, 224]),
        do_crop=config.get('data', {}).get('do_crop', False),
        fov=config.get('data', {}).get('fov', 90.0),
        overlap_ratio=config.get('data', {}).get('overlap_ratio', 0.2)
    )
    print(f"Train dataset length: {len(train_dataset)}")
    
    eval_dataset = PanoVLMDataset(
        data_dir/valid_file,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=config.get('data', {}).get('max_length', 512),
        split="valid",
        image_size=config.get('data', {}).get('image_size', [224, 224]),
        do_crop=config.get('data', {}).get('do_crop', False),
        fov=config.get('data', {}).get('fov', 90.0),
        overlap_ratio=config.get('data', {}).get('overlap_ratio', 0.2)
    )
    print(f"Eval dataset length: {len(eval_dataset)}")

    # 학습 인자 설정
    train_config = config.get('training', {})
    training_args = TrainingArguments(
        output_dir=train_config.get('output_dir', 'outputs/panovlm'),
        run_name=train_config.get('run_name', 'panovlm_run'),
        num_train_epochs=train_config.get('num_epochs', 3),
        per_device_train_batch_size=train_config.get('batch_size', {}).get('train', 4),
        per_device_eval_batch_size=train_config.get('batch_size', {}).get('eval', 8),
        gradient_accumulation_steps=train_config.get('gradient_accumulation_steps', 4),
        gradient_checkpointing=train_config.get('gradient_checkpointing', True),
        learning_rate=float(train_config.get('learning_rate', 2e-5)),
        warmup_ratio=train_config.get('warmup_ratio', 0.1),
        weight_decay=train_config.get('weight_decay', 0.01),
        max_grad_norm=train_config.get('max_grad_norm', 1.0),
        dataloader_num_workers=train_config.get('dataloader_num_workers', 0),
        logging_dir=train_config.get('logging_dir', 'logs/panovlm'),
        logging_steps=train_config.get('logging_steps', 100),
        eval_strategy=train_config.get('eval_strategy', 'epoch'),
        eval_steps=train_config.get('eval_steps', 500),
        save_strategy=train_config.get('save_strategy', 'epoch'),
        save_steps=train_config.get('save_steps', 500),
        save_total_limit=train_config.get('save_total_limit', 3),
        load_best_model_at_end=train_config.get('load_best_model_at_end', True),
        metric_for_best_model=train_config.get('metric_for_best_model', 'eval_loss'),
        greater_is_better=train_config.get('greater_is_better', False),
        fp16=train_config.get('fp16', True),
        deepspeed=(config.get('deepspeed', {}).get('config') 
                   if config.get('deepspeed', {}).get('enabled', False) else None),
        report_to=train_config.get('report_to', 'wandb'),
        save_only_model=True
    )

    # 학습 가능한 파라미터 요약
    def count_parameters(model):
        """모델의 전체 파라미터 수와 학습 가능한 파라미터 수를 계산"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    total, trainable = count_parameters(model)
    print(f"\n===== 모델 파라미터 요약 =====")
    print(f"총 파라미터 수: {total:,}")
    print(f"학습 가능한 파라미터 수: {trainable:,}")
    print(f"동결된 파라미터 수: {total - trainable:,}")
    print(f"학습 가능 비율: {trainable / total * 100:.2f}%")
    print(f"=============================\n")
    
    # 트레이너 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        # compute_metrics=compute_metrics_wrapper,  # 필요시 추가
    )
  
    # 학습 실행
    trainer.train()
    
    # 모델 저장
    save_dir = Path(config.get('model', {}).get('save_dir', 'saved_models/panovlm'))
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    if image_processor:
        image_processor.save_pretrained(save_dir)
    
    # wandb 종료
    wandb.finish()

if __name__ == "__main__":
    main()
