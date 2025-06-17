"""
PanoVLM 모델 학습 스크립트
파노라마 이미지를 처리하는 Vision-Language 모델을 학습하기 위한 코드입니다.
"""
import os
import torch
import inspect
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from transformers.training_args import TrainingArguments
from src.models.panovlm import PanoVLM
from src.models.panovlm_config import PanoVLMConfig

# AutoTokenizer와 AutoProcessor 사용
import transformers
from transformers import AutoTokenizer, AutoProcessor, Trainer
import pandas as pd
from PIL import Image
# 최대 픽셀 수 제한 해제 (None으로 설정)
Image.MAX_IMAGE_PIXELS = None

import wandb
from pathlib import Path

import yaml
import argparse
import inspect
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
            try:
                # 다양한 이미지 프로세서 타입 처리
                processor_type = type(self.image_processor).__name__
                
                # 1. AutoImageProcessor 또는 일반적인 이미지 프로세서 처리
                if hasattr(self.image_processor, '__call__') and not hasattr(self.image_processor, 'feature_extractor'):
                    # 일반적인 이미지 프로세서 (CLIPImageProcessor, DeiTImageProcessor 등)
                    processor_kwargs = {'return_tensors': 'pt'}
                    
                    # size 파라미터 지원 여부 확인 (모든 프로세서가 size를 지원하지는 않음)
                    if hasattr(self.image_processor, 'size') or 'size' in inspect.signature(self.image_processor.__call__).parameters:
                        processor_kwargs['size'] = self.image_size
                        
                    pixel_values = self.image_processor(image, **processor_kwargs).pixel_values
                    
                # 2. 복합 프로세서(CLIPProcessor 등)를 위한 처리
                elif hasattr(self.image_processor, 'feature_extractor'):
                    # CLIP 기반 이미지 프로세서의 feature_extractor 활용
                    processor_kwargs = {'return_tensors': 'pt'}
                    
                    # size 파라미터 지원 여부 확인
                    if hasattr(self.image_processor.feature_extractor, 'size') or 'size' in inspect.signature(self.image_processor.feature_extractor.__call__).parameters:
                        processor_kwargs['size'] = self.image_size
                        
                    pixel_values = self.image_processor.feature_extractor(image, **processor_kwargs).pixel_values
                
                # 3. 특수 케이스 처리 (필요시)
                else:
                    # 그 외의 경우에 대한 기본 처리
                    pixel_values = self.image_processor(
                        image, 
                        return_tensors="pt",
                        size=self.image_size
                    ).pixel_values
                
                if idx == 0:  # 첫 번째 샘플에 대한 디버깅
                    print(f"이미지 프로세서 타입: {processor_type}")
                    print(f"생성된 pixel_values 형태: {pixel_values.shape}")
            
            except Exception as e:
                print(f"이미지 프로세서 오류 발생: {str(e)}, 기본 전처리 사용")
                # 오류 발생 시 기본 전처리로 폴백
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                pixel_values = transform(image).unsqueeze(0)
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
        
        # 비전 모델의 차원 자동 감지
        from transformers import AutoConfig
        vision_config = AutoConfig.from_pretrained(config["vision_model_name_or_path"])
        
        # 비전 모델 차원 가져오기
        vision_dim = 384  # DinoV2-small 기본값
        if hasattr(vision_config, 'hidden_size'):
            vision_dim = vision_config.hidden_size
        elif hasattr(vision_config, 'embed_dim'):
            vision_dim = vision_config.embed_dim  # CLIP 모델
        
        # 언어 모델 차원 가져오기
        language_config = AutoConfig.from_pretrained(config["language_model_name_or_path"])
        lang_dim = 2560  # Gemma-3-4B 기본값
        if hasattr(language_config, 'hidden_size'):
            lang_dim = language_config.hidden_size
        elif hasattr(language_config, 'model_dim'):
            lang_dim = language_config.model_dim  # Gemma-3 모델
        elif hasattr(language_config, 'hidden_dim'):
            lang_dim = language_config.hidden_dim
        elif hasattr(language_config, 'd_model'):
            lang_dim = language_config.d_model  # 일부 transformer 모델
        
        print(f"Vision model dimension: {vision_dim}")
        print(f"Language model dimension: {lang_dim}")
        
        # 프로젝터 구성 생성
        pano_config.projector_config = ProjectorConfig(
            type=config["projector_type"],
            in_features=config.get("projector_dim_in", vision_dim),  # 비전 모델 차원 사용
            out_features=config.get("projector_dim_out", lang_dim)   # 언어 모델 차원 사용
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
            # 비전 모델에 맞는 프로세서 로드
            from transformers import AutoImageProcessor
            image_processor = AutoImageProcessor.from_pretrained(config["vision_processor"])
            print(f"이미지 프로세서 로드 완료: {type(image_processor).__name__}")
        except Exception as e:
            print(f"Failed to load image processor: {str(e)}")
            print("Using default preprocessing instead")
    
    # 비전 프로세서가 로드되지 않은 경우 비전 모델에서 직접 프로세서 로드 시도
    if image_processor is None and "vision_model_name_or_path" in config:
        try:
            from transformers import AutoImageProcessor
            image_processor = AutoImageProcessor.from_pretrained(config["vision_model_name_or_path"])
            print(f"비전 모델에서 이미지 프로세서 로드 완료: {type(image_processor).__name__}")
        except Exception as e:
            print(f"Failed to load image processor from vision model: {str(e)}")
            print("Using default preprocessing")
    
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
            unfrozen_count = 0
            for name, param in model_part.named_parameters():
                param.requires_grad = True
                unfrozen_count += 1
                
            print(f"{model_name} 전체 파라미터({unfrozen_count}개)가 학습 가능한 상태로 설정되었습니다.")
            
            # 학습 가능한 파라미터 수 계산 및 로깅
            trainable_params = sum(p.numel() for p in model_part.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model_part.parameters())
            print(f"{model_name} 학습 가능한 파라미터: {trainable_params:,}/{total_params:,} ({trainable_params/total_params*100:.2f}%)")
            return

        # 모든 파라미터 동결
        for name, param in model_part.named_parameters():
            param.requires_grad = False
            
        # 특정 레이어는 학습 가능하게 설정
        if freeze_except_layers:
            unfrozen_count = 0
            unfrozen_params = 0
            unfrozen_layers = []
            
            for name, param in model_part.named_parameters():
                if any(except_layer in name for except_layer in freeze_except_layers):
                    param.requires_grad = True
                    unfrozen_count += 1
                    unfrozen_params += param.numel()
                    unfrozen_layers.append(name)
            
            if unfrozen_count > 0:
                print(f"{model_name} 모델에서 {unfrozen_count}개 파라미터가 학습 가능한 상태로 설정되었습니다.")
                print(f"학습 가능한 레이어: {', '.join(unfrozen_layers)}")
        
        print(f"{model_name} 모델 파라미터가 동결되었습니다.")
        
        # 학습 가능한 파라미터 수 계산
        trainable_params = sum(p.numel() for p in model_part.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model_part.parameters())
        print(f"{model_name} 학습 가능한 파라미터: {trainable_params:,}/{total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # Vision 모델 파라미터 동결 여부 설정 (완전 동결)
    vision_freeze = config.get("freeze_vision", True)
    vision_freeze_except = config.get("freeze_vision_except", [])
    freeze_model_parameters(model.vision_model, "Vision Encoder", vision_freeze, vision_freeze_except)
    
    # LLM 모델 파라미터 동결 여부 설정 (완전 동결)
    language_freeze = config.get("freeze_language", True)
    language_freeze_except = config.get("freeze_language_except", [])
    freeze_model_parameters(model.language_model, "Language Model", language_freeze, language_freeze_except)
    
    # Projector 파라미터 동결 여부 설정 (학습 가능하게 유지)
    if model.projector is not None:
        projector_freeze = config.get("freeze_projector", False)
        
        print("\n===== Projector 상세 정보 =====")
        print(f"Projector 구조: {model.projector}")
        
        if projector_freeze:
            for param in model.projector.parameters():
                param.requires_grad = False
            print("Projector 파라미터가 동결되었습니다.")
        else:
            for param in model.projector.parameters():
                param.requires_grad = True
            
            # Projector 학습 가능한 파라미터 수 계산 및 로깅
            projector_trainable_params = sum(p.numel() for p in model.projector.parameters() if p.requires_grad)
            projector_total_params = sum(p.numel() for p in model.projector.parameters())
            
            print(f"Projector 모든 파라미터({projector_total_params:,}개)가 학습 가능한 상태로 설정되었습니다.")
            print(f"Projector 학습 가능한 파라미터: {projector_trainable_params:,}/{projector_total_params:,} ({projector_trainable_params/projector_total_params*100:.2f}%)")
            
            # 각 레이어의 파라미터 상태 출력
            print("\n----- Projector 레이어별 상태 -----")
            for name, param in model.projector.named_parameters():
                print(f"{name}: 학습 가능={param.requires_grad}, 크기={param.size()}, 파라미터 수={param.numel():,}")
        print("============================\n")
    
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
        # gradient_checkpointing 설정 (비-재진입 모드 사용)
        gradient_checkpointing=train_config.get('gradient_checkpointing', True),
        gradient_checkpointing_kwargs={"use_reentrant": False},  # 더 안정적인 모드
        torch_compile=False,  # 그래디언트 추적 문제 방지를 위해 비활성화
        learning_rate=float(train_config.get('learning_rate', 2e-5)),
        warmup_ratio=train_config.get('warmup_ratio', 0.1),
        weight_decay=train_config.get('weight_decay', 0.01),
        max_grad_norm=train_config.get('max_grad_norm', 1.0),
        # FP16 및 BF16 설정
        fp16=train_config.get('fp16', False),
        bf16=train_config.get('bf16', False),
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
    
    # 컴포넌트별 파라미터 상세 분석
    def analyze_parameters_by_component(model):
        """모델의 컴포넌트별 파라미터 상세 분석"""
        components = {
            "vision_model": model.vision_model if hasattr(model, "vision_model") else None,
            "language_model": model.language_model if hasattr(model, "language_model") else None,
            "projector": model.projector if hasattr(model, "projector") else None
        }
        
        results = {}
        
        for name, component in components.items():
            if component is None:
                continue
                
            total = sum(p.numel() for p in component.parameters())
            trainable = sum(p.numel() for p in component.parameters() if p.requires_grad)
            frozen = total - trainable
            
            results[name] = {
                "total": total,
                "trainable": trainable,
                "frozen": frozen,
                "trainable_percent": (trainable / total * 100) if total > 0 else 0
            }
            
        return results
    
    # 트레이너 실행 전 파라미터 검증 함수
    def validate_training_parameters(model):
        """학습 전 모델 파라미터 상태 검증"""
        is_valid = True
        messages = []
        
        # 비전 모델과 언어 모델 파라미터 검사
        if hasattr(model, "vision_model"):
            vision_trainable = any(p.requires_grad for p in model.vision_model.parameters())
            if vision_trainable:
                is_valid = False
                messages.append("경고: 비전 모델의 일부 파라미터가 학습 가능한 상태입니다. 비전 모델은 동결해야 합니다.")
        
        if hasattr(model, "language_model"):
            language_trainable = any(p.requires_grad for p in model.language_model.parameters())
            if language_trainable:
                is_valid = False
                messages.append("경고: 언어 모델의 일부 파라미터가 학습 가능한 상태입니다. 언어 모델은 동결해야 합니다.")
        
        # 프로젝터 파라미터 검사
        if hasattr(model, "projector"):
            projector_trainable = any(p.requires_grad for p in model.projector.parameters())
            if not projector_trainable:
                is_valid = False
                messages.append("경고: 프로젝터의 모든 파라미터가 동결 상태입니다. 프로젝터는 학습 가능해야 합니다.")
                
        return is_valid, messages
    
    # 전체 파라미터 요약
    total, trainable = count_parameters(model)
    print(f"\n===== 모델 파라미터 요약 =====")
    print(f"총 파라미터 수: {total:,}")
    print(f"학습 가능한 파라미터 수: {trainable:,}")
    print(f"동결된 파라미터 수: {total - trainable:,}")
    print(f"학습 가능 비율: {trainable / total * 100:.2f}%")
    
    # 컴포넌트별 분석
    component_analysis = analyze_parameters_by_component(model)
    print("\n----- 컴포넌트별 파라미터 분석 -----")
    for component_name, stats in component_analysis.items():
        print(f"{component_name}:")
        print(f"  총 파라미터: {stats['total']:,}")
        print(f"  학습 가능한 파라미터: {stats['trainable']:,} ({stats['trainable_percent']:.2f}%)")
        print(f"  동결된 파라미터: {stats['frozen']:,}")
    
    # 학습 설정 검증
    is_valid, messages = validate_training_parameters(model)
    print("\n----- 학습 설정 검증 -----")
    if is_valid:
        print("✅ 모든 파라미터가 올바르게 설정되었습니다. 비전 및 언어 모델은 동결되고 프로젝터만 학습 가능합니다.")
    else:
        print("⚠️ 파라미터 설정에 문제가 있습니다:")
        for msg in messages:
            print(f" - {msg}")
            
    print(f"=============================\n")
    
    # 학습 중 파라미터 모니터링을 위한 콜백 클래스
    class ParameterMonitoringCallback(transformers.TrainerCallback):
        """학습 중 파라미터 상태를 모니터링하는 콜백"""
        
        def __init__(self, model):
            self.model = model
            # 프로젝터 파라미터 초기값 저장 (변화 추적용)
            self.init_params = {}
            if hasattr(model, "projector") and model.projector is not None:
                for name, param in model.projector.named_parameters():
                    if param.requires_grad:
                        self.init_params[name] = param.data.clone().detach().cpu()
            
        def on_train_begin(self, args, state, control, **kwargs):
            """학습 시작 시 파라미터 상태 확인"""
            print("\n===== 학습 시작 시 파라미터 상태 확인 =====")
            self._log_parameter_status()
            
        def on_step_end(self, args, state, control, **kwargs):
            """일정 스텝마다 파라미터 변화 확인 (로깅 스텝의 5배 간격으로)"""
            if state.global_step > 0 and state.global_step % (args.logging_steps * 5) == 0:
                print(f"\n===== Step {state.global_step}: 파라미터 상태 확인 =====")
                self._log_parameter_status()
                self._check_param_updates()
                
        def _log_parameter_status(self):
            """모델의 현재 파라미터 상태 로깅"""
            # 비전 모델 파라미터 상태
            vision_requires_grad = any(p.requires_grad for p in self.model.vision_model.parameters())
            vision_grad_exists = any(p.grad is not None for p in self.model.vision_model.parameters() if p.requires_grad)
            
            # 언어 모델 파라미터 상태
            language_requires_grad = any(p.requires_grad for p in self.model.language_model.parameters())
            language_grad_exists = any(p.grad is not None for p in self.model.language_model.parameters() if p.requires_grad)
            
            # 프로젝터 파라미터 상태
            projector_requires_grad = any(p.requires_grad for p in self.model.projector.parameters())
            projector_grad_exists = any(p.grad is not None for p in self.model.projector.parameters() if p.requires_grad)
            
            print(f"Vision Model - requires_grad: {vision_requires_grad}, grad exists: {vision_grad_exists}")
            print(f"Language Model - requires_grad: {language_requires_grad}, grad exists: {language_grad_exists}")
            print(f"Projector - requires_grad: {projector_requires_grad}, grad exists: {projector_grad_exists}")
            
            
        def _check_param_updates(self):
            """프로젝터 파라미터가 실제로 업데이트되고 있는지 확인"""
            if not self.init_params:
                return
                
            print("\n----- 프로젝터 파라미터 변화 확인 -----")
            any_changed = False
            
            for name, param in self.model.projector.named_parameters():
                if name in self.init_params:
                    # 현재 파라미터 값
                    current = param.data.cpu()
                    # 초기 파라미터 값
                    initial = self.init_params[name]
                    
                    # 차이 계산
                    diff = torch.norm(current - initial).item()
                    max_diff = torch.max(torch.abs(current - initial)).item()
                    
                    change_status = "변화 없음" if diff < 1e-5 else "변화 있음"
                    if diff >= 1e-5:
                        any_changed = True
                        
                    print(f"{name}: {change_status} (L2 차이: {diff:.6f}, 최대 차이: {max_diff:.6f})")
            
            if any_changed:
                print("✅ 프로젝터 파라미터가 실제로 업데이트되고 있습니다.")
            else:
                print("❗ 경고: 프로젝터 파라미터가 업데이트되지 않고 있습니다!")
                
            # 더 정확한 진단을 위해 각 단계 확인
            print("\n----- 학습 진단 -----")
            if not any_changed:
                # 그래디언트 존재 여부
                has_grads = any(p.grad is not None for p in self.model.projector.parameters() if p.requires_grad)
                if not has_grads:
                    print("❌ 문제 진단: 프로젝터에 그래디언트가 없습니다.")
                    print("   가능한 원인:")
                    print("   1. 역전파가 프로젝터까지 도달하지 않음")
                    print("   2. 비전 모델에서 분리된 텐서가 그래디언트를 전달하지 않음")
                    print("   3. forward 함수에서 텐서가 detach()되었을 수 있음")
                else:
                    print("❌ 문제 진단: 그래디언트는 있지만 파라미터가 업데이트되지 않음.")
                    print("   가능한 원인:")
                    print("   1. 옵티마이저가 프로젝터 파라미터를 포함하지 않음")
                    print("   2. 학습률이 너무 낮음")
                    print("   3. 그래디언트 클리핑이 너무 강함")
                    
            # 업데이트된 초기값으로 갱신 (최근 상태 비교용)
            for name, param in self.model.projector.named_parameters():
                if param.requires_grad:
                    self.init_params[name] = param.data.clone().detach().cpu()
    
    # 커스텀 옵티마이저 클래스를 사용하여 프로젝터만 학습되도록 설정
    class ProjectorOnlyOptimizer(torch.optim.AdamW):
        """프로젝터만 학습하는 커스텀 옵티마이저"""
        
        def __init__(self, model, lr=1e-5, weight_decay=0.0, **kwargs):
            # 프로젝터 파라미터만 필터링
            projector_params = []
            param_names = []
            
            if hasattr(model, "projector") and model.projector is not None:
                # 프로젝터 파라미터 추출
                for name, param in model.projector.named_parameters():
                    if param.requires_grad:
                        projector_params.append(param)
                        param_names.append(f"projector.{name}")
            
            print(f"\n===== 프로젝터 전용 옵티마이저 초기화 =====")
            print(f"학습할 총 파라미터 수: {len(projector_params)}")
            print(f"학습할 총 파라미터 개수: {sum(p.numel() for p in projector_params):,}")
            
            if len(projector_params) == 0:
                raise ValueError("학습할 프로젝터 파라미터가 없습니다. 프로젝터가 동결되었거나 존재하지 않습니다.")
            
            # 옵티마이저 초기화
            super().__init__([{'params': projector_params}], lr=lr, weight_decay=weight_decay, **kwargs)
            
            # Vision 및 Language 모델 동결 상태 확인 (중요!)
            vision_trainable = hasattr(model, "vision_model") and any(p.requires_grad for p in model.vision_model.parameters())
            lang_trainable = hasattr(model, "language_model") and any(p.requires_grad for p in model.language_model.parameters())
            
            if vision_trainable:
                print("⚠️ 경고: Vision 모델에 학습 가능한 파라미터가 있습니다!")
                print("Vision 모델은 완전히 동결되어야 합니다.")
                
            if lang_trainable:
                print("⚠️ 경고: Language 모델에 학습 가능한 파라미터가 있습니다!")
                print("Language 모델은 완전히 동결되어야 합니다.")
            
            if not (vision_trainable or lang_trainable):
                print("✅ Vision 및 Language 모델은 올바르게 동결되었습니다.")
            
            # 스케일링 상태를 위한 키 초기화 (FP16 훈련 호환성)
            self._step_supports_amp_scaling = True
            self.state = defaultdict(dict)
                
        def step(self, closure=None):
            # AMP 스케일러와 호환되도록 하는 로직 추가
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
                    
            # 그래디언트 체크: 실제로 업데이트가 필요한 파라미터만 처리
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None and p.requires_grad:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            p.grad.zero_()
                            
            # 기본 스텝 수행
            return super().step(closure)
    
    # 커스텀 옵티마이저 생성 함수
    def projector_optimizer_factory(model):
        # 프로젝터 전용 옵티마이저 생성
        return ProjectorOnlyOptimizer(
            model=model,
            lr=float(train_config.get('learning_rate', 2e-5)),
            weight_decay=train_config.get('weight_decay', 0.01),
            eps=1e-8
        )
    
    # 트레이너 초기화 (커스텀 옵티마이저 적용)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        optimizers=(projector_optimizer_factory(model), None),  # (optimizer, scheduler)
        # compute_metrics=compute_metrics_wrapper,  # 필요시 추가
        callbacks=[ParameterMonitoringCallback(model)]
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
