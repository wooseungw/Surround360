from typing import Dict, Union, Optional
import pandas as pd
from PIL import Image
# 최대 픽셀 수 제한 해제 (None으로 설정)
Image.MAX_IMAGE_PIXELS = None
from torch.utils.data import Dataset
from transformers import Blip2Processor
from py360convert import e2p
import numpy as np
import torch    
from torchvision import transforms
from copy import deepcopy

PAD_TOKEN_ID = 1
IGNORE_INDEX = -100

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
                 # --- [핵심 2] 데이터 증강을 제어하는 인자 추가 ---
                 use_augmentation: bool = True):
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
            
        # --- [핵심 3] 증강 파이프라인 정의 ---
        self.use_augmentation = use_augmentation
        if self.use_augmentation and self.split == 'train':
            print("Applying data augmentation (ColorJitter, GaussianBlur).")
            # 1단계 학습을 위한 강력한 증강 설정
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                # 필요에 따라 다른 증강 기법 추가 가능
                # 예: transforms.RandomErasing(), transforms.RandomAffine(...)
            ])
        else:
            # 학습 데이터가 아니거나, 증강을 사용하지 않을 경우
            self.transform = None
            print("Not applying data augmentation.")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        image_path = self.df.iloc[idx]["url"]
        question = str(self.df.iloc[idx]["query"])
        answer = str(self.df.iloc[idx]["annotation"])
        
        # [수정] 평가 시에는 정답을 제외한 프롬프트만 모델에 제공
        if self.split == 'train' or self.split == 'valid':
            text_to_process = f"Query: {question}###Answer: {answer}"
        else: # 'test' 등
            text_to_process = f"Query: {question}###Answer:"

        image = Image.open(image_path).convert("RGB")
        # --- [핵심 4] 정의된 증강 파이프라인 적용 ---
        # processor에 들어가기 전, PIL Image 상태에서 증강을 적용합니다.
        # --- [수정] Processor를 deepcopy하여 원본 객체 보호 ---
        self.processor = deepcopy(self.processor)
        
        # --- [수정] 별도의 transform 대신 Processor에 증강 로직 통합 ---
        # self.transform을 사용하는 기존 방식은 삭제하고 아래 로직으로 대체합니다.
        if self.transform and self.split == 'train':
            print("Applying data augmentation by modifying the processor.")
            # ToTensor와 Normalize 사이에 증강 파이프라인을 삽입합니다.
            # 이렇게 하면 텐서 기반의 안정적인 증강 함수가 사용되어 OverflowError가 발생하지 않습니다.
            self.processor.image_processor.transform.transforms.insert(
                -1, # Normalize 바로 앞에(-1) 삽입
                transforms.Compose([
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                ])
            )
        else:
            print("Not applying data augmentation.")

        # 이미지를 로드합니다.
        inputs = self.processor(
                images=image,
                text=text_to_process,
                size=self.image_size,
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )

        if self.do_crop:
            inputs["pixel_values"] = self.crop_equirectangular_tensor(inputs["pixel_values"])
        
        labels = inputs.input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
        
        # 디버깅 (첫 번째 샘플에 대해서만)
        if idx == 0:
            print("==Input sequence==")
            print(inputs["input_ids"][0])
            print(self.processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False))
            print("==Attention mask==")
            print(inputs["attention_mask"][0])
            print("==Labels==")
            print(labels[0])
            
        # Hugging Face Trainer가 기대하는 평평한 구조로 반환
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),  # (Num Crops ,C, H, W)
            "input_ids": inputs["input_ids"].squeeze(0),        # (L1)
            "attention_mask": inputs["attention_mask"].squeeze(0),  # (L1)
            "labels": labels.squeeze(0),          # (L2)
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