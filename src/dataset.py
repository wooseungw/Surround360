from typing import Dict, Union, Optional, List
import pandas as pd
import numpy as np
from PIL import Image
# 최대 픽셀 수 제한 해제 (None으로 설정)
Image.MAX_IMAGE_PIXELS = None
from copy import deepcopy
import torch
from torch.utils.data import Dataset
from transformers import Blip2Processor
from torchvision import transforms
from py360convert import e2p

# from py360convert import e2p # 필요시 활성화

# 상수 정의
PAD_TOKEN_ID = 1  # BLIP-2의 패딩 토큰 ID
IGNORE_INDEX = -100

class QuIC360Dataset(Dataset):
    """
    모든 모범 사례를 적용한 최종 데이터셋 클래스.
    - Processor 통합 데이터 증강
    - 정교한 레이블 마스킹
    - 이미지 로딩 에러 처리
    """
    def __init__(self,
                 csv_file: str,
                 processor: Blip2Processor,
                 split: str = "train",
                 max_length: Optional[int] = 128,
                 image_size: List[int] = [224, 224],
                 do_crop: bool = False,
                 fov: Optional[float] = 90.0,
                 overlap_ratio: Optional[float] = 0.5):
        super().__init__()
        
        self.df = pd.read_csv(csv_file)
        self.split = split
        self.max_length = max_length
        self.do_crop = do_crop
        
        # [핵심 개선 1] 별도의 transform 대신 Processor에 증강 로직 통합
        self.processor = deepcopy(processor) # 원본 processor 보호
        # if self.split == 'train':
        #     print("Applying data augmentation by modifying the processor's transform pipeline.")
        #     # ToTensor와 Normalize 사이에 증강 파이프라인 삽입 (텐서 기반의 안정적 증강)
        #     self.processor.image_processor.transform.transforms.insert(
        #         -1, # Normalize 바로 앞에 삽입
        #         transforms.Compose([
        #             transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        #             transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        #         ])
        #     )
        # else:
        #     print(f"Not applying data augmentation for '{self.split}' split.")

        # 이미지 크기 설정 (기존과 동일)
        self.pers_image_size = tuple(image_size)
        if self.do_crop:
            self.eq_image_size = (int(image_size[0] * 2), int(image_size[1] * 4))
            self.fov = fov
            self.overlap_ratio = overlap_ratio
        else:
            self.eq_image_size = self.pers_image_size
        
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Union[torch.Tensor, str]]]:
        try:
            row = self.df.iloc[idx]
            image_path = row["url"]
            question = str(row["query"])
            answer = str(row.get("annotation", "")) # 테스트셋에 정답이 없을 수 있음
            
            # [추가] 이미지 ID 추출 - CSV에 image_id 컬럼이 있으면 사용, 없으면 경로에서 추출
            if "image_id" in row:
                image_id = str(row["image_id"])
            else:
                # 이미지 경로에서 파일명을 image_id로 사용
                import os
                image_id = os.path.splitext(os.path.basename(image_path))[0]
            
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # [핵심 개선 2] 이미지 로딩 실패 시 에러를 내지 않고 해당 샘플을 건너뜀
            print(f"Warning: Could not load data at index {idx}, path {image_path}. Skipping sample. Error: {e}")
            return None

        # 프롬프트와 정답 텍스트 준비
        prompt_text = f"Query: {question}###Answer: "
        if self.split in ['train', 'valid']:
            answer_text = f"{answer}{self.processor.tokenizer.eos_token}"
        else:
            answer_text = ""

        # Processor가 이미지 처리(증강 포함)와 텍스트 토큰화를 모두 수행
        inputs = self.processor(
            images=image,
            text=prompt_text + answer_text,
            size={"height": self.eq_image_size[0], "width": self.eq_image_size[1]},
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        # [핵심 개선 3] 정교한 레이블 마스킹
        # 모델이 질문(prompt) 부분은 학습하지 않고 정답(answer) 부분만 학습하도록 레이블 생성
        labels = inputs.input_ids.clone()
        if self.split in ['train', 'valid']:
            # 프롬프트 부분에 해당하는 토큰들을 IGNORE_INDEX(-100)로 마스킹
            prompt_tokens = self.processor.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).input_ids.shape[1]
            # [CLS] 같은 스페셜 토큰을 고려하여, 실제 프롬프트 길이 직전까지 마스킹
            labels[:, :prompt_tokens - 1] = IGNORE_INDEX
        else:
            labels[:] = IGNORE_INDEX
            
        # 패딩 토큰도 마스킹
        labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX

        # crop 로직은 텐서 기반이므로 그대로 사용
        if self.do_crop:
            inputs["pixel_values"] = self.crop_equirectangular_tensor(inputs["pixel_values"])
            
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "image_path": image_path,
            "image_id": image_id,  # [추가] 이미지 ID
            "question": question,
            "answer": answer,
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
    
    # [추가] image_ids를 텐서로 변환 (문자열을 해시값으로 변환)
    if "image_id" in first:
        image_ids = [f["image_id"] for f in features]
        # 문자열 image_id를 정수로 변환 (같은 이미지는 같은 값)
        unique_ids = list(set(image_ids))
        id_to_int = {img_id: i for i, img_id in enumerate(unique_ids)}
        batch["image_ids"] = torch.tensor([id_to_int[img_id] for img_id in image_ids], dtype=torch.long)
    
    # 문자열 필드들은 리스트로
    if "image_path" in first:
        batch["image_path"] = [f["image_path"] for f in features]
    if "question" in first:
        batch["question"] = [f["question"] for f in features]
    if "answer" in first:
        batch["answer"] = [f["answer"] for f in features]
    
    return batch