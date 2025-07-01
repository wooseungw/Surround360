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
import random

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
                 mode: str = "none",
                 transform: bool = False):
        super().__init__()
        
        self.df = pd.read_csv(csv_file)
        self.processor = processor
        self.max_length = max_length
        self.split = split
        self.do_crop = do_crop
        self.mode = mode.lower()
        # 데이터셋 모드 설정
        if self.do_crop:
            self.image_size = (int(image_size[0] * 2), int(image_size[1] * 4))
            self.fov = fov
            self.overlap_ratio = overlap_ratio
            print(f"Do Crop, Image size: {self.image_size}")
        else:
            self.image_size = tuple(image_size)
            print(f"Do not Crop, Image size: {self.image_size}")
        # 데이터셋 모드 설정
        if self.mode == "pretraining":
            print("Dataset in PRETRAINING mode for ITM.")
            # ITM을 위해 이미지 경로별로 텍스트 그룹화
            self.image_to_texts = self.df.groupby('url').apply(lambda x: x[['query', 'annotation']].to_dict('records')).to_dict()
            self.image_urls = list(self.image_to_texts.keys())
            print(f"Found {len(self.image_urls)} unique images for ITM.")
        else:
            print("Dataset in FINETUNING mode for VQA/Generation.")
            
        self.transform = transform
        
    def __len__(self):
        # ITM 모드일 때는 고유 이미지 수를 길이로 사용
        return len(self.image_urls) if self.mode == "pretraining" else len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        if self.mode == "pretraining":
            return self._get_itm_item(idx)
        else:
            return self._get_vqa_item(idx)

    def _process_image(self, image_path: str) -> torch.Tensor:
        """이미지 로딩부터 크롭까지의 공통 로직을 처리하는 함수"""
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Failed to load {image_path} ({e}). Returning dummy tensor.")
            # 실패 시 더미 텐서 반환 (형태를 맞춰주어야 함)
            p_dim = int(np.ceil(360.0 / (self.fov * (1.0 - self.overlap_ratio)))) if self.do_crop else 1
            h, w = self.final_patch_size if self.do_crop else self.image_size
            return torch.zeros((p_dim, 3, h, w))

        # 1. 이미지 리사이즈: Processor를 사용하여 __init__에서 정의된 크기로 변환
        # do_crop=True이면 큰 중간 크기로, False이면 최종 크기로 변환됨
        processed = self.processor.image_processor(
            images=image,
            size={"height": self.image_size[0], "width": self.image_size[1]},
            return_tensors="pt"
        )
        pixel_values = processed.pixel_values

        # 2. 크롭 또는 최종 형태 변환
        if self.do_crop:
            # (1, C, H_large, W_large) -> (P, C, H_final, W_final)
            pixel_values = self.crop_equirectangular_tensor(pixel_values)
        else:
            # (1, C, H, W) -> (P, C, H, W) 여기서 P=1
            pixel_values = pixel_values
        
        # collate를 위해 맨 앞의 배치 차원(1) 제거
        return pixel_values.squeeze(0)
    
    def _get_vqa_item(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        row = self.df.iloc[idx]
        image_path = row["url"]
        question = str(row["query"])
        answer = str(row["annotation"])
        
        pixel_values = self._process_image(image_path)
        
        prompt = f"Query: {question} Answer:"
        prompt_ids = self.processor.tokenizer(prompt, return_tensors="pt").input_ids
        
        answer_ids = self.processor.tokenizer(
            " " + answer, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
        
        labels = answer_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
        
        return {
            "pixel_values": pixel_values,
            "input_ids": prompt_ids.squeeze(0),
            "labels": labels.squeeze(0),
        }

    def _get_itm_item(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        image_path = self.image_urls[idx]
        pixel_values = self._process_image(image_path)
        
        # 1. 긍정 샘플 생성 (기존과 동일)
        associated_texts = self.image_to_texts[image_path]
        positive_sample = random.choice(associated_texts)
        positive_text = f"Query: {positive_sample['query']} Answer: {positive_sample['annotation']}"
        
        # --- 네거티브 샘플 생성 (혼용 전략 적용) ---
        negative_text = ""
        
        # 50% 확률로 하드 네거티브 샘플링 시도
        use_hard_negative = random.random() < 0.5
        
        if use_hard_negative and len(associated_texts) > 1:
            # 하드 네거티브 생성: 같은 이미지에 대한 다른 텍스트
            negative_sample = random.choice([t for t in associated_texts if t != positive_sample])
            negative_text = f"Query: {negative_sample['query']} Answer: {negative_sample['annotation']}"
        
        # 하드 네거티브를 생성하지 않은 경우 (확률에 해당하지 않거나, 텍스트가 하나뿐인 경우)
        # -> 이지 네거티브 생성
        if not negative_text:
            # 이지 네거티브 생성: 다른 이미지에서 텍스트 가져오기
            other_image_urls = [url for url in self.image_urls if url != image_path]
            if not other_image_urls:
                other_image_urls = self.image_urls
            
            negative_image_path = random.choice(other_image_urls)
            negative_sample = random.choice(self.image_to_texts[negative_image_path])
            negative_text = f"Query: {negative_sample['query']} Answer: {negative_sample['annotation']}"

        # --- 최종 데이터 구성 ---
        texts_for_itm = [positive_text, negative_text]
        labels_itm = torch.tensor([0, 1], dtype=torch.long)
        
        text_inputs = self.processor.tokenizer(
            texts_for_itm, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
            "labels_itm": labels_itm,
        }

    def crop_equirectangular_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        B, C, H2, W4 = img_tensor.shape
        assert B == 1
        H, W = self.final_patch_size

        step = self.fov * (1.0 - self.overlap_ratio)
        num_patches = int(np.ceil(360.0 / step))
        yaw_centers = (np.arange(num_patches) * step) % 360.0
        yaw_centers = np.where(yaw_centers > 180.0, yaw_centers - 360.0, yaw_centers)
        
        # GPU 텐서를 CPU의 NumPy 배열로 변환
        img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

        patches = []
        for u_deg in yaw_centers:
            pers = e2p(img_np, fov_deg=self.fov, u_deg=float(u_deg), v_deg=0.0,
                       out_hw=(H, W), in_rot_deg=0.0, mode="bilinear")
            t = torch.from_numpy(pers).permute(2, 0, 1)
            patches.append(t)
        
        # (P, C, H, W) -> (1, P, C, H, W)
        # collate를 위해 배치 차원(1)을 유지하여 반환
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