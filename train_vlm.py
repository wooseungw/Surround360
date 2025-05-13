import os
import argparse
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
    AutoProcessor,
    AutoTokenizer,
)
from torch.utils.data import Dataset
from copy import deepcopy
# from peft import LoraConfig, TaskType, get_peft_model

import yaml
from omegaconf import OmegaConf
from types import SimpleNamespace
from typing import Dict, Union, Optional
import numpy as np
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from py360convert import e2p

from src.models.config import VisionLanguageConfig
from src.models.build import CustomVLMModel
PAD_TOKEN_ID = 1

IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100
# ---------------------------------------------------------------------------- #
# 1. Dataset for Images & Video Frames
# ---------------------------------------------------------------------------- #

class QuIC360Dataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        image_processor: AutoProcessor,
        tokenizer: AutoTokenizer,
        image_size: tuple[int, int] = (224, 224),
        max_length: int = 256,
        split: str = "train",
        do_crop: bool = False,
        fov: float | None = None,
        overlap_ratio: float | None = None,
        transform: bool = False,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        self.do_crop = do_crop
        self.image_size = (
            (int(image_size[0] * 2), int(image_size[1] * 4)) if do_crop else image_size
        )
        print(f"image_size: {self.image_size}")
        self.fov, self.overlap_ratio = fov, overlap_ratio
        self.transform = transform

        # pad_token 미정의 LLM 대비용
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[{split}] img_size={self.image_size}  do_crop={self.do_crop}")

    # ──────────────────────────────────────
    def __len__(self):
        return len(self.df)

    # ──────────────────────────────────────
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        rec = self.df.iloc[idx]
        img_path: str = rec["url"]
        question: str = str(rec["query"])
        answer: str = str(rec["annotation"])

        # 1) 이미지 전처리 → pixel_values: (1, C, H, W)
        image = Image.open(img_path).convert("RGB")
        pixel_dict = self.image_processor(
            images=image, 
            do_resize=True, 
            size={"height": self.image_size[0], "width": self.image_size[1]},
            return_tensors="pt",
            do_center_crop=False        # crop_size 무시
        )
        # print("pixel_dict:", pixel_dict.keys())
        # print(pixel_dict["pixel_values"].shape)  # (1, C, H, W)

        # (선택) equirectangular crop
        if self.do_crop:
            pixel_dict["pixel_values"] = self.crop_equirectangular_tensor(
                pixel_dict["pixel_values"]
            )  # (B, C, H, W) or (B, T, C, H, W)

        # 2) 텍스트 시퀀스 구성:  [question] <image>  [answer]  <eos>
        prompt = f"{question} {IMAGE_TOKEN}"
        full_text = prompt + " " + answer

        tok = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 라벨: question+<image> 영역 마스킹
        labels = tok.input_ids.clone()
        q_len = len(self.tokenizer(prompt).input_ids)  # 질문+<image> token 개수
        labels[:, :q_len] = IGNORE_INDEX
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX

        print("pixel_values:", pixel_dict["pixel_values"].shape)
        return {
            # vision
            "pixel_values": pixel_dict["pixel_values"],  # (C, H, W)
            # text
            "input_ids": tok.input_ids.squeeze(0),
            "attention_mask": tok.attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
            # 로그/디버깅용 부가 정보
            "image_path": img_path,
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
        return torch.stack(patches, dim=0)

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


# ---------------------------------------------------------------------------- #
# 3. Argument Parser
# ---------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Train VLM with parameters from YAML")
    parser.add_argument("--config", type=str, default="config/train_vlm_baseline.yaml", help="Path to the YAML config")
    return parser.parse_args()

def load_config(config_path: str):
    """
    Load YAML into an OmegaConf DictConfig so we can keep dot‑access,
    then resolve any {model_name} placeholders.
    """
    cfg = OmegaConf.load(config_path)

    # Resolve {model_name} templates
    model_name = cfg.model.name
    cfg.training.output_dir = cfg.training.output_dir.format(model_name=model_name)
    cfg.training.run_name   = cfg.training.run_name.format(model_name=model_name)
    cfg.training.logging_dir = cfg.training.logging_dir.format(model_name=model_name)
    # Backward compatibility: allow config with only "data:" section
    if "dataset" not in cfg and "data" in cfg:
        cfg.dataset = cfg.data  # backward compatibility alias
    return cfg
# ---------------------------------------------------------------------------- #
# 4. Main training flow
# ---------------------------------------------------------------------------- #
def main():
    args = parse_args()
    cfg = load_config(args.config)
    # Config & Processor
    model_config = VisionLanguageConfig(
        vision_model_name=cfg.model.vision_model_name,
        language_model_name=cfg.model.llm_model_name,
        projector_type=cfg.model.projector_type,
        use_resampler=cfg.model.use_resampler,
        mm_spatial_pool_mode=cfg.model.mm_spatial_pool_mode,
        mm_newline_position=getattr(cfg.model, "mm_newline_position", "grid"),
        freeze_vision=cfg.model.freeze_vision,
        freeze_llm=cfg.model.freeze_llm,
    )

    # Load the model
    model = CustomVLMModel(model_config)
    vision_processor = AutoProcessor.from_pretrained(cfg.model.vision_model_name)
    language_processor = deepcopy(model.tokenizer)

    # Dataset -----------------------------------------------------------------
    ds_cfg = cfg.dataset
    train_ds = QuIC360Dataset(
        csv_file=ds_cfg.train_csv,
        image_processor=vision_processor,
        tokenizer = language_processor,
        image_size=ds_cfg.image_size,
        max_length=ds_cfg.max_length,
        split="train",
        do_crop=ds_cfg.do_crop,
        fov=ds_cfg.fov,
        overlap_ratio=ds_cfg.overlap_ratio,
    )
    valid_ds = QuIC360Dataset(
        csv_file=ds_cfg.valid_csv,
        image_processor=vision_processor,
        tokenizer = language_processor,
        image_size=ds_cfg.image_size,
        max_length=ds_cfg.max_length,
        split="valid",
        do_crop=ds_cfg.do_crop,
        fov=ds_cfg.fov,
        overlap_ratio=ds_cfg.overlap_ratio,
    )


    # Move model to GPU explicitly (Trainer would do this as well, but we place it early for PEFT initialization safety)
    if torch.cuda.is_available():
        model.cuda()

    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        run_name=cfg.training.run_name,
        logging_dir=cfg.training.logging_dir,
        deepspeed=cfg.deepspeed.config if "deepspeed" in cfg and cfg.deepspeed.enabled else None,

        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.batch_size.train,
        
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        gradient_checkpointing=False,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        dataloader_num_workers=cfg.training.dataloader_num_workers,

        per_device_eval_batch_size=cfg.training.batch_size.eval,
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps,

        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=False,   # we only keep the last N checkpoints

        report_to=cfg.training.report_to,
        logging_steps=cfg.training.logging_steps,
        max_grad_norm=cfg.training.max_grad_norm,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
    )

    trainer.train()

# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
