from transformers import Blip2Processor, Blip2Model
import torch.nn as nn, torch, torch.nn.functional as F

from transformers import TrainingArguments, Trainer

from dataset import QuIC360Dataset, data_collator  # Assuming dataset.py is in the same directory
#!/usr/bin/env python
"""
train_surround360.py  – Panorama-aware BLIP-2 (Surround360) trainer

▶ 사용법
$ python train_surround360.py --cfg config/surround360.yaml

- YAML 구성 파일 하나만 있으면 전체 학습 파이프라인이 구동됩니다.
- Deepspeed / Accelerate / WandB 연동까지 모두 옵션으로 처리합니다.

필요 패키지
-------------
conda create -n surround360 python=3.10
conda activate surround360
pip install "transformers[torch] >=4.52.3" datasets wandb timm einops omegaconf pillow accelerate deepspeed

(⚠️ Deepspeed 는 CUDA11+ 환경 필요)
"""
import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Blip2Model,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
)
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import wandb
from PIL import Image
# 최대 픽셀 수 제한 해제 (None으로 설정)
Image.MAX_IMAGE_PIXELS = None
from einops import rearrange

# -----------------------
# Stage‑1 모델 (ITC+ITM)
# -----------------------
class BLIP2Stage1(nn.Module):
    def __init__(self, blip2: Blip2Model, proj_dim: int = 256):
        super().__init__()
        self.blip2 = blip2

        # ───────────────────────────────────────────────
        # 1. vision/text projection 레이어 확보
        # ───────────────────────────────────────────────
        def get_or_create(name_old, name_new, in_dim):
            # ① 이미 존재?
            if hasattr(blip2, name_old):
                return getattr(blip2, name_old)
            if hasattr(blip2, name_new):
                return getattr(blip2, name_new)

            # ② 없으면 새로 만든 뒤 setattr
            layer = nn.Linear(in_dim, proj_dim)
            nn.init.xavier_uniform_(layer.weight)
            setattr(blip2, name_new, layer)      # 모델에 추가
            return layer

        # in_dim 추정
        vision_hid = blip2.config.vision_config.hidden_size
        text_hid   = blip2.config.qformer_config.hidden_size

        self.vision_proj = get_or_create("vision_proj", "vision_proj", vision_hid)
        self.text_proj   = get_or_create("text_proj",   "text_proj",   text_hid)

        # ───────────────────────────────────────────────
        self.itm_head   = nn.Linear(proj_dim, 2)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    # Trainer 콜백용
    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.blip2, "gradient_checkpointing_enable"):
            self.blip2.gradient_checkpointing_enable(**kwargs)
    def gradient_checkpointing_disable(self):
        if hasattr(self.blip2, "gradient_checkpointing_disable"):
            self.blip2.gradient_checkpointing_disable()

    def forward(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        **kwargs
    ):
        # ---------------------------------------------
        # ① BLIP-2 인코더
        # ---------------------------------------------
        out = self.blip2(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # ---------------------------------------------
        # ② 256-D 공통 임베딩
        # ---------------------------------------------
        img_emb = self.blip2.vision_proj(out.vision_outputs.pooler_output)      # (B, 256)
        txt_emb = self.blip2.text_proj(  out.qformer_outputs.pooler_output)     # (B, 256)

        img_emb = F.normalize(img_emb, dim=-1)
        txt_emb = F.normalize(txt_emb, dim=-1)

        # ---------------------------------------------
        # ③ ITC loss
        # ---------------------------------------------
        logit_scale = self.logit_scale.exp()
        logits_per_img = logit_scale * img_emb @ txt_emb.T          # (B, B)
        targets = torch.arange(logits_per_img.size(0), device=img_emb.device)
        loss_itc = (F.cross_entropy(logits_per_img, targets) +
                    F.cross_entropy(logits_per_img.T, targets)) * 0.5

        # ---------------------------------------------
        # ④ ITM loss
        #    50% negative: txt_emb permuted
        # ---------------------------------------------
        bs = img_emb.size(0)
        neg_txt = txt_emb[torch.randperm(bs, device=img_emb.device)]
        pair_emb = torch.cat([img_emb, img_emb], dim=0)             # (2B, 256)
        txt_pair = torch.cat([txt_emb, neg_txt], dim=0)             # (2B, 256)
        itm_logits = self.itm_head(pair_emb * txt_pair)             # simple fusion
        itm_labels = torch.cat([torch.ones(bs), torch.zeros(bs)]).long().to(img_emb.device)
        loss_itm = F.cross_entropy(itm_logits, itm_labels)

        total_loss = loss_itc + loss_itm          # (0-D Tensor)
        return {
        "loss": total_loss,
        "itc_loss": loss_itc.detach(),        # 로깅용 (그래프 분리)
        "itm_loss": loss_itm.detach(),
        }
# -----------------------
# 헬퍼: Q‑Former 확장
# -----------------------

def resize_qformer_token(model: Blip2Model, num_query_tokens: int, hidden_size: Optional[int] = None) -> Blip2Model:
    old_tok = model.query_tokens  # (old_n, D)
    d = old_tok.size(-1)
    if num_query_tokens == old_tok.size(0):
        return model
    new_tok = nn.Parameter(torch.zeros(1, num_query_tokens, hidden_size))
    with torch.no_grad():
        new_tok[: old_tok.size(0)] = old_tok
    model.query_tokens = new_tok
    model.config.num_query_tokens = num_query_tokens
    return model

# -----------------------
# main
# -----------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(cfg):
    set_seed()

    # WandB 초기화
    wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, config=OmegaConf.to_container(cfg, resolve=True))

    # 모델 & 프로세서 로드
    processor = Blip2Processor.from_pretrained(cfg.model.pretrain_name)
    base_model = Blip2Model.from_pretrained(cfg.model.pretrain_name)

    # Q‑Former 토큰/hidden 조정
    base_model = resize_qformer_token(base_model, cfg.model.num_query_tokens, cfg.model.qformer.hidden_size)
    if cfg.model.qformer.hidden_size != base_model.config.qformer_config.hidden_size:
        base_model.qformer = nn.Linear(base_model.config.qformer_config.hidden_size, cfg.model.qformer.hidden_size)  # 간단 매핑, 상세 구현 필요
        base_model.config.qformer_config.hidden_size = cfg.model.qformer.hidden_size

    # Loss 모드 선택
    if cfg.training.train_itm:
        model = BLIP2Stage1(base_model)
    else:
        model = Blip2ForConditionalGeneration.from_pretrained(cfg.model.pretrain_name)

    # 파라미터 동결 (논문 동일)
    for n, p in model.named_parameters():
        if not n.startswith("blip2.qformer"):
            p.requires_grad = False

    # 데이터셋
    train_dataset = QuIC360Dataset(
        csv_file=os.path.join(cfg.data.dir, cfg.data.train_file),
        processor=processor,
        image_size=cfg.data.image_size,
        max_length=cfg.data.max_length,
        do_crop=False,
        fov=cfg.data.fov,
        overlap_ratio=cfg.data.overlap_ratio,
    )
    valid_dataset = QuIC360Dataset(
        csv_file=os.path.join(cfg.data.dir, cfg.data.valid_file),
        processor=processor,
        image_size=cfg.data.image_size,
        max_length=cfg.data.max_length,
        do_crop=False,  # 검증에서는 랜덤 크롭 X
        fov=cfg.data.fov,
        overlap_ratio=cfg.data.overlap_ratio,
    )

    # TrainingArguments
    targs = TrainingArguments(
        output_dir=cfg.training.output_dir,
        run_name=cfg.training.run_name,
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.batch_size.train,
        per_device_eval_batch_size=cfg.training.batch_size.eval,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        fp16=True,
        weight_decay=cfg.training.weight_decay,
        logging_dir=cfg.training.logging_dir,
        logging_steps=cfg.training.logging_steps,
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        report_to=[cfg.training.report_to],
        remove_unused_columns=False,  # image tensor 유지
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        max_grad_norm=cfg.training.max_grad_norm,
        deepspeed=cfg.deepspeed.config if cfg.deepspeed.enabled else None,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(cfg.model.save_dir)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    main(cfg)
