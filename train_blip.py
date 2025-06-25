#!/usr/bin/env python
"""
train_surround360.py – Panorama‑aware BLIP‑2 Stage‑1 trainer
-----------------------------------------------------------

• YAML 하나만 넘기면 ▶ 데이터 로드 → 학습 → 체크포인트 저장까지 전자동
• BLIP‑2(OPT 2.7B) + Q‑Former + ITC/ITM loss (논문 재현)
• Deepspeed / WandB / Gradient‑Checkpointing 모두 옵션 지원
"""

from __future__ import annotations

import argparse, os, random, math
from typing import Optional, Dict, Any
from pathlib import Path

import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import (
    Blip2Model, Blip2ForConditionalGeneration, Blip2Processor,
    TrainingArguments, Trainer,
)
from omegaconf import OmegaConf
import numpy as np, wandb
from PIL import Image; Image.MAX_IMAGE_PIXELS = None

from dataset import QuIC360Dataset, data_collator  # 사용자 정의 (동일 디렉토리)

# --------------------------- utils ---------------------------

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# -------------------- Q‑Former token resize -------------------

def resize_qformer_token(model: Blip2Model, num_q: int) -> Blip2Model:
    """쿼리 토큰 수를 늘리거나 줄인다 (기존 가중치 보존)."""
    old_tok: torch.Tensor = model.query_tokens            # (old, D)
    D = old_tok.size(-1)
    if num_q == old_tok.size(0):
        return model

    new_tok = nn.Parameter(torch.randn(num_q, D) * 0.02)
    with torch.no_grad():
        copy_n = min(num_q, old_tok.size(0))
        new_tok[:copy_n].copy_(old_tok[:copy_n])
    model.query_tokens = new_tok
    model.config.num_query_tokens = num_q
    return model

# --------------------- Stage‑1 wrapper ------------------------

class BLIP2Stage1(nn.Module):
    """BLIP‑2 Stage‑1: Image‑Text Contrastive + Matching"""

    def __init__(self, blip2: Blip2Model, proj_dim: int = 256):
        super().__init__()
        self.blip2 = blip2

        # vision / text projection 레이어 확보 혹은 생성
        def ensure_proj(attr_old: str, attr_new: str, in_dim: int):
            if hasattr(blip2, attr_old):
                return getattr(blip2, attr_old)
            if hasattr(blip2, attr_new):
                return getattr(blip2, attr_new)
            proj = nn.Linear(in_dim, proj_dim); nn.init.xavier_uniform_(proj.weight)
            setattr(blip2, attr_new, proj); return proj

        vision_hid = blip2.config.vision_config.hidden_size
        text_hid   = blip2.config.qformer_config.hidden_size
        self.vision_proj = ensure_proj("vision_proj", "vision_proj", vision_hid)
        self.text_proj   = ensure_proj("text_proj",   "text_proj",   text_hid)

        self.itm_head   = nn.Linear(proj_dim, 2)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1/0.07))

    # gradient‑checkpoint 패스스루
    def gradient_checkpointing_enable(self, **kw):
        if hasattr(self.blip2, "gradient_checkpointing_enable"):
            self.blip2.gradient_checkpointing_enable(**kw)
    def gradient_checkpointing_disable(self):
        if hasattr(self.blip2, "gradient_checkpointing_disable"):
            self.blip2.gradient_checkpointing_disable()

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, **kwargs):
        out = self.blip2(pixel_values=pixel_values,
                         input_ids=input_ids,
                         attention_mask=attention_mask,
                         return_dict=True)

        img_emb = F.normalize(self.vision_proj(out.vision_outputs.pooler_output), dim=-1)
        txt_emb = F.normalize(self.text_proj(  out.qformer_outputs.pooler_output), dim=-1)

        # ITC loss
        logit_scale = self.logit_scale.exp()
        sim = logit_scale * img_emb @ txt_emb.T                     # (B,B)
        tgt = torch.arange(sim.size(0), device=sim.device)
        loss_itc = (F.cross_entropy(sim, tgt) + F.cross_entropy(sim.T, tgt)) * 0.5

        # ITM loss (50% negatives)
        bs = img_emb.size(0)
        neg_txt = txt_emb[torch.randperm(bs, device=sim.device)]
        pair = torch.cat([txt_emb, neg_txt], 0)
        img_rep = torch.cat([img_emb, img_emb], 0)
        itm_logits = self.itm_head(img_rep * pair)
        itm_labels = torch.cat([torch.ones(bs), torch.zeros(bs)]).long().to(sim.device)
        loss_itm = F.cross_entropy(itm_logits, itm_labels)

        total_loss = loss_itc + loss_itm
        return {"loss": total_loss, "itc_loss": loss_itc.detach(), "itm_loss": loss_itm.detach()}

# --------------------------- main -----------------------------

def main(cfg):
    set_seed()

    # ---------------- WandB ----------------
    if "wandb" in cfg:
        wandb.init(project=cfg.wandb.project,
                   name=cfg.wandb.name,
                   config=OmegaConf.to_container(cfg, resolve=True))
    else:
        os.environ["WANDB_MODE"] = "disabled"

    # -------------- model / processor --------------
    processor = Blip2Processor.from_pretrained(cfg.model.pretrain_name)
    base = Blip2Model.from_pretrained(cfg.model.pretrain_name)

    base = resize_qformer_token(base, cfg.model.num_query_tokens)

    if cfg.model.qformer.hidden_size != base.config.qformer_config.hidden_size:
        # 간단 매핑: 새 Linear 로 치환
        old_hid = base.config.qformer_config.hidden_size
        mapper = nn.Linear(old_hid, cfg.model.qformer.hidden_size)
        base.qformer = mapper; base.config.qformer_config.hidden_size = cfg.model.qformer.hidden_size

    model = BLIP2Stage1(base) if cfg.training.train_itm else Blip2ForConditionalGeneration.from_pretrained(cfg.model.pretrain_name)

    # 비‑QFormer 파라미터 동결
    for n, p in model.named_parameters():
        if not n.startswith("blip2.qformer"):
            p.requires_grad = False

    # ---------------- dataset ----------------
    train_ds = QuIC360Dataset(csv_file=os.path.join(cfg.data.dir, cfg.data.train_file),
                              processor=processor,
                              image_size=cfg.data.image_size,
                              max_length=cfg.data.max_length,
                              do_crop=False)
    val_ds   = QuIC360Dataset(csv_file=os.path.join(cfg.data.dir, cfg.data.valid_file),
                              processor=processor,
                              image_size=cfg.data.image_size,
                              max_length=cfg.data.max_length,
                              do_crop=False)

    # ---------------- training args ----------------
    targs = TrainingArguments(
        output_dir=cfg.training.output_dir,
        run_name=cfg.training.run_name,
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.batch_size.train,
        per_device_eval_batch_size=cfg.training.batch_size.eval,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        fp16=True,
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=cfg.training.logging_dir,
        logging_steps=cfg.training.logging_steps,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        remove_unused_columns=False,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        max_grad_norm=cfg.training.max_grad_norm,
        report_to=[cfg.training.report_to] if "report_to" in cfg.training else [],
        deepspeed=cfg.deepspeed.config if cfg.deepspeed.enabled else None,
    )

    trainer = Trainer(model=model,
                      args=targs,
                      train_dataset=train_ds,
                      eval_dataset=val_ds,
                      data_collator=data_collator)

    trainer.train(); trainer.save_model(cfg.model.save_dir)
    if wandb.run is not None: wandb.finish()

# ------------------------- entry ------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--cfg", required=True)
    CFG = OmegaConf.load(p.parse_args().cfg)
    main(CFG)
