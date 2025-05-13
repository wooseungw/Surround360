from transformers import Trainer
from ..loss.blip2_loss import compute_itc_loss, compute_itm_loss

import torch
import torch.nn.functional as F

import torch.nn.functional as F
from transformers import Trainer

class ContrastiveMatchingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):
        # === (A) 언어모델 손실 ===
        outputs = model(
            pixel_values  = inputs["pixel_values"].to(model.device),
            input_ids     = inputs["input_ids"].to(model.device),
            attention_mask= inputs["attention_mask"].to(model.device),
            labels        = inputs["labels"].to(model.device),
        )
        lm_loss = outputs.loss

        # === (B) ITC 손실 ===
        vision_out = model.get_image_features(
            pixel_values=inputs["pixel_values"].to(model.device),
            return_dict=True
        )
        image_feats = F.normalize(vision_out.pooler_output, dim=-1)  # (B, D)

        text_out = model.get_text_features(
            input_ids=inputs["labels"].to(model.device),
            attention_mask=(inputs["labels"]!=-100).long().to(model.device),
            return_dict=True
        )
        # encoder–decoder 모델 가정: 첫 토큰 풀링
        text_feats  = F.normalize(text_out.last_hidden_state[:,0,:], dim=-1)  # (B, D)

        itc_loss = compute_itc_loss(image_feats, text_feats, model.tau)

        # === (C) ITM 손실 ===
        qformer_out = model.get_qformer_features(
            pixel_values=inputs["pixel_values"].to(model.device),
            return_dict=True
        )
        query_feats = qformer_out.last_hidden_state[:,0,:]                  # (B, H)
        itm_logits  = model.itm_head(query_feats)                           # (B, 2)

        # 만약 Dataset에서 itm_labels 준비했다면:
        itm_labels = inputs.get("itm_labels", 
                                torch.ones(query_feats.size(0), dtype=torch.long, device=model.device))
        itm_loss = compute_itm_loss(itm_logits, itm_labels)

        # === (D) 전체 손실 조합 ===
        alpha, beta = 1.0, 1.0
        loss = lm_loss + alpha * itc_loss + beta * itm_loss

        return (loss, outputs) if return_outputs else loss
