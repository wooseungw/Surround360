from transformers import Trainer
from ..loss.blip2_loss import compute_itc_loss, compute_itm_loss

import torch
import torch.nn.functional as F
def extract_image_embeds(model, pixel_values, 
                         output_attentions=False, 
                         output_hidden_states=False, 
                         return_dict=True, 
                         interpolate_pos_encoding=False):
        """
        SurroundBlip 의 (B,P,C,H,W) pixel_values → 
        (B, D) 형태의 이미지 임베딩으로 변환
        """
        B, P, C, H, W = pixel_values.shape
        # 1) (B*P, C, H, W)
        pix = pixel_values.view(B * P, C, H, W)
        
        # 2) Vision Encoder
        vision_out = model.vision_model(
            pixel_values=pix,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        # vision_out.last_hidden_state: (B*P, S, D)
        hidden = vision_out.last_hidden_state
        S, D   = hidden.shape[1], hidden.shape[2]
        
        # 3) (B, P*S, D)
        hidden = hidden.view(B, P * S, D)
        # attention mask for P*S patches
        attn_mask = torch.ones((B, P * S), device=hidden.device, dtype=torch.long)
        
        # 4) Q-Former 에 멀티뷰 패치 주입
        #    query_tokens: (1, Q, D_q) → expand → (B, Q, D_q)
        query = model.query_tokens.expand(B, -1, -1)
        q_out = model.qformer(
            query_embeds=query,
            encoder_hidden_states=hidden,
            encoder_attention_mask=attn_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # q_out.last_hidden_state: (B, Q, D_q)
        qfeat = q_out.last_hidden_state
        
        # 5) Linear projection to text hidden space
        img_proj = model.language_projection(qfeat)  # (B, Q, D_text)
        
        # 6) Query 차원 평균 풀링 → (B, D_text)
        image_embeds = F.normalize(img_proj.mean(dim=1), dim=-1)
        return image_embeds
    
class ContrastiveMatchingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # (A) 언어모델 손실
        outputs = model(
            pixel_values   = inputs["pixel_values"].to(model.device),
            input_ids      = inputs["input_ids"].to(model.device),
            attention_mask = inputs["attention_mask"].to(model.device),
            labels         = inputs["labels"].to(model.device),
        )
        lm_loss = outputs.loss

        # (B) 이미지 임베딩 추출 (ITC 용)
        image_feats = extract_image_embeds(model, inputs["pixel_values"].to(model.device))
        
        # (C) 텍스트 임베딩 추출 (ITC 용)
        text_out = model.get_text_features(
            input_ids      = inputs["labels"].to(model.device),
            attention_mask = (inputs["labels"] != -100).long().to(model.device),
            return_dict    = True
        )
        # encoder–decoder 모델의 [CLS] 토큰 풀링
        text_feats = F.normalize(text_out.last_hidden_state[:,0,:], dim=-1)

        # (D) ITC 손실
        itc_loss = compute_itc_loss(image_feats, text_feats, model.tau)

        # (E) ITM 손실
        q_out     = model.qformer(
            query_embeds           = model.query_tokens.expand(image_feats.size(0), -1, -1),
            encoder_hidden_states  = image_feats,
        )
        query_feats = q_out.last_hidden_state[:,0,:]
        itm_logits  = model.itm_head(query_feats)
        itm_labels  = inputs.get("itm_labels",
                                 torch.ones(query_feats.size(0), dtype=torch.long, device=model.device))
        itm_loss    = compute_itm_loss(itm_logits, itm_labels)

        # (F) 종합 손실
        loss = lm_loss + itc_loss + itm_loss
        return (loss, outputs) if return_outputs else loss
