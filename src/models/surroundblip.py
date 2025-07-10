from typing import Any, Optional, Tuple, Dict

import torch
from torch import nn
import torch.nn.functional as F

from transformers import Blip2Config, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.blip_2.modeling_blip_2 import Blip2PreTrainedModel, Blip2VisionModel
from transformers import BertLMHeadModel # Blip2QFormerModel 대신 사용
from transformers.utils import logging

from ..loss.vicreg import VICRegLoss # 사용자 정의 Loss, 경로는 실제 위치에 맞게 수정

logger = logging.get_logger(__name__)


class SurroundBlip(Blip2PreTrainedModel, GenerationMixin):
    """
    BLIP-2 아키텍처를 기반으로 다단계 학습(Vision-pretrain, Q-Former-pretrain, Finetune)을
    지원하는 커스텀 모델.
    - Q-Former로 BertLMHeadModel을 사용하여 텍스트 인코딩 및 멀티모달 융합 수행.
    """
    config_class = Blip2Config
    main_input_name = "pixel_values"

    def __init__(self, config: Blip2Config):
        super().__init__(config)
        
        # 1. Vision Encoder
        self.vision_model = Blip2VisionModel(config.vision_config)

        # 2. Q-Former (BertLMHeadModel로 교체)
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = BertLMHeadModel(config.qformer_config) # ✨ 핵심: BertLMHeadModel 사용
        
        # 3. Language Model
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        if config.use_decoder_only_language_model:
            self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            self.language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)

        # 4. Loss Modules
        self.vicreg_loss = VICRegLoss() # 1단계 Vision Pre-training용
        self.temp = nn.Parameter(torch.ones([]) * config.temperature) # 2단계 ITC Loss용
        self.itm_head = nn.Linear(config.qformer_config.hidden_size, 2) # 2단계 ITM Loss용

        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in self.language_model._tied_weights_keys]
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        self.language_model.set_input_embeddings(value)

    def _reshape_vision_outputs_to_spatial(self, vision_outputs: BaseModelOutput, B: int, P: int) -> Optional[Tuple[torch.Tensor, int, int]]:
        image_embeds = vision_outputs.last_hidden_state
        S, D = image_embeds.shape[1], image_embeds.shape[2]
        try:
            # CLS 토큰 제외 패치 수 계산
            num_patches = S - 1
            if num_patches <= 0: return None
            H_p = W_p = int(num_patches**0.5)
            patch_embeds = image_embeds[:, -num_patches:]
            spatial_embeds = patch_embeds.view(B, P, H_p, W_p, D)
            return spatial_embeds, H_p, W_p
        except (RuntimeError, ValueError):
            logger.warning("Could not reshape vision outputs to spatial representation.")
            return None

    def _compute_overlap_loss(self, vision_outputs: BaseModelOutput, B: int, P: int, overlap_consistency_weight: float) -> torch.Tensor:
        if P <= 1:
            return torch.tensor(0.0, device=vision_outputs.last_hidden_state.device)
            
        reshape_result = self._reshape_vision_outputs_to_spatial(vision_outputs, B, P)
        if reshape_result is None:
            return torch.tensor(0.0, device=vision_outputs.last_hidden_state.device)
            
        spatial_embeds, H, W = reshape_result
        # 겹치는 영역 추출
        left_patches_right_half = spatial_embeds[:, :-1, :, W//2:, :]
        right_patches_left_half = spatial_embeds[:, 1:, :, :W//2, :]
        
        loss, _ = self.vicreg_loss(left_patches_right_half, right_patches_left_half)
        return loss * overlap_consistency_weight

    def _compute_generative_loss(self, language_model_inputs, input_ids, attention_mask, labels, **kwargs) -> Dict[str, torch.Tensor]:
        text_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, text_embeds], dim=1)
        
        lang_model_attention_mask = torch.ones(language_model_inputs.size()[:-1], dtype=torch.long, device=inputs_embeds.device)
        extended_attention_mask = torch.cat([lang_model_attention_mask, attention_mask], dim=1)
        
        # Vision 토큰 부분은 loss 계산에서 제외
        target_labels = torch.full_like(extended_attention_mask, -100)
        num_vision_tokens = language_model_inputs.shape[1]
        target_labels[:, num_vision_tokens:] = labels
        
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=target_labels,
            return_dict=True,
            **kwargs
        )
        return outputs

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        stage: str = "finetune",
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        B, P, C, H, W = pixel_values.shape
        pixel_values_flat = pixel_values.view(B * P, C, H, W)

        # =======================================================
        # === 1단계: Vision Pre-training
        # =======================================================
        if stage == "vision_pretrain":
            vision_outputs = self.vision_model(pixel_values=pixel_values_flat, output_hidden_states=True, return_dict=True)
            loss = self._compute_overlap_loss(vision_outputs, B, P, kwargs.get("overlap_consistency_weight", 1.0))
            return {"loss": loss}

        # Vision 특징 추출 (2, 3단계 공통)
        image_embeds = self.vision_model(pixel_values=pixel_values_flat)[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # =======================================================
        # === 2단계: Q-Former Pre-training (ITC, ITM, LM)
        # =======================================================
        if stage == "qformer_pretrain":
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

            # 1. 이미지 특징 인코딩
            image_outputs = self.qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
            )
            image_features = image_outputs.last_hidden_state

            # 2. 텍스트 특징 인코딩
            text_outputs = self.qformer.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            text_features = text_outputs.last_hidden_state

            # 3. ITC Loss
            image_feat_itc = F.normalize(image_features[:, 0, :], dim=-1)
            text_feat_itc = F.normalize(text_features[:, 0, :], dim=-1)
            sim_i2t = torch.matmul(image_feat_itc, text_feat_itc.t()) * self.temp
            sim_t2i = torch.matmul(text_feat_itc, image_feat_itc.t()) * self.temp
            targets = torch.arange(B, device=pixel_values.device)
            loss_itc = (F.cross_entropy(sim_i2t, targets) + F.cross_entropy(sim_t2i, targets)) / 2

            # 4. ITM Loss
            with torch.no_grad():
                weights = sim_i2t.clone()
                weights.fill_diagonal_(-1e9)
                _, neg_indices = weights.min(dim=1)
            input_ids_neg = input_ids[neg_indices]
            attention_mask_neg = attention_mask[neg_indices]
            neg_text_outputs = self.qformer.bert(input_ids=input_ids_neg, attention_mask=attention_mask_neg, return_dict=True)
            
            text_features_all = torch.cat([text_features, neg_text_outputs.last_hidden_state], dim=0)
            text_attention_mask_all = torch.cat([attention_mask, attention_mask_neg], dim=0)
            image_features_repeated = image_features.repeat(2, 1, 1)
            
            itm_outputs = self.qformer.bert(
                query_embeds=image_features_repeated,
                encoder_hidden_states=text_features_all,
                encoder_attention_mask=text_attention_mask_all,
                return_dict=True,
            )
            itm_logits = self.itm_head(itm_outputs.last_hidden_state[:, 0, :])
            itm_labels = torch.cat([torch.ones(B, dtype=torch.long), torch.zeros(B, dtype=torch.long)], dim=0).to(itm_logits.device)
            loss_itm = F.cross_entropy(itm_logits, itm_labels)

            # 5. LM Loss
            lm_labels = input_ids.clone()
            lm_labels[lm_labels == self.qformer.config.pad_token_id] = -100
            lm_outputs = self.qformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=image_features,
                encoder_attention_mask=torch.ones(image_features.size()[:-1], dtype=torch.long, device=image_features.device),
                labels=lm_labels,
                return_dict=True,
            )
            loss_lm = lm_outputs.loss

            total_loss = loss_itc + loss_itm + loss_lm
            return {"loss": total_loss, "loss_itc": loss_itc, "loss_itm": loss_itm, "loss_lm": loss_lm}

        # =======================================================
        # === 3단계: Instruction Fine-tuning
        # =======================================================
        if stage == "finetune":
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = self.qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True
            )
            language_model_inputs = self.language_projection(query_outputs.last_hidden_state)
            
            lm_outputs = self._compute_generative_loss(language_model_inputs, input_ids, attention_mask, labels, **kwargs)

            if not return_dict:
                return (lm_outputs.loss, lm_outputs.logits) if lm_outputs.loss is not None else (lm_outputs.logits,)

            return {
                "loss": lm_outputs.loss,
                "logits": lm_outputs.logits,
            }

    @torch.no_grad()
    def generate(self, pixel_values: torch.FloatTensor, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.LongTensor] = None, **generate_kwargs) -> torch.LongTensor:
        B = pixel_values.shape[0]
        P = pixel_values.shape[1]
        pixel_values_flat = pixel_values.view(B * P, -1, *pixel_values.shape[3:])
        image_embeds = self.vision_model(pixel_values=pixel_values_flat)[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True
        )
        language_model_inputs = self.language_projection(query_outputs.last_hidden_state)

        if input_ids is None:
            input_ids = torch.tensor([[self.config.text_config.bos_token_id]], dtype=torch.long, device=pixel_values.device).repeat(B, 1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        text_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, text_embeds], dim=1)
        lang_model_attention_mask = torch.ones(language_model_inputs.size()[:-1], dtype=torch.long, device=inputs_embeds.device)
        attention_mask = torch.cat([lang_model_attention_mask, attention_mask], dim=1)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
        return outputs