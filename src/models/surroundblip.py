import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
    torch_int,
)
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoConfig
from transformers.models.blip_2.modeling_blip_2 import Blip2ForConditionalGenerationModelOutput, Blip2PreTrainedModel, Blip2VisionModel, Blip2QFormerModel
from transformers import Blip2Config, Blip2QFormerConfig, Blip2VisionConfig
from ..loss.vicreg import VICRegLoss

logger = logging.get_logger(__name__)

@dataclass
class PretrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    itm_loss: Optional[torch.FloatTensor] = None
    overlap_loss: Optional[torch.FloatTensor] = None
    itm_logits: Optional[torch.FloatTensor] = None
    vision_outputs: Optional[BaseModelOutput] = None
    qformer_outputs: Optional[BaseModelOutput] = None

class SurroundBlip(Blip2PreTrainedModel, GenerationMixin):
    config_class = Blip2Config
    main_input_name = "pixel_values"

    def __init__(self, config: Blip2Config):
        # ... (이전과 동일한 __init__ 구현) ...
        super().__init__(config)
        self.vision_model = Blip2VisionModel(config.vision_config)
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        self.vicreg_loss = VICRegLoss()
        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in self.language_model._tied_weights_keys]
        self.post_init()
        
        # --- [해결책 2] 누락되었던 get_input_embeddings 메서드 추가 ---
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    # _compute_overlap_loss 메서드는 이전과 동일
    def _compute_overlap_loss(self, vision_outputs: BaseModelOutput, B: int, P: int) -> torch.Tensor:
        """슬라이딩 윈도우로 추출된 패치들의 겹치는 영역에 대한 VICReg 손실을 계산합니다."""
        if P <= 1:
            return torch.tensor(0.0, device=vision_outputs.last_hidden_state.device)

        reshape_result = self._reshape_vision_outputs_to_spatial(vision_outputs, B, P)
        if reshape_result is None:
            return torch.tensor(0.0, device=vision_outputs.last_hidden_state.device)
        
        spatial_embeds, H, W = reshape_result
        total_loss = 0.0
        num_pairs = P - 1
        
        if num_pairs == 0:
            return torch.tensor(0.0, device=vision_outputs.last_hidden_state.device)

        for i in range(num_pairs):
            current_patch_right_half = spatial_embeds[:, i, :, W//2:, :]
            next_patch_left_half = spatial_embeds[:, i + 1, :, :W//2, :]
            loss, _ = self.vicreg_loss(current_patch_right_half, next_patch_left_half)
            total_loss += loss
            
        return total_loss / num_pairs
    # --- [해결책 1] 누락되었던 _reshape_vision_outputs_to_spatial 메서드 추가 ---
    def _reshape_vision_outputs_to_spatial(self, vision_outputs: BaseModelOutput, B: int, P: int) -> Optional[Tuple[torch.Tensor, int, int]]:
        image_embeds = vision_outputs.last_hidden_state
        S, D = image_embeds.shape[1], image_embeds.shape[2]
        try:
            num_patches = S - 1 if (S > 1 and (S-1)**0.5 == int((S-1)**0.5)) else S
            if num_patches <= 0: return None
            H_p = W_p = int(num_patches**0.5)
            patch_embeds = image_embeds[:, -num_patches:]
            spatial_embeds = patch_embeds.view(B, P, H_p, W_p, D)
            return spatial_embeds, H_p, W_p
        except (RuntimeError, ValueError):
            return None
        
    # [핵심 2] `prepare_inputs_for_generation` 메서드는 generate의 헬퍼 역할
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # `past_key_values`가 있다면, 두 번째 스텝 이후이므로 비전 처리를 생략
        if past_key_values:
            return {
                "inputs_embeds": self.get_input_embeddings()(input_ids),
                "attention_mask": kwargs.get("attention_mask"),
                "past_key_values": past_key_values,
            }
        
        pixel_values = kwargs.pop("pixel_values")
        B, P, C, H, W = pixel_values.shape
        vision_outputs = self.vision_model(pixel_values=pixel_values.view(B*P, C, H, W))
        query_outputs = self.qformer(encoder_hidden_states=vision_outputs.last_hidden_state.view(B, P * vision_outputs.last_hidden_state.shape[1], -1))
        language_model_inputs = self.language_projection(query_outputs.last_hidden_state)
        text_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, text_embeds], dim=1)
        lang_attn_mask = torch.ones(language_model_inputs.size()[:-1], dtype=torch.long, device=inputs_embeds.device)
        attn_mask = kwargs.get("attention_mask", torch.ones_like(input_ids))
        attention_mask = torch.cat([lang_attn_mask, attn_mask], dim=1)
        return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}

    # [핵심 3] `forward`는 분기 로직을 가지면서, 2단계/추론 시 표준 출력을 반환
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        return_dict: Optional[bool] = None,
        pretrain_vision_only: bool = False,
        overlap_consistency_weight: float = 1.0,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # === 1단계: Vision Pre-training 경로 (다른 Loss 사용) ===
        if pretrain_vision_only:
            B, P, C, H, W = pixel_values.shape
            vision_outputs = self.vision_model(pixel_values.view(B*P,C,H,W), output_hidden_states=True, return_dict=True)
            loss = self._compute_overlap_loss(vision_outputs, B, P) * overlap_consistency_weight
            return CausalLMOutputWithPast(loss=loss) # 표준 형식으로 반환

        # === 2단계 학습 및 추론(generate) 경로 ===
        if inputs_embeds is None:
            # 학습 시 또는 generate 첫 스텝
            # prepare_inputs_for_generation과 유사한 로직으로 임베딩 생성
            B, P, C, H, W = pixel_values.shape
            pixel_values_flat = pixel_values.view(B*P, C, H, W)
            vision_outputs = self.vision_model(pixel_values=pixel_values_flat)
            image_embeds = vision_outputs.last_hidden_state
            S, D = image_embeds.shape[1], image_embeds.shape[2]
            image_embeds_reshaped = image_embeds.view(B, P*S, D)
            image_attention_mask = torch.ones(image_embeds_reshaped.size()[:-1], dtype=torch.long, device=image_embeds_reshaped.device)
            query_tokens = self.query_tokens.expand(B, -1, -1)
            query_outputs = self.qformer(query_embeds=query_tokens, encoder_hidden_states=image_embeds_reshaped, encoder_attention_mask=image_attention_mask)
            language_model_inputs = self.language_projection(query_outputs.last_hidden_state)
            text_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([language_model_inputs, text_embeds], dim=1)
            lang_attn_mask = torch.ones(language_model_inputs.size()[:-1], dtype=torch.long, device=inputs_embeds.device)
            if attention_mask is None: attention_mask = torch.ones_like(input_ids)
            attention_mask = torch.cat([lang_attn_mask, attention_mask], dim=1)
        
        # 언어 모델 통과
        outputs = self.language_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            past_key_values=past_key_values, use_cache=use_cache,
            return_dict=True, output_attentions=output_attentions, output_hidden_states=output_hidden_states,
        )
        
        loss = None
        if labels is not None:
            logits = outputs.logits
            num_image_tokens = inputs_embeds.shape[1] - input_ids.shape[1]
            full_labels = torch.full(logits.shape[:2], fill_value=-100, dtype=labels.dtype, device=logits.device)
            full_labels[:, num_image_tokens:] = labels
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = full_labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))

        # generate가 요구하는 '약속'을 지키기 위해 표준 CausalLMOutputWithPast 형식으로 반환
        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values, # <--- 가장 중요한 부분!
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
