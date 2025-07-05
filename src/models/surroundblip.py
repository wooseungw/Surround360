from typing import Any, Optional, Tuple, Dict

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel

from transformers.generation import GenerationMixin
from transformers.utils import (
    logging,
)
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoConfig
from transformers.models.blip_2.modeling_blip_2 import Blip2PreTrainedModel, Blip2VisionModel, Blip2QFormerModel
from transformers import Blip2Config
from ..loss.vicreg import VICRegLoss

logger = logging.get_logger(__name__)


class SurroundBlip(Blip2PreTrainedModel, GenerationMixin):
    config_class = Blip2Config
    main_input_name = "pixel_values"

    def __init__(self, config: Blip2Config):
        super().__init__(config)
        self.vision_model = Blip2VisionModel(config.vision_config)
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        
        if config.use_decoder_only_language_model:
            self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            self.language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
        
        self.vicreg_loss = VICRegLoss()
        
        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in self.language_model._tied_weights_keys]
        self.post_init()

    # --- [해결책 1] Gradient Checkpointing 지원 메서드 ---
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={}):
        self.vision_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            
    def gradient_checkpointing_disable(self):
        self.vision_model.gradient_checkpointing_disable()
        self.language_model.gradient_checkpointing_disable()

    # --- [해결책 2] _reshape_vision_outputs_to_spatial 헬퍼 메서드 ---
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

    def _compute_overlap_loss(self, vision_outputs: BaseModelOutput, B: int, P: int) -> torch.Tensor:
        if P <= 1:
            return torch.tensor(0.0, device=vision_outputs.last_hidden_state.device)
        reshape_result = self._reshape_vision_outputs_to_spatial(vision_outputs, B, P)
        if reshape_result is None:
            return torch.tensor(0.0, device=vision_outputs.last_hidden_state.device)
        spatial_embeds, H, W = reshape_result
        left_patches_right_half = spatial_embeds[:, :-1, :, W//2:, :]
        right_patches_left_half = spatial_embeds[:, 1:, :, :W//2, :]
        loss, _ = self.vicreg_loss(left_patches_right_half, right_patches_left_half)
        return loss
        
    # --- [해결책 3] get_input_embeddings 메서드 ---
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    # --- 2단계 학습 및 generate 호환성을 위한 forward 메서드 (수정 완료) ---
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        pretrain_vision_only: bool = False,
        overlap_consistency_weight: float = 1.0,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        
        # kwargs에서 'output_attentions', 'output_hidden_states' 등을 가져오도록 설정
        # generate 메소드와의 호환성을 위해 필요
        output_attentions = kwargs.get("output_attentions", False)
        output_hidden_states = kwargs.get("output_hidden_states", False)
        
        B, P, C, H, W = pixel_values.shape
        pixel_values_flat = pixel_values.view(B * P, C, H, W)
        # 비전 모델에 output_hidden_states 전달
        vision_outputs = self.vision_model(pixel_values=pixel_values_flat, output_hidden_states=True, return_dict=True)

        # === 1단계: Vision Pre-training 경로 ===
        if pretrain_vision_only:
            loss = self._compute_overlap_loss(vision_outputs, B, P) * overlap_consistency_weight
            return {"loss": loss}

        # === 2단계: Instruction Fine-tuning 경로 ===
        image_embeds = vision_outputs.last_hidden_state
        S, D = image_embeds.shape[1], image_embeds.shape[2]
        
        image_embeds_reshaped = image_embeds.view(B, P * S, D)
        image_attention_mask = torch.ones(image_embeds_reshaped.size()[:-1], dtype=torch.long, device=image_embeds_reshaped.device)
        
        query_tokens = self.query_tokens.expand(B, -1, -1)
        query_outputs = self.qformer(query_embeds=query_tokens, encoder_hidden_states=image_embeds_reshaped, encoder_attention_mask=image_attention_mask, return_dict=True)
        query_output = query_outputs.last_hidden_state
        
        language_model_inputs = self.language_projection(query_output)
        
        # generate 메소드에서는 input_ids 대신 inputs_embeds만 전달될 수 있음
        if input_ids is not None:
            text_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([language_model_inputs, text_embeds], dim=1)
        else:
            # generate의 두 번째 step부터는 이 경로를 사용
            inputs_embeds = language_model_inputs

        # attention_mask 확장
        lang_model_attention_mask = torch.ones(language_model_inputs.size()[:-1], dtype=torch.long, device=inputs_embeds.device)
        if attention_mask is None and input_ids is not None:
            attention_mask = torch.ones_like(input_ids)
        
        if attention_mask is not None:
            attention_mask = torch.cat([lang_model_attention_mask, attention_mask], dim=1)
        
        # --- [핵심 수정] labels 텐서를 inputs_embeds 길이에 맞게 동적으로 확장 ---
        # 학습 시에만 (labels가 존재할 때만) 이 로직이 실행됩니다.
        if labels is not None:
            # 1. 목표 시퀀스 길이는 확장된 inputs_embeds의 길이
            target_length = inputs_embeds.shape[1]
            
            # 2. IGNORE_INDEX(-100)로 채워진 새로운 레이블 텐서 생성
            #    shape: [batch_size, target_length]
            new_labels = torch.full(
                (B, target_length), 
                -100,  # IGNORE_INDEX
                dtype=torch.long, 
                device=inputs_embeds.device
            )
            
            # 3. 이미지 토큰(Q-Former 출력)의 길이를 계산
            num_vision_tokens = language_model_inputs.shape[1]
            
            # 4. 새로운 레이블 텐서의 뒷부분에 원래 텍스트 레이블을 복사
            #    이때, 원래 텍스트 레이블의 길이를 그대로 사용
            new_labels[:, num_vision_tokens:] = labels
            
            # 5. 확장된 new_labels를 실제 사용할 레이블로 지정
            labels_for_loss = new_labels
        else:
            # 추론/생성 시에는 labels가 필요 없음
            labels_for_loss = None

        # 확장된 `labels_for_loss`와 `attention_mask`를 사용하여 언어 모델 호출
        outputs = self.language_model(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask, 
            labels=labels_for_loss, 
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # generate 메소드 호환성을 위해 past_key_values 전달
            past_key_values=kwargs.get("past_key_values")
        )
        
        # `generate`는 loss를 반환하지 않으므로, loss가 없는 경우를 처리
        loss = outputs.loss if outputs.loss is not None else None
        
        # return_dict가 False일 수 있는 경우를 대비 (Hugging Face 표준)
        if not return_dict:
            # loss가 None일 때 outputs.logits만 반환하도록 처리
            return (outputs.logits,) if loss is None else (loss, outputs.logits)

        return {
            "loss": loss,
            "logits": outputs.logits,
            "past_key_values": outputs.past_key_values,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }

    # --- generate 호환성을 위한 '생성 위임' 방식의 generate 메서드 ---
    @torch.no_grad()
    def generate(self, pixel_values: torch.FloatTensor, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.LongTensor] = None, **generate_kwargs,) -> torch.LongTensor:
        B = pixel_values.shape[0]
        
        # 1. 시각 정보 처리
        B, P, C, H, W = pixel_values.shape
        pixel_values_flat = pixel_values.view(B * P, C, H, W)
        vision_outputs = self.vision_model(pixel_values=pixel_values_flat, return_dict=True)
        image_embeds = vision_outputs.last_hidden_state
        S, D = image_embeds.shape[1], image_embeds.shape[2]
        image_embeds_reshaped = image_embeds.view(B, P * S, D)
        image_attention_mask = torch.ones(image_embeds_reshaped.size()[:-1], dtype=torch.long, device=image_embeds_reshaped.device)
        query_tokens = self.query_tokens.expand(B, -1, -1)
        query_outputs = self.qformer(query_embeds=query_tokens, encoder_hidden_states=image_embeds_reshaped, encoder_attention_mask=image_attention_mask, return_dict=True)
        language_model_inputs = self.language_projection(query_outputs.last_hidden_state)
        
        # 2. 텍스트 프롬프트 처리
        if input_ids is None:
            input_ids = torch.tensor([[self.config.text_config.bos_token_id]], dtype=torch.long, device=pixel_values.device).repeat(B, 1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # 3. 임베딩 결합
        text_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, text_embeds], dim=1)
        lang_model_attention_mask = torch.ones(language_model_inputs.size()[:-1], dtype=torch.long, device=inputs_embeds.device)
        attention_mask = torch.cat([lang_model_attention_mask, attention_mask], dim=1)

        # 4. 생성 작업을 language_model에 위임
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )
        return outputs