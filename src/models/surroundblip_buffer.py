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

class SurroundBlip(Blip2PreTrainedModel):
    """
    1단계 표현 학습(ITM, VICReg)을 위한 모델. LLM이 없어 가볍습니다.
    """
    config_class = Blip2Config
    main_input_name = "pixel_values"
    _keep_in_fp32_modules = ["query_tokens", "qformer"]

    def __init__(self, config: Blip2Config):
        super().__init__(config)
        
        self.vision_model = Blip2VisionModel(config.vision_config)
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)
        
        # ITM을 위한 텍스트 인코딩은 Q-Former가 담당
        # Image-Text Matching (ITM)을 위한 분류 헤드
        self.itm_head = nn.Linear(config.qformer_config.hidden_size, 2)
        
        self.vicreg_loss = VICRegLoss(
            sim_coef=getattr(config, "vicreg_sim_coef", 25.0),
            var_coef=getattr(config, "vicreg_var_coef", 25.0),
            cov_coef=getattr(config, "vicreg_cov_coef", 1.0)
        )
        self.post_init()
        
    def _reshape_vision_outputs_to_spatial(self, vision_outputs: BaseModelOutput, B: int, P: int) -> Optional[Tuple[torch.Tensor, int, int]]:
        """Vision Encoder의 출력을 (B, P, H, W, D)의 5D 공간 텐서로 변환합니다."""
        image_embeds = vision_outputs.last_hidden_state
        S, D = image_embeds.shape[1], image_embeds.shape[2]
        try:
            if (S - 1) > 0 and math.isqrt(S - 1) ** 2 == S - 1:
                H_p = W_p = math.isqrt(S - 1)
                patch_embeds = image_embeds[:, 1:, :]
            elif math.isqrt(S) ** 2 == S:
                H_p = W_p = math.isqrt(S)
                patch_embeds = image_embeds
            else:
                return None
            spatial_embeds = patch_embeds.view(B, P, H_p, W_p, D)
            return spatial_embeds, H_p, W_p
        except RuntimeError:
            return None

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
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels_itm: torch.LongTensor,
        overlap_consistency_weight: float = 0.1,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PretrainingOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        B, P, C, H, W = pixel_values.shape
        pixel_values_flat = pixel_values.view(B * P, C, H, W)
        
        # 1. Vision Encoder를 통해 이미지 특징 추출
        vision_outputs = self.vision_model(
            pixel_values=pixel_values_flat,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        # 2. 오버랩 일관성 손실(VICReg) 계산
        overlap_loss = self._compute_overlap_loss(vision_outputs, B, P)
        
        # 3. ITM(Image-Text Matching) 손실 계산
        image_embeds = vision_outputs.last_hidden_state

        # 데이터로더에서 온 텍스트 배치는 (B*2, L), 이미지 배치는 (B,...) 이므로
        # 이미지 특징을 텍스트 배치 크기에 맞게 확장
        num_text_samples = input_ids.shape[0]
        image_embeds_expanded = image_embeds.repeat_interleave(num_text_samples // (B * P), dim=0)
        image_attention_mask_expanded = torch.ones(image_embeds_expanded.size()[:-1], dtype=torch.long, device=image_embeds.device)
        
        # Q-Former에 이미지와 텍스트를 함께 입력하여 관계성 분석
        qformer_outputs = self.qformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds_expanded,
            encoder_attention_mask=image_attention_mask_expanded,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        # 텍스트의 [CLS] 토큰 특징으로 ITM 로짓 계산
        itm_features = qformer_outputs.last_hidden_state[:, 0, :]
        itm_logits = self.itm_head(itm_features)
        
        itm_loss = F.cross_entropy(itm_logits.view(-1, 2), labels_itm.view(-1))
        
        # 4. 최종 손실 결합
        total_loss = itm_loss + overlap_consistency_weight * overlap_loss

        if not return_dict:
            return (total_loss, itm_logits)

        return PretrainingOutput(
            loss=total_loss,
            itm_loss=itm_loss,
            overlap_loss=overlap_loss,
            itm_logits=itm_logits,
            vision_outputs=vision_outputs,
            qformer_outputs=qformer_outputs,
        )
# ==============================================================================
# 섹션 2: 2단계 조건부 생성 모델 클래스
# ==============================================================================

class SurroundBlipForConditionalGeneration(Blip2PreTrainedModel, GenerationMixin):
    """
    2단계 조건부 생성(VQA, Captioning)을 위한 모델.
    1단계에서 사전 학습된 가중치를 불러와 사용합니다.
    """
    config_class = Blip2Config

    def __init__(self, config):
        super().__init__(config)
        
        self.vision_model = Blip2VisionModel(config.vision_config)
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        
        if config.use_decoder_only_language_model:
            self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            self.language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)

        self.post_init()
    
    def get_input_embeddings(self): return self.language_model.get_input_embeddings()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        B, P, C, H, W = pixel_values.shape
        pixel_values_flat = pixel_values.view(B * P, C, H, W)
        
        # 1. 이미지 인코딩
        vision_outputs = self.vision_model(pixel_values_flat, return_dict=True)
        image_embeds = vision_outputs.last_hidden_state
        
        # 2. Q-Former로 이미지 특징 요약
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        qformer_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device),
            return_dict=True,
        )
        query_output = qformer_outputs.last_hidden_state.view(B, P * self.config.num_query_tokens, -1)
        
        # 3. LLM 입력 준비
        language_model_inputs = self.language_projection(query_output)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        language_attention_mask = torch.ones(language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds], dim=1)
        attention_mask = torch.cat([language_attention_mask, attention_mask], dim=1)
        
        # 4. 생성 손실 계산
        if self.config.use_decoder_only_language_model:
            # Decoder-only 모델의 경우 label padding 처리 필요
            if labels is not None:
                # 생성될 텍스트 부분만 손실 계산
                # 이미지 임베딩 부분은 -100으로 마스킹
                image_tokens_len = language_model_inputs.shape[1]
                labels = labels.to(logits.device)
                empty_labels = torch.full((B, image_tokens_len), -100, dtype=torch.long).to(labels.device)
                labels = torch.cat([empty_labels, labels], dim=1)

            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
            )
        else: # Encoder-Decoder 모델
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=labels, # Teacher forcing
                return_dict=True,
            )
        
        loss = outputs.loss
        logits = outputs.logits

        if not return_dict:
            return (loss, logits)
            
        return Blip2ForConditionalGenerationModelOutput(
            loss=loss, logits=logits, vision_outputs=vision_outputs,
            qformer_outputs=qformer_outputs, language_model_outputs=outputs
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        # Flatten patch dimension like in forward
        B, P, C, H, W = pixel_values.shape
        pixel_values_flat = pixel_values.view(B * P, C, H, W)
        image_embeds = self.vision_model(pixel_values_flat, return_dict=True).last_hidden_state
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        qformer_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        )
        query_output = qformer_outputs[0].view(B, P * self.config.num_query_tokens, -1)
        language_model_inputs = self.language_projection(query_output)
        
        if input_ids is None:
            input_ids = torch.ones((B, 1), dtype=torch.long, device=pixel_values.device) * self.config.text_config.bos_token_id

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds], dim=1)
        
        return self.language_model.generate(inputs_embeds=inputs_embeds, **generate_kwargs)
