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
        # --- 기존 모듈 정의 (변경 없음) ---
        self.vision_model = Blip2VisionModel(config.vision_config)
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = Blip2QFormerModel(config.qformer_config)
        self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
        
        if config.use_decoder_only_language_model:
            self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            self.language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
        
        self.vicreg_loss = VICRegLoss()
        
        # --- [신규] 2단계 Q-Former 학습을 위한 모듈 추가 ---
        # Image-Text Contrastive (ITC) Loss를 위한 Temperature 파라미터
        self.temp = nn.Parameter(torch.ones([]) * config.temperature)
        
        # Image-Text Matching (ITM) Loss를 위한 분류기 헤드
        self.itm_head = nn.Linear(config.qformer_config.hidden_size, 2)
        
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
    
    def _compute_generative_loss(self, language_model_inputs, input_ids, attention_mask, labels, **kwargs) -> Dict[str, torch.Tensor]:
        """2단계와 3단계에서 공통으로 사용될 생성 손실 계산 로직"""
        # 임베딩 결합
        text_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, text_embeds], dim=1)
        
        # Attention Mask 확장
        lang_model_attention_mask = torch.ones(language_model_inputs.size()[:-1], dtype=torch.long, device=inputs_embeds.device)
        attention_mask = torch.cat([lang_model_attention_mask, attention_mask], dim=1)
        
        # Labels 확장
        target_length = inputs_embeds.shape[1]
        new_labels = torch.full(
            (inputs_embeds.shape[0], target_length), 
            -100, dtype=torch.long, device=inputs_embeds.device
        )
        num_vision_tokens = language_model_inputs.shape[1]
        new_labels[:, num_vision_tokens:] = labels
        
        # 언어 모델 호출
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=new_labels,
            return_dict=True,
            **kwargs
        )
        return outputs

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
        # --- 스테이지 제어 인자 ---
        stage: str = "finetune", # "vision_pretrain", "qformer_pretrain", "finetune"
        # --- 1단계 인자 ---
        pretrain_vision_only: bool = False, # 하위 호환성을 위해 유지
        overlap_consistency_weight: float = 1.0,
        # --- 2단계 인자 ---
        itm_head: bool = True,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 하위 호환성: pretrain_vision_only=True 이면 stage를 vision_pretrain으로 간주
        if pretrain_vision_only:
            stage = "vision_pretrain"
            
        # === 공통 비전 피처 추출 ===
        B, P, C, H, W = pixel_values.shape
        pixel_values_flat = pixel_values.view(B * P, C, H, W)
        vision_outputs = self.vision_model(pixel_values=pixel_values_flat, output_hidden_states=True, return_dict=True)

        # =======================================================
        # === 1단계: Vision Pre-training
        # =======================================================
        if stage == "vision_pretrain":
            loss = self._compute_overlap_loss(vision_outputs, B, P) * overlap_consistency_weight
            return {"loss": loss}
        
        # === Q-Former 입력 준비 (2, 3단계 공통) ===
        image_embeds = vision_outputs.last_hidden_state
        S, D = image_embeds.shape[1], image_embeds.shape[2]
        image_embeds_reshaped = image_embeds.view(B, P * S, D)
        image_attention_mask = torch.ones(image_embeds_reshaped.size()[:-1], dtype=torch.long, device=image_embeds_reshaped.device)
        query_tokens = self.query_tokens.expand(B, -1, -1)
        
        # =======================================================
        # === [수정] 2단계: Q-Former Pre-training (3가지 Loss)
        # =======================================================
        if stage == "qformer_pretrain":
            # --- 2.1: Image-Text Contrastive (ITC) Loss ---
            
            # [수정] LLM의 임베딩 대신, Q-Former의 임베딩 레이어를 사용합니다.
            # self.qformer.embeddings는 768 차원의 텐서를 생성합니다.
            text_embeds = self.qformer.embeddings(input_ids=input_ids)
            
            # 이제 text_embeds를 query_embeds로 전달하여 Q-Former를 통과시킵니다.
            text_qformer_outputs = self.qformer(
                query_embeds=text_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
            text_feat = F.normalize(text_qformer_outputs.last_hidden_state[:, 0, :], dim=-1)

            # 이미지 피처 추출 (기존과 동일)
            image_qformer_outputs = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_reshaped,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
            )
            image_feat = F.normalize(image_qformer_outputs.last_hidden_state[:, 0, :], dim=-1)
            
            sim_i2t = torch.matmul(image_feat, text_feat.t()) * self.temp
            sim_t2i = torch.matmul(text_feat, image_feat.t()) * self.temp

            # [유지] 개선된 ITC Loss 로직 (매우 좋은 아이디어입니다)
            image_ids = kwargs.get("image_id") # 데이터셋에서 'image_id'를 전달받아야 함
            if image_ids is not None:
                image_ids_tensor = torch.tensor([hash(img_id) for img_id in image_ids], device=pixel_values.device)
                same_image_mask = (image_ids_tensor.unsqueeze(1) == image_ids_tensor.unsqueeze(0))
                sim_i2t = sim_i2t.masked_fill(~same_image_mask, -1e9)
                sim_t2i = sim_t2i.masked_fill(~same_image_mask, -1e9)
            
            targets = torch.arange(B, device=pixel_values.device)
            loss_itc = (F.cross_entropy(sim_i2t, targets) + F.cross_entropy(sim_t2i, targets)) / 2
            
            # --- 2.2: Image-Text Matching (ITM) Loss ---
            # [유지] Hard Negative 샘플링 로직 (매우 좋은 아이디어입니다)
            with torch.no_grad():
                weights = sim_i2t.clone()
                weights.fill_diagonal_(-1e9) # 자기 자신은 제외
                if image_ids is not None: # 같은 이미지도 제외
                    weights.masked_fill_(same_image_mask, -1e9)
                
                # 각 이미지에 대해 가장 유사도가 낮은(어려운) 텍스트를 negative로 선택
                _, neg_indices = weights.min(dim=1)

            input_ids_neg = input_ids[neg_indices]
            attention_mask_neg = attention_mask[neg_indices]
            
            # [수정] Negative 샘플의 input_ids도 임베딩으로 변환
            text_embeds_neg = self.get_input_embeddings()(input_ids_neg)
            
            # [수정] ITM을 위한 Q-Former 호출 시, 텍스트 부분을 encoder_hidden_states로 전달
            # Positive 텍스트
            itm_pos_outputs = self.qformer(
                query_embeds=image_qformer_outputs.last_hidden_state,
                encoder_hidden_states=text_embeds,
                encoder_attention_mask=attention_mask,
                return_dict=True
            ).last_hidden_state[:,0,:]
            # Negative 텍스트
            itm_neg_outputs = self.qformer(
                query_embeds=image_qformer_outputs.last_hidden_state,
                encoder_hidden_states=text_embeds_neg,
                encoder_attention_mask=attention_mask_neg,
                return_dict=True
            ).last_hidden_state[:,0,:]
            
            itm_logits = self.itm_head(torch.cat([itm_pos_outputs, itm_neg_outputs], dim=0))
            itm_labels = torch.cat([torch.ones(B, dtype=torch.long), torch.zeros(B, dtype=torch.long)], dim=0).to(pixel_values.device)
            loss_itm = F.cross_entropy(itm_logits, itm_labels)

            # --- 2.3: Image-Grounded Text Generation (LM) Loss ---
            language_model_inputs = self.language_projection(image_qformer_outputs.last_hidden_state)
            lm_outputs = self._compute_generative_loss(language_model_inputs, input_ids, attention_mask, labels)
            loss_lm = lm_outputs.loss

            # --- 최종 손실 결합 ---
            total_loss = loss_itc + loss_itm + loss_lm
            return {"loss": total_loss, "loss_itc": loss_itc, "loss_itm": loss_itm, "loss_lm": loss_lm}

        # =======================================================
        # === 3단계: Instruction Fine-tuning
        # =======================================================
        if stage == "finetune":
            query_outputs = self.qformer(query_embeds=query_tokens, encoder_hidden_states=image_embeds_reshaped, encoder_attention_mask=image_attention_mask, return_dict=True)
            language_model_inputs = self.language_projection(query_outputs.last_hidden_state)
            
            lm_outputs = self._compute_generative_loss(language_model_inputs, input_ids, attention_mask, labels, **kwargs)

            if not return_dict:
                return (lm_outputs.loss, lm_outputs.logits) if lm_outputs.loss is not None else (lm_outputs.logits,)

            return {
                "loss": lm_outputs.loss,
                "logits": lm_outputs.logits,
                "past_key_values": lm_outputs.past_key_values,
                "hidden_states": lm_outputs.hidden_states,
                "attentions": lm_outputs.attentions,
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