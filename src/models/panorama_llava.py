# 파일 경로: src/models/panorama_llava.py

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Optional, List, Union, Dict

from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel, PretrainedConfig

class PanoramaLLaVAConfig(PretrainedConfig):
    """
    PanoramaLLaVA 모델을 위한 커스텀 설정 클래스.
    Vision Encoder와 Language Model의 설정을 포함합니다.
    """
    model_type = "panorama_llava"
    main_input_name = "pixel_values"

    def __init__(
        self,
        vision_config=None,
        language_config=None,
        mm_hidden_size=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_config = vision_config
        self.language_config = language_config
        self.mm_hidden_size = mm_hidden_size

class PanoramaLLaVA(PreTrainedModel):
    config_class = PanoramaLLaVAConfig
    
    def __init__(self, config: PanoramaLLaVAConfig):
        super().__init__(config)
        
        # 1. Vision Encoder (Vision Tower) 로드
        self.vision_tower = AutoModel.from_config(config.vision_config)

        # 2. Language Model 로드
        self.language_model = AutoModelForCausalLM.from_config(config.language_config)

        # 3. Vision -> Language 연결을 위한 MLP Projector 정의
        self.mm_projector = nn.Sequential(
            nn.Linear(config.vision_config.hidden_size, config.mm_hidden_size),
            nn.GELU(),
            nn.Linear(config.mm_hidden_size, config.language_config.hidden_size)
        )

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:

        B, P, C, H, W = pixel_values.shape
        
        # 1. 모든 패치를 Vision Tower에 통과
        pixel_values_flat = pixel_values.view(B * P, C, H, W)
        vision_outputs = self.vision_tower(pixel_values=pixel_values_flat, output_hidden_states=True)
        
        # [CLS] 토큰에 해당하는 마지막 히든 스테이트를 사용 (모델에 따라 인덱스 조정 가능)
        image_features = vision_outputs.last_hidden_state[:, 0, :] # (B*P, vision_hidden_size)
        
        # 2. 파노라마를 대표하는 단일 피처 생성 (평균)
        # (B*P, D_vision) -> (B, P, D_vision) -> (B, D_vision)
        image_features_aggregated = image_features.view(B, P, -1).mean(dim=1)
        
        # 3. MLP Projector를 통과시켜 LLM의 임베딩 공간으로 변환
        image_features_projected = self.mm_projector(image_features_aggregated) # (B, llm_hidden_size)

        # 4. 텍스트 임베딩과 결합
        text_embeds = self.get_input_embeddings()(input_ids)
        
        # 이미지 피처를 시퀀스의 첫 부분에 추가하기 위해 차원 확장
        # (B, D_llm) -> (B, 1, D_llm)
        image_features_expanded = image_features_projected.unsqueeze(1)
        
        inputs_embeds = torch.cat([image_features_expanded, text_embeds], dim=1)
        
        # 새로운 Attention Mask 생성
        image_attention_mask = torch.ones(image_features_expanded.size()[:-1], dtype=torch.long, device=inputs_embeds.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
        
        # 5. Language Model 통과
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels, # LM이 내부적으로 shift하여 loss 계산
        )
        
        return {"loss": outputs.loss, "logits": outputs.logits}

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ):
        B = pixel_values.shape[0]
        
        # 1. 시각 정보 처리 및 대표 피처 생성
        pixel_values_flat = pixel_values.view(B * pixel_values.shape[1], *pixel_values.shape[2:])
        vision_outputs = self.vision_tower(pixel_values=pixel_values_flat, output_hidden_states=True)
        image_features = vision_outputs.last_hidden_state[:, 0, :]
        image_features_aggregated = image_features.view(B, pixel_values.shape[1], -1).mean(dim=1)
        image_features_projected = self.mm_projector(image_features_aggregated).unsqueeze(1)

        # 2. 텍스트 프롬프트 처리
        if input_ids is None:
            input_ids = torch.tensor([[self.language_model.config.bos_token_id]], dtype=torch.long, device=pixel_values.device).repeat(B, 1)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        # 3. 시각/텍스트 임베딩 결합
        text_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([image_features_projected, text_embeds], dim=1)
        image_attention_mask = torch.ones(image_features_projected.size()[:-1], dtype=torch.long, device=inputs_embeds.device)
        attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)

        # 4. 언어 모델의 generate 함수에 직접 위임
        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs
        )
        
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={}):
        # 부모의 일반적인 방법 대신, 구체적인 지시를 내림
        self.vision_tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        self.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
