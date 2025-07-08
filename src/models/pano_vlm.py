from typing import Optional, Dict, Any, Union, List
import torch
from torch import nn
import torch.nn.functional as F
from transformers import (
    PreTrainedModel, 
    PretrainedConfig,
    AutoModel, 
    AutoModelForCausalLM,
    AutoConfig
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from .pano_processor import PanoVLMProcessor


class PanoVLMConfig(PretrainedConfig):
    """
    PanoVLM 모델의 설정 클래스
    """
    model_type = "pano_vlm"

    def __init__(
        self,
        vision_tower_name: str = "openai/clip-vit-large-patch14-336",
        llm_name: str = "meta-llama/Llama-2-7b-chat-hf",
        projector_type: str = "mlp",
        projector_hidden_dim: int = 4096,
        mm_vision_select_layer: int = -1,
        mm_vision_select_feature: str = "patch",
        mm_patch_merge_type: str = "flat",
        image_token_len: int = 576,  # CLIP ViT-L의 경우 24*24=576
        # SurroundBlip 호환성을 위한 기본 설정들
        use_return_dict: bool = True,
        temperature: float = 0.07,
        **kwargs,
    ):
        self.vision_tower_name = vision_tower_name
        self.llm_name = llm_name
        self.projector_type = projector_type
        self.projector_hidden_dim = projector_hidden_dim
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_patch_merge_type = mm_patch_merge_type
        self.image_token_len = image_token_len
        self.use_return_dict = use_return_dict
        self.temperature = temperature
        
        # 자동으로 vision과 llm config 로드
        try:
            self.vision_config = AutoConfig.from_pretrained(vision_tower_name)
            self.llm_config = AutoConfig.from_pretrained(llm_name)
        except Exception as e:
            print(f"Config loading warning: {e}")
            self.vision_config = None
            self.llm_config = None
        
        super().__init__(**kwargs)


class PanoVLMMultiModalProjector(nn.Module):
    """
    비전 특징을 언어 모델 임베딩 공간으로 매핑하는 프로젝터
    """
    def __init__(self, config: PanoVLMConfig):
        super().__init__()
        self.config = config
        
        # 입력/출력 차원 설정
        if config.vision_config:
            vision_hidden_size = config.vision_config.hidden_size
        else:
            # CLIP ViT-L 기본값
            vision_hidden_size = 1024
            
        if config.llm_config:
            llm_hidden_size = config.llm_config.hidden_size
        else:
            # LLaMA-7B 기본값
            llm_hidden_size = 4096
        
        # 프로젝터 타입에 따른 구현
        if config.projector_type == "linear":
            self.projector = nn.Linear(vision_hidden_size, llm_hidden_size)
        elif config.projector_type == "mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_hidden_size, config.projector_hidden_dim),
                nn.GELU(),
                nn.Linear(config.projector_hidden_dim, llm_hidden_size)
            )
        elif config.projector_type == "mlp2x_gelu":
            # LLaVA-1.5 스타일 2-layer MLP
            self.projector = nn.Sequential(
                nn.Linear(vision_hidden_size, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )
        else:
            raise ValueError(f"Unknown projector type: {config.projector_type}")
    
    def forward(self, image_features):
        return self.projector(image_features)


class PanoVLM(PreTrainedModel):
    """
    확장 가능한 파노라마 비전-언어 모델
    임의의 비전 인코더와 LLM을 결합할 수 있습니다.
    """
    config_class = PanoVLMConfig
    
    def __init__(self, config: PanoVLMConfig):
        super().__init__(config)
        
        # 비전 타워 (비전 인코더)
        self.vision_tower = AutoModel.from_pretrained(
            config.vision_tower_name,
            torch_dtype=torch.float16
        )
        
        # 언어 모델
        self.language_model = AutoModelForCausalLM.from_pretrained(
            config.llm_name,
            torch_dtype=torch.float16
        )
        
        # 멀티모달 프로젝터
        self.mm_projector = PanoVLMMultiModalProjector(config)
        
        # 비전 타워는 기본적으로 고정
        self.vision_tower.requires_grad_(False)
        
        # 특별 토큰 관리
        self.image_token_id = None  # 프로세서에서 설정됨
        
    def get_vision_tower(self):
        """비전 타워 반환"""
        return self.vision_tower
    
    def get_mm_projector(self):
        """멀티모달 프로젝터 반환"""
        return self.mm_projector
    
    def encode_images(self, images):
        """
        이미지를 인코딩하여 특징 벡터 추출
        
        Args:
            images: [B, C, H, W] 또는 [B, P, C, H, W] (파노라마용)
            
        Returns:
            image_features: [B, seq_len, hidden_size]
        """
        # 파노라마 이미지 처리
        if images.dim() == 5:  # [B, P, C, H, W]
            B, P, C, H, W = images.shape
            images = images.view(B * P, C, H, W)
            
        # 비전 인코더 통과
        vision_outputs = self.vision_tower(
            images, 
            output_hidden_states=True
        )
        
        # 선택된 레이어에서 특징 추출
        if self.config.mm_vision_select_layer >= 0:
            image_features = vision_outputs.hidden_states[self.config.mm_vision_select_layer]
        else:
            image_features = vision_outputs.hidden_states[self.config.mm_vision_select_layer]
        
        # 특징 선택 (patch vs cls)
        if self.config.mm_vision_select_feature == "patch":
            # [CLS] 토큰 제거하고 패치 토큰만 사용
            image_features = image_features[:, 1:]
        elif self.config.mm_vision_select_feature == "cls_patch":
            # 모든 토큰 사용
            pass
        else:
            raise ValueError(f"Unexpected select feature: {self.config.mm_vision_select_feature}")
        
        # 파노라마 차원 복원
        if images.dim() == 5:
            seq_len = image_features.shape[1]
            hidden_size = image_features.shape[2]
            image_features = image_features.view(B, P * seq_len, hidden_size)
        
        return image_features
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images
    ):
        """
        멀티모달 입력을 위한 전처리
        LLaVA 스타일로 <image> 토큰을 실제 이미지 특징으로 교체
        """
        if images is None or input_ids.shape[1] == 1:
            # 이미지가 없거나 생성 단계
            return input_ids, attention_mask, past_key_values, None, labels
        
        # 이미지 인코딩
        image_features = self.encode_images(images)
        image_features = self.mm_projector(image_features)
        
        # 새로운 입력 준비
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # <image> 토큰 위치 찾기
            if self.image_token_id is None:
                # 임시로 찾기 (실제로는 프로세서에서 설정해야 함)
                image_token_positions = []
            else:
                image_token_positions = (cur_input_ids == self.image_token_id).nonzero(as_tuple=True)[0]
            
            if len(image_token_positions) == 0:
                # 이미지 토큰이 없으면 텍스트만 사용
                cur_input_embeds = self.get_input_embeddings()(cur_input_ids)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                continue
            
            # 임베딩 가져오기
            cur_input_embeds = self.get_input_embeddings()(cur_input_ids)
            cur_image_features = image_features[batch_idx]
            
            # <image> 토큰을 이미지 특징으로 교체
            for img_idx, img_pos in enumerate(image_token_positions):
                # 이미지 특징 길이만큼 확장
                num_patches = cur_image_features.shape[0]
                
                if img_pos == 0:
                    # 맨 앞에 이미지
                    cur_input_embeds = torch.cat([
                        cur_image_features,
                        cur_input_embeds[img_pos + 1:]
                    ], dim=0)
                elif img_pos == len(cur_input_ids) - 1:
                    # 맨 뒤에 이미지
                    cur_input_embeds = torch.cat([
                        cur_input_embeds[:img_pos],
                        cur_image_features
                    ], dim=0)
                else:
                    # 중간에 이미지
                    cur_input_embeds = torch.cat([
                        cur_input_embeds[:img_pos],
                        cur_image_features,
                        cur_input_embeds[img_pos + 1:]
                    ], dim=0)
                
                # 레이블도 같이 조정
                if labels is not None:
                    cur_labels = labels[batch_idx]
                    # 이미지 토큰 위치는 -100으로 마스킹
                    ignore_labels = torch.full((num_patches,), -100, 
                                             dtype=cur_labels.dtype, device=cur_labels.device)
                    
                    if img_pos == 0:
                        cur_labels = torch.cat([ignore_labels, cur_labels[img_pos + 1:]], dim=0)
                    elif img_pos == len(cur_input_ids) - 1:
                        cur_labels = torch.cat([cur_labels[:img_pos], ignore_labels], dim=0)
                    else:
                        cur_labels = torch.cat([
                            cur_labels[:img_pos], 
                            ignore_labels, 
                            cur_labels[img_pos + 1:]
                        ], dim=0)
            
            new_input_embeds.append(cur_input_embeds)
            if labels is not None:
                new_labels.append(cur_labels)
        
        # 패딩 처리
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        
        new_input_embeds_padded = torch.zeros(
            batch_size, max_len, new_input_embeds[0].shape[-1],
            dtype=new_input_embeds[0].dtype, device=new_input_embeds[0].device
        )
        new_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=input_ids.device)
        
        if labels is not None:
            new_labels_padded = torch.full(
                (batch_size, max_len), -100, dtype=labels.dtype, device=labels.device
            )
        
        for i, (cur_new_embed, cur_len) in enumerate(zip(new_input_embeds, [x.shape[0] for x in new_input_embeds])):
            new_input_embeds_padded[i, :cur_len] = cur_new_embed
            new_attention_mask[i, :cur_len] = 1
            if labels is not None:
                new_labels_padded[i, :cur_len] = new_labels[i]
        
        return None, new_attention_mask, past_key_values, new_input_embeds_padded, new_labels_padded if labels is not None else None
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,  # SurroundBlip 호환성
        return_dict: Optional[bool] = None,
        # 3단계 학습을 위한 stage 인자들
        stage: str = "finetune",  # "vision_pretrain", "qformer_pretrain", "finetune"
        overlap_consistency_weight: float = 1.0,
        **kwargs
    ) -> Union[tuple, CausalLMOutputWithPast, Dict[str, torch.Tensor]]:
        
        # SurroundBlip 호환성: pixel_values를 images로 변환
        if pixel_values is not None and images is None:
            images = pixel_values
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # =======================================================
        # === 1단계: Vision Pre-training (VICReg Loss)
        # =======================================================
        if stage == "vision_pretrain":
            if images is None:
                raise ValueError("Vision pretraining requires images")
            
            # 파노라마 이미지 처리
            if images.dim() == 5:  # [B, P, C, H, W]
                B, P = images.shape[:2]
                if P <= 1:
                    loss = torch.tensor(0.0, device=images.device, requires_grad=True)
                else:
                    # 간단한 VICReg 손실 (인접 패치 간 일관성)
                    image_features = self.encode_images(images)  # [B, P*seq_len, hidden_size]
                    
                    # 패치별로 분리
                    seq_len_per_patch = image_features.shape[1] // P
                    patch_features = image_features.view(B, P, seq_len_per_patch, -1)
                    
                    # 인접 패치 간 평균 특징 비교
                    left_patches = patch_features[:, :-1].mean(dim=2)  # [B, P-1, hidden_size]
                    right_patches = patch_features[:, 1:].mean(dim=2)  # [B, P-1, hidden_size]
                    
                    # MSE Loss로 간단한 일관성 손실
                    loss = F.mse_loss(left_patches, right_patches) * overlap_consistency_weight
            else:
                # 단일 이미지의 경우 손실 없음
                loss = torch.tensor(0.0, device=images.device, requires_grad=True)
            
            return {"loss": loss}
        
        # =======================================================
        # === 2단계: Q-Former Pre-training (Contrastive + Generation)
        # =======================================================
        elif stage == "qformer_pretrain":
            if images is None or input_ids is None:
                raise ValueError("Q-Former pretraining requires both images and text")
            
            # 이미지 특징 추출
            image_features = self.encode_images(images)
            image_features = self.mm_projector(image_features)
            
            # 간단한 contrastive loss (이미지-텍스트 유사도)
            text_embeds = self.get_input_embeddings()(input_ids).mean(dim=1)  # [B, hidden_size]
            image_embeds = image_features.mean(dim=1)  # [B, hidden_size]
            
            # L2 정규화
            text_embeds = F.normalize(text_embeds, dim=-1)
            image_embeds = F.normalize(image_embeds, dim=-1)
            
            # 유사도 행렬
            sim_matrix = torch.matmul(image_embeds, text_embeds.t()) / 0.07  # temperature=0.07
            targets = torch.arange(len(sim_matrix), device=sim_matrix.device)
            
            # Contrastive loss
            loss_i2t = F.cross_entropy(sim_matrix, targets)
            loss_t2i = F.cross_entropy(sim_matrix.t(), targets)
            contrastive_loss = (loss_i2t + loss_t2i) / 2
            
            # Generation loss
            if labels is not None:
                # 멀티모달 입력 준비
                (
                    _input_ids,
                    _attention_mask,
                    _past_key_values,
                    _inputs_embeds,
                    _labels
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids, attention_mask, past_key_values, labels, images
                )
                
                # 언어 모델 포워드
                lm_outputs = self.language_model(
                    input_ids=_input_ids,
                    attention_mask=_attention_mask,
                    past_key_values=_past_key_values,
                    inputs_embeds=_inputs_embeds,
                    labels=_labels,
                    return_dict=True
                )
                generation_loss = lm_outputs.loss if lm_outputs.loss is not None else torch.tensor(0.0, device=images.device)
            else:
                generation_loss = torch.tensor(0.0, device=images.device)
            
            total_loss = contrastive_loss + generation_loss
            return {
                "loss": total_loss,
                "contrastive_loss": contrastive_loss,
                "generation_loss": generation_loss
            }
        
        # =======================================================
        # === 3단계: Instruction Fine-tuning
        # =======================================================
        else:  # stage == "finetune" or default
            # 멀티모달 입력 준비
            if inputs_embeds is None and images is not None:
                (
                    input_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids, attention_mask, past_key_values, labels, images
                )
            
            # 언어 모델 포워드
            outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            
            if not return_dict:
                return outputs
            
            return {
                "loss": outputs.loss,
                "logits": outputs.logits,
                "past_key_values": outputs.past_key_values,
                "hidden_states": outputs.hidden_states,
                "attentions": outputs.attentions,
            }
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,  # SurroundBlip 호환성
        pixel_values: Optional[torch.Tensor] = None,  # SurroundBlip 호환성
        images: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """생성 메서드 - SurroundBlip과 호환"""
        
        # SurroundBlip 호환성
        if input_ids is not None and inputs is None:
            inputs = input_ids
        if pixel_values is not None and images is None:
            images = pixel_values
            
        if inputs is None:
            B = images.shape[0] if images is not None else 1
            # 기본 BOS 토큰으로 시작
            if hasattr(self.language_model.config, 'bos_token_id') and self.language_model.config.bos_token_id is not None:
                inputs = torch.tensor([[self.language_model.config.bos_token_id]], 
                                    dtype=torch.long, device=images.device if images is not None else 'cpu').repeat(B, 1)
            else:
                inputs = torch.tensor([[1]], dtype=torch.long, device=images.device if images is not None else 'cpu').repeat(B, 1)
                
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs)
        
        position_ids = kwargs.pop("position_ids", None)
        
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("inputs_embeds는 generate에서 지원되지 않습니다")
        
        if images is not None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                attention_mask,
                None,
                None,
                images
            )
        else:
            inputs_embeds = self.get_input_embeddings()(inputs)
        
        return self.language_model.generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    
    def get_input_embeddings(self):
        """언어 모델의 입력 임베딩 레이어 반환"""
        return self.language_model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        """언어 모델의 입력 임베딩 레이어 설정"""
        self.language_model.set_input_embeddings(value)
    
    def get_output_embeddings(self):
        """언어 모델의 출력 임베딩 레이어 반환"""
        return self.language_model.get_output_embeddings()
    
    def set_output_embeddings(self, new_embeddings):
        """언어 모델의 출력 임베딩 레이어 설정"""
        self.language_model.set_output_embeddings(new_embeddings)
    
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None):
        """토큰 임베딩 크기 조정 (새로운 특별 토큰 추가 시)"""
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens)
        return model_embeds
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """사전 훈련된 모델 로드 - 기존 체크포인트 호환성"""
        try:
            # PanoVLM 체크포인트 로드 시도
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        except:
            # 기본 설정으로 새 모델 생성
            config = PanoVLMConfig(**kwargs)
            return cls(config)
    
    def save_pretrained(self, save_directory, **kwargs):
        """모델 저장"""
        super().save_pretrained(save_directory, **kwargs)
        
        # 프로세서 설정도 함께 저장
        processor_config = {
            "vision_processor_name": self.config.vision_tower_name,
            "tokenizer_name": self.config.llm_name,
            "image_token": "<image>",
        }
        
        import json
        import os
        processor_config_path = os.path.join(save_directory, "processor_config.json")
        with open(processor_config_path, "w") as f:
            json.dump(processor_config, f, indent=2)
