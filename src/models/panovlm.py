from typing import Optional, Union, Tuple, Any, Dict, List
from transformers import PretrainedConfig, AutoConfig, AutoModel, PreTrainedModel, AutoModelForCausalLM
import transformers
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from contextlib import nullcontext
import math
import logging

# 특수 토큰 인덱스 설정
IMAGE_TOKEN_INDEX = -200  # 이미지 토큰 전용 인덱스
IGNORE_INDEX = -100  # 손실 계산 무시 인덱스

# 특수 토큰 문자열 설정
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# VLM 모델 매핑 - 언어 모델 및 시각 모델 분리를 위한 구성
VLM_MAPPING = {
    # Gemma 3 모델
    "google/gemma-3-4b-it": {
        "model_class": transformers.Gemma3ForConditionalGeneration,
        "vision_attr": None,  # 독립적인 vision encoder 사용
        "language_attr": "language_model"  # language_model 속성을 참조
    },
    "google/gemma-3-7b-it": {
        "model_class": transformers.Gemma3ForConditionalGeneration,
        "vision_attr": None,
        "language_attr": "language_model"
    },
    
    # Qwen 모델
    "Qwen/Qwen2.5-VL-3B-Instruct": {
        "model_class": transformers.Qwen2VLForConditionalGeneration,
        "vision_attr": None,  # 독립적인 vision encoder 사용
        "language_attr": "model"  # model 속성을 참조
    },
    "Qwen/Qwen2.5-VL-7B-Instruct": {
        "model_class": transformers.Qwen2VLForConditionalGeneration,
        "vision_attr": None,
        "language_attr": "model"
    },
    
    # Llama 3.1 모델 계열
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "model_class": transformers.LlamaForCausalLM,
        "vision_attr": None,
        "language_attr": None  # 전체 모델이 언어 모델
    },
    "meta-llama/Meta-Llama-3.1-70B-Instruct": {
        "model_class": transformers.LlamaForCausalLM,
        "vision_attr": None,
        "language_attr": None
    },
    
    # Mistral AI 최신 모델
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "model_class": transformers.MistralForCausalLM,
        "vision_attr": None,
        "language_attr": None
    },
}

# 비전 모델 매핑 (다양한 비전 인코더 지원)
VISION_MAPPING = {
    # CLIP 계열 모델들
    "openai/clip-vit-large-patch14": AutoModel,  # CLIP 비전 인코더
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": AutoModel,  # LAION CLIP 비전 인코더
    
    # DINO v2 계열 모델들
    "facebook/dinov2-large": AutoModel,  # DINO v2 비전 인코더
    "facebook/dinov2-giant": AutoModel,  # DINO v2 Giant 비전 인코더
    
    # SigLIP 모델들 (Google Research의 확장된 CLIP)
    "google/siglip-base-patch16": AutoModel,  # SigLIP base 모델
    "google/siglip-large-patch16": AutoModel,  # SigLIP large 모델
    "google/siglip-so400m-patch14-384": AutoModel,  # SigLIP SO400M 모델
    
    # EVA 모델 (MIM + CLIP)
    "BAAI/EVA-CLIP": AutoModel,  # EVA-CLIP
    
    # DINOv2+
    "facebook/dinov2-xlarge": AutoModel,  # DINO v2 XLarge
    
    # SAM (Segment Anything Model) 관련 인코더
    "facebook/sam-vit-huge": AutoModel,  # SAM ViT-H 모델의 이미지 인코더
    
    # 나머지는 기본 AutoModel로 로드
}

def get_model_class(model_name: str):
    """허깅페이스 모델 이름에 따라 적절한 모델 클래스 반환"""
    if model_name in VLM_MAPPING:
        return VLM_MAPPING[model_name]["model_class"]
    return transformers.AutoModelForCausalLM

def get_vision_model_class(model_name: str):
    """비전 모델 이름에 따라 적절한 모델 클래스 반환"""
    if model_name in VISION_MAPPING:
        return VISION_MAPPING[model_name]
    return AutoModel

def extract_text_model(full_vlm: Any, model_name: str = None) -> Any:
    """
    전체 VLM 인스턴스(full_vlm)로부터 순수 텍스트 생성 LM 부분만 추출하여 반환한다.
    
    Args:
        full_vlm: 전체 VLM 모델 인스턴스
        model_name: 모델 이름 (선택적, 제공되면 매핑을 사용해 직접 속성에 접근)
        
    Returns:
        순수 텍스트 생성 LM 부분
    
    새 VLM을 추가하려면 VLM_MAPPING에 매핑 정보를 추가하세요.
    """
    # 모델 이름이 제공되고 매핑에 있으면 바로 적절한 속성 반환
    if model_name and model_name in VLM_MAPPING:
        attr_name = VLM_MAPPING[model_name]["language_attr"]
        if attr_name:
            return getattr(full_vlm, attr_name)
        return full_vlm
    
    # 기존 인스턴스 기반 로직 유지
    if isinstance(full_vlm, transformers.Gemma3ForConditionalGeneration):
        return full_vlm.language_model
    elif isinstance(full_vlm, transformers.Qwen2VLForConditionalGeneration):
        return full_vlm.model
    else:
        return full_vlm

class PanoVLM(PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        cache_dir = getattr(config, "cache_dir", "./.cache")
        
        # 1) Vision encoder 로드 - 다양한 비전 모델 지원
        vision_cls = get_vision_model_class(config.vision_model_name_or_path)
        self.vision_model = vision_cls.from_pretrained(
            config.vision_model_name_or_path,
            config=config.vision_config,
            cache_dir=cache_dir
        )
        
        # 2) Language model 로드 - 다양한 언어 모델 지원
        lang_cls = get_model_class(config.language_model_name_or_path)
        
        # Gemma3 모델을 위한 eager 어텐션 구현 적용
        load_kwargs = {
            "config": config.language_config,
            "cache_dir": cache_dir
        }
        
        # Gemma3 모델인 경우 eager 어텐션 구현으로 설정
        if "gemma-3" in config.language_model_name_or_path.lower():
            print("Gemma3 모델에 eager 어텐션 구현 적용")
            load_kwargs["attn_implementation"] = "eager"
        
        full_vlm = lang_cls.from_pretrained(
            config.language_model_name_or_path,
            **load_kwargs
        )
        self.language_model = extract_text_model(full_vlm, config.language_model_name_or_path)

        # 3) Projector 설정 - 비전 임베딩을 언어 모델에 맞게 투영
        if hasattr(config, "projector_config") and config.projector_config:
            from src.models.projector import build_vision_projector
            
            # 프로젝터 구성 파라미터 가져오기
            # 비전 모델의 차원 가져오기
            if hasattr(self.vision_model.config, 'hidden_size'):
                vision_dim = self.vision_model.config.hidden_size
            elif hasattr(self.vision_model.config, 'embed_dim'):
                vision_dim = self.vision_model.config.embed_dim  # CLIP 모델
            else:
                vision_dim = 384  # DinoV2-small 기본값
                
            # 언어 모델의 차원 가져오기
            if hasattr(self.language_model.config, 'hidden_size'):
                lang_dim = self.language_model.config.hidden_size
            elif hasattr(self.language_model.config, 'model_dim'):
                lang_dim = self.language_model.config.model_dim  # Gemma-3 모델
            elif hasattr(self.language_model.config, 'hidden_dim'):
                lang_dim = self.language_model.config.hidden_dim
            elif hasattr(self.language_model.config, 'd_model'):
                lang_dim = self.language_model.config.d_model  # 일부 transformer 모델
            else:
                lang_dim = 2560  # 기본값 (Gemma-3-4B)
                
            in_dim = getattr(config.projector_config, "in_features", vision_dim)
            out_dim = getattr(config.projector_config, "out_features", lang_dim)
            projector_type = getattr(config.projector_config, "type", "mlp2x_gelu")
            
            # 프로젝터 생성 (mlp2x_gelu는 추가 설정이 필요 없음)
            self.projector = build_vision_projector(
                d_v=in_dim, 
                d_l=out_dim, 
                projector_type=projector_type,
                vision_cfg=self.vision_model.config if projector_type == "pooler" else None
            )
        else:
            self.projector = None

        # 4) 메모리 효율성 설정
        self.use_fp16 = getattr(config, "use_fp16", True)
        self.gradient_checkpointing = getattr(config, "gradient_checkpointing", False)
        
        if self.gradient_checkpointing:
            self.vision_model.gradient_checkpointing_enable()
            if hasattr(self.language_model, "gradient_checkpointing_enable"):
                self.language_model.gradient_checkpointing_enable()
                
        # 5) 이미지 토크나이저 설정
        self.image_token_index = getattr(config, "image_token_index", IMAGE_TOKEN_INDEX)
        self.tokenizer = getattr(config, "tokenizer", None)
        
        # 6) 새로 정의된 레이어 초기화
        self.init_weights()
    
    def preprocess_image_tokens(self, input_ids, image_token_id=None):
        """
        텍스트 입력에서 이미지 토큰을 특수 인덱스로 변환
        """
        if image_token_id is None and hasattr(self, "tokenizer") and self.tokenizer is not None:
            image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        
        if image_token_id is None:
            return input_ids
        
        # 입력 ID 복사본 생성
        processed_input_ids = input_ids.clone()
        
        # 이미지 토큰을 특수 인덱스로 교체
        mask = (processed_input_ids == image_token_id)
        processed_input_ids[mask] = self.image_token_index
        
        return processed_input_ids
    
    def _replace_image_tokens_with_features(
        self, 
        input_ids, 
        labels=None, 
        attention_mask=None, 
        vision_embeds=None,
        ignore_index=IGNORE_INDEX
    ):
        """
        입력 텍스트 시퀀스의 이미지 토큰을 비전 임베딩으로 대체
        
        Args:
            input_ids: 입력 토큰 ID (B, L)
            labels: 레이블 토큰 ID (B, L)
            attention_mask: 어텐션 마스크 (B, L)
            vision_embeds: 비전 임베딩 (B, P, S, D) 또는 (B, S, D)
            ignore_index: 손실 계산 무시 인덱스
            
        Returns:
            combined_embeds: 결합된 임베딩
            combined_labels: 결합된 레이블
            combined_mask: 결합된 어텐션 마스크
            position_ids: 위치 인덱스
        """
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # 어텐션 마스크가 없으면 패딩 토큰을 제외한 모든 토큰에 마스크 적용
        if attention_mask is None and hasattr(self, "tokenizer") and self.tokenizer is not None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id)
        elif attention_mask is None:
            attention_mask = torch.ones_like(input_ids).bool()
        else:
            attention_mask = attention_mask.bool()
            
        # 레이블이 없으면 입력 ID와 동일하게 설정
        if labels is None:
            labels = input_ids.clone()
        
        # 입력 임베딩 가져오기
        embed_tokens_fn = self.language_model.get_input_embeddings()
        
        # 패딩 제거하고 실제 토큰만 처리
        ids_list = [ids[mask] for ids, mask in zip(input_ids, attention_mask)]
        lbl_list = [lbl[mask] for lbl, mask in zip(labels, attention_mask)]
        
        # 배치별 처리
        seq_embeds, seq_labels = [], []
        is_panorama = len(vision_embeds.shape) == 4  # (B, P, S, D) 형태인지 확인
        
        for b in range(batch_size):
            ids = ids_list[b]
            lbls = lbl_list[b]
            
            # 비전 임베딩 가져오기 - 배치별로 해당하는 임베딩 사용
            if is_panorama:
                # (B, P, S, D) -> (P*S, D)로 변환
                B, P, S, D = vision_embeds.shape
                vis_emb = vision_embeds[b].reshape(-1, D)
            else:
                # (B, S, D) 형태
                vis_emb = vision_embeds[b]
            
            if vis_emb.dim() == 1:
                vis_emb = vis_emb.unsqueeze(0)
                
            # 이미지 토큰 위치 찾기
            img_pos = (ids == self.image_token_index).nonzero(as_tuple=False).flatten()
            if img_pos.numel() == 0:
                # 이미지 토큰이 없으면 그냥 텍스트만 처리
                seq_embeds.append(embed_tokens_fn(ids))
                seq_labels.append(lbls)
                continue
                
            # 분할 지점 설정 (이미지 토큰 위치 기준)
            split_pts = torch.cat([torch.tensor([-1], device=device), img_pos, torch.tensor([ids.size(0)], device=device)])
            seg_emb, seg_lbl = [], []
            
            # 세그먼트 별로 처리
            for i in range(split_pts.numel() - 1):
                s = split_pts[i] + 1
                e = split_pts[i + 1]
                txt_ids = ids[s:e]
                txt_lbl = lbls[s:e]
                
                # 텍스트 세그먼트 임베딩
                if txt_ids.numel() > 0:
                    txt_emb = embed_tokens_fn(txt_ids)
                else:
                    # 빈 세그먼트
                    txt_emb = vis_emb[:0]
                
                seg_emb.append(txt_emb)
                seg_lbl.append(txt_lbl)
                
                # 이미지 토큰 위치에 비전 임베딩 삽입
                if i < img_pos.numel():
                    seg_emb.append(vis_emb)
                    # 이미지 토큰도 학습에 포함 (이미지 토큰 ID를 레이블로 사용)
                    seg_lbl.append(torch.full((vis_emb.size(0),), self.image_token_index, dtype=lbls.dtype, device=device))
            
            # 세그먼트 결합
            seq_embeds.append(torch.cat(seg_emb, dim=0))
            seq_labels.append(torch.cat(seg_lbl, dim=0))
        
        # 최대 길이 패딩 처리
        max_length = max(e.size(0) for e in seq_embeds)
        embed_dim = seq_embeds[0].size(1)
        
        # 패딩을 위한 초기화
        pad_embeds = []
        pad_labels = torch.full((batch_size, max_length), ignore_index, dtype=labels.dtype, device=device)
        pad_mask = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
        pos_ids = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)
        
        # 패딩 함수
        pad_emb = lambda n: torch.zeros((n, embed_dim), dtype=seq_embeds[0].dtype, device=device)
        
        # 각 시퀀스를 최대 길이에 맞게 패딩
        padding_side = "right"  # 기본값으로 오른쪽 패딩
        if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "padding_side"):
            padding_side = self.tokenizer.padding_side
            
        for i, (emb, lab) in enumerate(zip(seq_embeds, seq_labels)):
            cur_len = emb.size(0)
            if padding_side == "left":
                pad = pad_emb(max_length - cur_len)
                emb = torch.cat([pad, emb], dim=0)
                pad_labels[i, -cur_len:] = lab
                pad_mask[i, -cur_len:] = 1
                pos_ids[i, -cur_len:] = torch.arange(cur_len, device=device)
            else:  # right padding
                pad = pad_emb(max_length - cur_len)
                emb = torch.cat([emb, pad], dim=0)
                pad_labels[i, :cur_len] = lab
                pad_mask[i, :cur_len] = 1
                pos_ids[i, :cur_len] = torch.arange(cur_len, device=device)
            
            pad_embeds.append(emb)
        
        # 최종 결합 임베딩
        combined_embeds = torch.stack(pad_embeds, dim=0)
        return combined_embeds, pad_labels, pad_mask, pos_ids

    def _combine_embeddings(
        self,
        pixel_values,
        input_ids,
        attention_mask=None,
        labels=None,
        interpolate_pos_encoding=False
    ):
        """
        비전 임베딩과 텍스트 임베딩을 결합하는 포괄적 메서드
        
        Args:
            pixel_values: 이미지 픽셀 값
            input_ids: 입력 토큰 ID
            attention_mask: 어텐션 마스크
            labels: 레이블 토큰 ID
            interpolate_pos_encoding: 위치 인코딩 보간 여부
            
        Returns:
            combined_embeds: 결합된 임베딩
            combined_labels: 결합된 레이블
            combined_mask: 결합된 어텐션 마스크
            position_ids: 위치 인덱스
        """
        # 1. 이미지 토큰 전처리
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
            processed_input_ids = self.preprocess_image_tokens(input_ids, image_token_id)
            if labels is not None:
                processed_labels = self.preprocess_image_tokens(labels, image_token_id)
            else:
                processed_labels = None
        else:
            processed_input_ids = input_ids
            processed_labels = labels
        
        # 2. 비전 모델 처리
        # 입력 텐서 형태 확인 및 조정
        pixel_values_reshaped, batch_size, P, is_panorama = self.process_input(pixel_values)
        
        # Vision encoder 처리
        vision_kwargs = {"return_dict": True}
        if "clip" in str(self.vision_model.__class__).lower() and interpolate_pos_encoding:
            vision_kwargs["interpolate_pos_encoding"] = interpolate_pos_encoding
        
        vision_outputs = self.vision_model(
            pixel_values=pixel_values_reshaped,
            **vision_kwargs
        )
        
        # Vision 출력에서 특성 추출
        if hasattr(vision_outputs, 'last_hidden_state'):
            vision_embeds = vision_outputs.last_hidden_state  # transformer 기반 출력
        elif isinstance(vision_outputs, tuple) and len(vision_outputs) > 0:
            vision_embeds = vision_outputs[0]  # 튜플 형태 출력 처리
        else:
            vision_embeds = vision_outputs  # 기타 출력
        
        # 파노라마 이미지인 경우 reshape
        if is_panorama:
            _, S, D = vision_embeds.shape
            vision_embeds = vision_embeds.view(batch_size, P, S, D)
        
        # 3. Projector 적용
        if self.projector is not None:
            if is_panorama:
                B, P, S, D = vision_embeds.shape
                vision_embeds = vision_embeds.view(-1, D)  # 마지막 차원만 투영
                vision_embeds = self.projector(vision_embeds)
                vision_embeds = vision_embeds.view(B, P, S, -1)  # 원래 형태로 복원
            else:
                vision_embeds = self.projector(vision_embeds)
        
        # 4. 텍스트와 이미지 임베딩 결합
        return self._replace_image_tokens_with_features(
            input_ids=processed_input_ids,
            labels=processed_labels,
            attention_mask=attention_mask,
            vision_embeds=vision_embeds
        )
    
    def process_input(self, pixel_values):
        """
        입력 이미지 데이터를 처리하는 헬퍼 메서드
        
        Args:
            pixel_values: 입력 이미지 픽셀 값
            
        Returns:
            pixel_values_reshaped: 재구성된 픽셀 값
            batch_size: 배치 크기
            P: 패치/뷰의 수
            is_panorama: 파노라마 이미지 여부
        """
        # 입력 텐서 형태 확인 및 조정
        if len(pixel_values.shape) == 5:  # [B, P, C, H, W]
            B, P, C, H, W = pixel_values.shape
            return pixel_values.view(B * P, C, H, W), B, P, True
        elif len(pixel_values.shape) == 4:  # [B, C, H, W]
            B = pixel_values.shape[0]
            return pixel_values, B, 1, False
        else:
            raise ValueError(f"Unexpected pixel_values shape: {pixel_values.shape}. " 
                          f"Expected [B, P, C, H, W] or [B, C, H, W]")
            
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **kwargs
        ):
        """
        Forward pass for the PanoVLM model.
        
        Args:
            pixel_values (torch.FloatTensor): 비전 모델을 위한 입력 픽셀 값
            input_ids (Optional[torch.LongTensor]): 언어 모델을 위한 입력 토큰 ID
            attention_mask (Optional[torch.LongTensor]): 언어 모델을 위한 어텐션 마스크
            labels (Optional[torch.LongTensor]): 언어 모델 학습을 위한 레이블
            interpolate_pos_encoding (bool): 위치 인코딩 보간 여부
            **kwargs: 추가 인자
            
        Returns:
            torch.FloatTensor: 언어 모델의 출력 로짓
        """
        # FP16 자동 변환 (메모리 효율성)
        autocast_ctx = torch.cuda.amp.autocast() if self.use_fp16 and torch.cuda.is_available() else nullcontext()
        
        with autocast_ctx:
            # 비전 임베딩과 텍스트 임베딩 결합
            inputs_embeds, combined_labels, combined_mask, position_ids = self._combine_embeddings(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                interpolate_pos_encoding=interpolate_pos_encoding
            )
            
            # 명시적으로 requires_grad=True 설정하여 그래디언트 계산 보장
            if self.training and not inputs_embeds.requires_grad:
                inputs_embeds.requires_grad_(True)
            
            # LLM 모델에 전달할 인자 구성
            model_kwargs = {
                "attention_mask": combined_mask if attention_mask is not None else None,
                "position_ids": position_ids,
                "return_dict": True
            }
            
            # 학습 모드에서는 레이블 전달
            if labels is not None:
                model_kwargs["labels"] = combined_labels
                
            # 언어 모델 호출
            return self.language_model(
                inputs_embeds=inputs_embeds,
                **model_kwargs
            )
    

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        interpolate_pos_encoding: bool = False,
        **generate_kwargs,
    ):
        """
        입력 픽셀 값과 선택적 텍스트 입력을 기반으로 텍스트 생성
        
        Args:
            pixel_values (torch.FloatTensor): 비전 모델을 위한 입력 픽셀 값
            input_ids (Optional[torch.LongTensor]): 언어 모델을 위한 입력 토큰 ID
            attention_mask (Optional[torch.LongTensor]): 언어 모델을 위한 어텐션 마스크
            interpolate_pos_encoding (bool): 위치 인코딩 보간 여부
            **generate_kwargs: 생성을 위한 추가 인자
            
        Returns:
            torch.LongTensor: 생성된 토큰 ID
        """
        # FP16 자동 변환 (메모리 효율성)
        autocast_ctx = torch.cuda.amp.autocast() if self.use_fp16 and torch.cuda.is_available() else nullcontext()
        
        with autocast_ctx:
            # 비전 임베딩과 텍스트 임베딩 결합
            inputs_embeds, _, combined_mask, position_ids = self._combine_embeddings(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                interpolate_pos_encoding=interpolate_pos_encoding
            )
            
            # 생성에 필요한 model_kwargs 구성
            model_kwargs = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": combined_mask if attention_mask is not None else None,
                "position_ids": position_ids,
            }
            
            # 언어 모델의 generate 메서드 호출
            return self.language_model.generate(
                input_ids=None,  # inputs_embeds를 사용하므로 input_ids는 None
                **model_kwargs,
                **generate_kwargs
            )
    
    def gradient_checkpointing_enable(self, **kwargs):
        """
        Enable gradient checkpointing for the model.
        Trainer는 모델에 이 메서드가 있어야 gradient_checkpointing=True 설정을 활용할 수 있습니다.
        
        gradient_checkpointing의 활성화 상태를 개선하여 메모리 효율성을 높입니다.
        프로젝터 레이어도 checkpointing이 가능한 경우 활성화합니다.
        """
        print("\n===== Gradient Checkpointing 활성화 =====")
        
        # 1. Vision 모델에 gradient checkpointing 활성화
        try:
            if hasattr(self.vision_model, "gradient_checkpointing_enable"):
                self.vision_model.gradient_checkpointing_enable()
                print("✅ Vision 모델: gradient_checkpointing_enable 메서드 사용 성공")
            elif hasattr(self.vision_model, "config") and hasattr(self.vision_model.config, "gradient_checkpointing"):
                self.vision_model.config.gradient_checkpointing = True
                print("✅ Vision 모델: config.gradient_checkpointing 설정 성공")
            else:
                # 대체 방법: 직접 속성 설정
                self.vision_model.gradient_checkpointing = True
                print("✅ Vision 모델: 직접 속성 설정 성공")
                
            # Vision 모델 파라미터 동결 확인
            if not any(p.requires_grad for p in self.vision_model.parameters()):
                print("📝 참고: Vision 모델은 동결 상태라 그래디언트 계산이 필요하지 않습니다.")
        except Exception as e:
            print(f"⚠️ Vision 모델 gradient checkpointing 활성화 실패: {str(e)}")
            
        # 2. Language 모델에 gradient checkpointing 활성화 
        try:
            if hasattr(self.language_model, "gradient_checkpointing_enable"):
                self.language_model.gradient_checkpointing_enable()
                print("✅ Language 모델: gradient_checkpointing_enable 메서드 사용 성공")
            elif hasattr(self.language_model, "config") and hasattr(self.language_model.config, "gradient_checkpointing"):
                self.language_model.config.gradient_checkpointing = True
                print("✅ Language 모델: config.gradient_checkpointing 설정 성공")
            else:
                # 대체 방법: 직접 속성 설정
                self.language_model.gradient_checkpointing = True
                print("✅ Language 모델: 직접 속성 설정 성공")
                
            # Language 모델 파라미터 동결 확인
            if not any(p.requires_grad for p in self.language_model.parameters()):
                print("📝 참고: Language 모델은 동결 상태라 그래디언트 계산이 필요하지 않습니다.")
        except Exception as e:
            print(f"⚠️ Language 모델 gradient checkpointing 활성화 실패: {str(e)}")
        
        # 3. Projector 모듈에 gradient checkpointing 적용 (가능한 경우)
        if self.projector is not None:
            try:
                # Linear 레이어와 같은 단일 연산이 아닌 경우에만 적용
                if isinstance(self.projector, nn.Sequential) and len(list(self.projector.children())) > 1:
                    # checkpoint 함수를 직접 적용할 수 있는 형태로 랩핑
                    original_forward = self.projector.forward
                    
                    def checkpointed_forward(*args, **kwargs):
                        if torch.is_grad_enabled() and any(p.requires_grad for p in self.projector.parameters()):
                            from torch.utils.checkpoint import checkpoint
                            return checkpoint(original_forward, *args, **kwargs)
                        else:
                            return original_forward(*args, **kwargs)
                    
                    # 프로젝터의 forward 메서드 교체
                    self.projector.forward = checkpointed_forward
                    print("✅ Projector: 커스텀 gradient checkpointing 적용 성공")
                    
                    # Projector 모델 학습 상태 확인
                    if any(p.requires_grad for p in self.projector.parameters()):
                        print("📝 Projector는 학습 상태이므로 gradient checkpointing이 적용됩니다.")
                else:
                    print("📝 Projector는 단순 구조라 gradient checkpointing이 불필요합니다.")
            except Exception as e:
                print(f"⚠️ Projector gradient checkpointing 설정 실패: {str(e)}")
            
        # 4. 캐시 사용을 비활성화하여 메모리 절약
        if hasattr(self.language_model, "config") and hasattr(self.language_model.config, "use_cache"):
            self.language_model.config.use_cache = False
            print("✅ Language 모델 use_cache 비활성화됨")
        
        # 5. 기타 사용자 제공 kwargs 처리
        for key, value in kwargs.items():
            print(f"📝 추가 설정: {key}={value}")
        
        self.gradient_checkpointing = True
        print("✅ 모델 전체 gradient_checkpointing 활성화 완료")
        print("========================================\n")

    def gradient_checkpointing_disable(self):
        """
        Disable gradient checkpointing for the model.
        
        모든 컴포넌트의 gradient checkpointing을 비활성화합니다.
        """
        print("\n===== Gradient Checkpointing 비활성화 =====")
        
        # 1. Vision 모델 gradient checkpointing 비활성화
        try:
            if hasattr(self.vision_model, "gradient_checkpointing_disable"):
                self.vision_model.gradient_checkpointing_disable()
                print("✅ Vision 모델: gradient_checkpointing_disable 메서드 호출 성공")
            elif hasattr(self.vision_model, "config") and hasattr(self.vision_model.config, "gradient_checkpointing"):
                self.vision_model.config.gradient_checkpointing = False
                print("✅ Vision 모델: config.gradient_checkpointing 비활성화")
            elif hasattr(self.vision_model, "gradient_checkpointing"):
                self.vision_model.gradient_checkpointing = False
                print("✅ Vision 모델: 직접 속성 비활성화")
        except Exception as e:
            print(f"⚠️ Vision 모델 gradient checkpointing 비활성화 실패: {str(e)}")
        
        # 2. Language 모델 gradient checkpointing 비활성화
        try:
            if hasattr(self.language_model, "gradient_checkpointing_disable"):
                self.language_model.gradient_checkpointing_disable()
                print("✅ Language 모델: gradient_checkpointing_disable 메서드 호출 성공")
            elif hasattr(self.language_model, "config") and hasattr(self.language_model.config, "gradient_checkpointing"):
                self.language_model.config.gradient_checkpointing = False
                print("✅ Language 모델: config.gradient_checkpointing 비활성화")
            elif hasattr(self.language_model, "gradient_checkpointing"):
                self.language_model.gradient_checkpointing = False
                print("✅ Language 모델: 직접 속성 비활성화")
        except Exception as e:
            print(f"⚠️ Language 모델 gradient checkpointing 비활성화 실패: {str(e)}")
        
        # 3. Projector 모듈에 적용된 gradient checkpointing 원복 (필요한 경우)
        if self.projector is not None and isinstance(self.projector, nn.Sequential):
            try:
                # 원래 forward 메서드가 랩핑되었다면 원래 메서드로 복구
                if hasattr(self.projector, "_original_forward"):
                    self.projector.forward = self.projector._original_forward
                    delattr(self.projector, "_original_forward")
                    print("✅ Projector: 원래 forward 메서드로 복구됨")
            except Exception as e:
                print(f"⚠️ Projector gradient checkpointing 비활성화 실패: {str(e)}")
        
        # 4. 캐시 사용 재활성화
        if hasattr(self.language_model, "config") and hasattr(self.language_model.config, "use_cache"):
            self.language_model.config.use_cache = True
            print("✅ Language 모델 use_cache 재활성화됨")
        
        self.gradient_checkpointing = False
        print("✅ 모델 전체 gradient_checkpointing 비활성화 완료")
        print("==========================================\n")
        
    def set_tokenizer(self, tokenizer):
        """
        모델에 토크나이저를 설정
        
        Args:
            tokenizer: 허깅페이스 토크나이저 인스턴스
        """
        self.tokenizer = tokenizer
        
        # 특수 토큰 등록 확인 및 처리
        special_tokens = {
            "additional_special_tokens": [
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IMAGE_PATCH_TOKEN,
                DEFAULT_IM_START_TOKEN,
                DEFAULT_IM_END_TOKEN,
            ]
        }
        
        # 토크나이저에 특수 토큰 추가
        num_added = 0
        if hasattr(tokenizer, "add_special_tokens"):
            num_added = tokenizer.add_special_tokens(special_tokens)
            
        # 임베딩 크기 조정
        if num_added > 0 and hasattr(self.language_model, "resize_token_embeddings"):
            self.language_model.resize_token_embeddings(len(tokenizer))
            
        # 패딩 관련 설정
        if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
            if hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                
        return tokenizer