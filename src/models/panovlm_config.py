from transformers import PretrainedConfig, AutoConfig

# PanoVLM에서 사용하는 상수
IMAGE_TOKEN_INDEX = -200  # 이미지 토큰 전용 인덱스
DEFAULT_IMAGE_TOKEN = "<image>"

# ProjectorConfig를 위한 데이터클래스
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ProjectorConfig:
    """
    프로젝터 레이어 구성 클래스.
    비전 임베딩을 언어 모델 임베딩 공간으로 변환하는 프로젝터의 구성 옵션을 정의합니다.
    """
    type: str = "mlp2x_gelu"  # 프로젝터 타입 (linear, mlp2x_gelu, pooler 등)
    in_features: int = 1024   # 입력 특성 차원 (비전 모델의 출력 차원)
    out_features: int = 4096  # 출력 특성 차원 (언어 모델의 임베딩 차원)

# @AutoConfig.register("panovlm", config=PretrainedConfig,)
class PanoVLMConfig(PretrainedConfig):
    model_type = "panovlm"

    def __init__(self, 
                 vision_model_name_or_path: str = "google/siglip2-base-patch16-224",
                 language_model_name_or_path: str = "google/gemma-3-4b-it",
                 vision_config=None,
                 language_config=None,
                 projector_config=None,
                 image_token_index: int = IMAGE_TOKEN_INDEX,
                 default_image_token: str = DEFAULT_IMAGE_TOKEN,
                 use_fp16: bool = True,
                 gradient_checkpointing: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.vision_model_name_or_path = vision_model_name_or_path
        self.language_model_name_or_path = language_model_name_or_path
        self.vision_config = vision_config if vision_config is not None else AutoConfig.from_pretrained(vision_model_name_or_path)
        self.language_config = language_config if language_config is not None else AutoConfig.from_pretrained(language_model_name_or_path)
        
        # 프로젝터 구성 설정
        if projector_config is None:
            # 기본 프로젝터 구성
            # 다양한 언어 모델 설정에서 임베딩 차원 얻기
            if hasattr(self.language_config, 'hidden_size'):
                lang_dim = self.language_config.hidden_size
            elif hasattr(self.language_config, 'model_dim'):
                lang_dim = self.language_config.model_dim  # Gemma-3 모델
            elif hasattr(self.language_config, 'hidden_dim'):
                lang_dim = self.language_config.hidden_dim
            elif hasattr(self.language_config, 'd_model'):
                lang_dim = self.language_config.d_model  # 일부 transformer 모델
            else:
                lang_dim = 2560  # 기본값 (Gemma-3-4B)
                
            # 비전 모델의 임베딩 차원 얻기
            if hasattr(self.vision_config, 'hidden_size'):
                vision_dim = self.vision_config.hidden_size
            elif hasattr(self.vision_config, 'embed_dim'):
                vision_dim = self.vision_config.embed_dim  # CLIP 모델
            else:
                vision_dim = 384  # DinoV2-small 기본값
                
            self.projector_config = ProjectorConfig(
                type="mlp2x_gelu",
                in_features=vision_dim,
                out_features=lang_dim,
            )
        else:
            self.projector_config = projector_config
        
        # 이미지 토큰 설정
        self.image_token_index = image_token_index
        self.default_image_token = default_image_token
        
        # 메모리 효율성 설정
        self.use_fp16 = use_fp16
        self.gradient_checkpointing = gradient_checkpointing