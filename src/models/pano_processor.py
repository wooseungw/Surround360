from typing import Dict, List, Optional, Union
import torch
from transformers import (
    ProcessorMixin, 
    AutoImageProcessor, 
    AutoTokenizer,
    BatchEncoding
)
from transformers.processing_utils import ProcessingKwargs
from transformers.tokenization_utils_base import AddedToken
from transformers.image_utils import ImageInput
from transformers.utils import TensorType, logging

logger = logging.get_logger(__name__)


class PanoVLMProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "truncation": False,
            "max_length": None,
            "return_tensors": None,
        },
        "images_kwargs": {
            "return_tensors": None,
        },
    }


class PanoVLMProcessor(ProcessorMixin):
    """
    확장 가능한 파노라마 VLM 프로세서.
    임의의 vision processor와 tokenizer를 결합할 수 있습니다.
    
    Args:
        vision_processor_name (str): Vision processor 모델명 (예: "openai/clip-vit-large-patch14-336")
        tokenizer_name (str): Tokenizer 모델명 (예: "meta-llama/Llama-2-7b-chat-hf")
        image_token (str): 이미지를 나타내는 특별 토큰 (기본값: "<image>")
        patch_size (int): 이미지를 패치로 나눌 때의 크기
        vision_feature_layer (int): Vision encoder에서 사용할 레이어 (-1: 마지막 레이어)
    """
    
    attributes = ["image_processor", "tokenizer"]
    
    def __init__(
        self,
        vision_processor_name: str = "openai/clip-vit-large-patch14-336",
        tokenizer_name: str = "meta-llama/Llama-2-7b-chat-hf",
        image_token: str = "<image>",
        patch_size: Optional[int] = None,
        vision_feature_layer: int = -1,
        **kwargs
    ):
        # Vision processor와 tokenizer 로드
        self.image_processor = AutoImageProcessor.from_pretrained(vision_processor_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # 패딩 토큰 설정 (LLaMA 등에서 필요)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 이미지 토큰 추가
        self.image_token = AddedToken(image_token, normalized=False, special=True)
        self.tokenizer.add_tokens([self.image_token], special_tokens=True)
        
        # 설정값 저장
        self.patch_size = patch_size
        self.vision_feature_layer = vision_feature_layer
        self.vision_processor_name = vision_processor_name
        self.tokenizer_name = tokenizer_name
        
        super().__init__(self.image_processor, self.tokenizer, **kwargs)
    
    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> BatchEncoding:
        """
        이미지와 텍스트를 처리하여 모델 입력 형태로 변환합니다.
        
        Args:
            images: 처리할 이미지(들)
            text: 처리할 텍스트(들)
            **kwargs: 추가 처리 옵션
            
        Returns:
            BatchEncoding: 모델에 전달할 수 있는 형태의 인코딩된 입력
        """
        if images is None and text is None:
            raise ValueError("이미지 또는 텍스트 중 하나는 반드시 제공해야 합니다.")
        
        # kwargs 처리
        output_kwargs = self._merge_kwargs(
            PanoVLMProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        
        return_tensors = output_kwargs["common_kwargs"].get("return_tensors", None)
        encoding = BatchEncoding()
        
        # 텍스트 처리
        if text is not None:
            if isinstance(text, str):
                text = [text]
            
            # LLaVA 스타일: 텍스트에서 <image> 토큰 위치 찾기
            processed_text = []
            for t in text:
                # 이미지가 있다면 <image> 토큰을 텍스트에 삽입
                if images is not None and self.image_token.content not in t:
                    # 기본적으로 텍스트 시작 부분에 이미지 토큰 추가
                    t = f"{self.image_token.content} {t}"
                processed_text.append(t)
            
            text_encoding = self.tokenizer(
                processed_text,
                **output_kwargs["text_kwargs"],
                return_tensors=return_tensors
            )
            encoding.update(text_encoding)
        
        # 이미지 처리
        if images is not None:
            # 파노라마 이미지의 경우 특별한 처리가 필요할 수 있음
            image_encoding = self.image_processor(
                images,
                **output_kwargs["images_kwargs"],
                return_tensors=return_tensors
            )
            encoding.update(image_encoding)
        
        return encoding
    
    def batch_decode(self, *args, **kwargs):
        """토크나이저의 batch_decode 메서드를 위임합니다."""
        return self.tokenizer.batch_decode(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        """토크나이저의 decode 메서드를 위임합니다."""
        return self.tokenizer.decode(*args, **kwargs)
    
    @property
    def model_input_names(self):
        """모델 입력에 필요한 키 이름들을 반환합니다."""
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
    
    def save_pretrained(self, save_directory, **kwargs):
        """프로세서 설정을 저장합니다."""
        # 기본 저장 로직
        super().save_pretrained(save_directory, **kwargs)
        
        # 추가 설정 정보 저장
        import json
        import os
        
        config = {
            "vision_processor_name": self.vision_processor_name,
            "tokenizer_name": self.tokenizer_name,
            "image_token": self.image_token.content,
            "patch_size": self.patch_size,
            "vision_feature_layer": self.vision_feature_layer,
        }
        
        config_path = os.path.join(save_directory, "processor_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """저장된 프로세서를 로드합니다."""
        import json
        import os
        
        # 설정 파일이 있는지 확인
        config_path = os.path.join(pretrained_model_name_or_path, "processor_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            
            return cls(**config, **kwargs)
        else:
            # 기본값으로 초기화
            return cls(**kwargs)


def create_processor_for_model(
    vision_tower_name: str,
    llm_name: str,
    image_token: str = "<image>",
    **kwargs
) -> PanoVLMProcessor:
    """
    특정 비전 타워와 LLM 조합에 맞는 프로세서를 생성하는 팩토리 함수
    
    Args:
        vision_tower_name: 비전 인코더 모델명
        llm_name: 언어 모델명
        image_token: 이미지 토큰
        **kwargs: 추가 설정
        
    Returns:
        PanoVLMProcessor: 설정된 프로세서
    """
    return PanoVLMProcessor(
        vision_processor_name=vision_tower_name,
        tokenizer_name=llm_name,
        image_token=image_token,
        **kwargs
    )
