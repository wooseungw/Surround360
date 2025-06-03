from typing import Optional, Union, Tuple, Any
from transformers import PretrainedConfig, AutoConfig, AutoModel, PreTrainedModel, AutoModelForCausalLM
import transformers

model_mapping = {
    "google/gemma-3-4b-it": transformers.Gemma3ForConditionalGeneration,
    "Qwen/Qwen2.5-VL-3B-Instruct": transformers.Qwen2VLForConditionalGeneration,
}

def get_model_class(model_name: str):
    """Get the appropriate model class from the transformers library based on model name."""
    return model_mapping.get(model_name, transformers.AutoModelForCausalLM)

def extract_text_model(full_vlm: Any) -> Any:
    """
    전체 VLM 인스턴스(full_vlm)로부터 순수 텍스트 생성 LM 부분만 추출하여 반환한다.
    1) Gemma3ForConditionalGeneration 계열은 `.language_model` 속성
    2) Qwen2VLForConditionalGeneration 계열은 `.model` 속성
    3) 그 외에는 전체 모델 자체를 텍스트 생성용으로 간주하여 그대로 반환

    새로운 VLM 모델을 추가할 때는, 아래 elif 블록에
        if isinstance(full_vlm, transformers.XXXForConditionalGeneration):
            return full_vlm.YYY
    와 같은 분기만 한 줄 추가하면 된다.
    """
    # 1) Gemma3 계열
    if isinstance(full_vlm, transformers.Gemma3ForConditionalGeneration):
        return full_vlm.language_model
    # 2) Qwen2.5-VL 계열
    elif isinstance(full_vlm, transformers.Qwen2VLForConditionalGeneration):
        return full_vlm.model
    # 3) 그 외 (AutoModel 따위로 불러온 경우)
    else:
        return full_vlm

class PanoVLM(PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        # 1) Vision encoder 로드
        self.vision_model = AutoModel.from_pretrained(
            config.vision_model_name_or_path,
            config=config.vision_config
        )
        # (a) 전체 VLM 클래스 인스턴스화
        full_vlm = get_model_class(config.language_model_name_or_path).from_pretrained(
            config.language_model_name_or_path,
            config=config.language_config
        )
        self.language_model = extract_text_model(full_vlm)

        # 3) Projector (필요 시 정의)
        self.projector = None  # 예: nn.Linear 또는 Sequential 등 직접 정의

        # 4) 새로 정의된 레이어(예: projector) 초기화
        self.init_weights()