from typing import Optional, Union, Tuple
from transformers import PretrainedConfig, AutoConfig, AutoModel, PreTrainedModel
import importlib
from typing import Union

def get_vision_pretrained_lm(config) -> Union[type, None]:
    """
    설정 정보를 기반으로 동적으로 모델 클래스를 임포트
    """
    model_type = config.get('model_type', '').lower()
    
    # 모델 타입별 클래스 매핑
    model_mappings = {
        'gemma': 'transformers.models.gemma.modeling_gemma.GemmaForCausalLM',
        'gemma3_text': 'transformers.models.gemma3_text.modeling_gemma3_text.Gemma3TextForCausalLM',
        'opt': 'transformers.models.opt.modeling_opt.OPTForCausalLM',
        't5': 'transformers.models.t5.modeling_t5.T5ForConditionalGeneration',
    }
    
    # 부분 매치를 위한 키워드 매핑
    keyword_mappings = {
        'gemma': 'transformers.AutoModelForCausalLM',
        'opt': 'transformers.AutoModelForCausalLM', 
        't5': 'transformers.AutoModelForSeq2SeqLM',
    }
    
    try:
        # 정확한 매치 확인
        if model_type in model_mappings:
            module_path, class_name = model_mappings[model_type].rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        
        # 키워드 매치 확인
        for keyword, class_path in keyword_mappings.items():
            if keyword in model_type:
                if '.' in class_path:
                    module_path, class_name = class_path.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    return getattr(module, class_name)
                else:
                    return eval(class_path)
        
        # 기본값으로 AutoModel 사용
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM
        
    except (ImportError, AttributeError) as e:
        print(f"모델 클래스 로딩 실패: {e}")
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM


class PanoVLM(PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.vision_model = AutoModel.from_pretrained(config.vision_model_name_or_path, config=config.vision_config)
        self.language_model = AutoModel.from_pretrained(config.language_model_name_or_path, config=config.language_config)
        self.projector = None  # Define your projector here if needed
        self.init_weights()