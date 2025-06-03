from transformers import PretrainedConfig, AutoConfig

@AutoConfig.register("panovlm")
class PanoVLMConfig(PretrainedConfig):
    model_type = "panovlm"

    def __init__(self, 
                 vision_model_name_or_path: str = "google/gemma-3-4b-it",
                 language_model_name_or_path: str = "google/gemma-3-4b-it",
                 vision_config=None,
                 language_config=None,
                 projector_config=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.vision_model_name_or_path = vision_model_name_or_path
        self.language_model_name_or_path = language_model_name_or_path
        self.vision_config = vision_config if vision_config is not None else AutoConfig.from_pretrained(vision_model_name_or_path)
        self.language_config = language_config if language_config is not None else AutoConfig.from_pretrained(language_model_name_or_path)
        self.projector_config = projector_config