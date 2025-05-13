from transformers import PretrainedConfig, AutoConfig
from typing import Callable, Optional

class VisionLanguageConfig(PretrainedConfig):
    """
    Custom configuration for Vision-Language Models, including nested
    configs for vision and language backbones.

    Attributes:
        vision_model_name (str): HF model ID for vision encoder.
        language_model_name (str): HF model ID for language model.
        vision_config (PretrainedConfig): Parsed config of the vision encoder.
        language_config (PretrainedConfig): Parsed config of the language model.
        projector_type (str): Type of projector for vision→language mapping.
        use_resampler (bool): Whether to apply a resampler to vision features.
        compressor_type (Optional[str]): Name of compressor strategy.
        mm_spatial_pool_mode (str): Spatial pooling mode: "average" | "max" | "bilinear".
        mm_patch_merge_type (str): Patch merge strategy: "maxpool2x2", "unpad", "unpad_anyres_max".
        max_num_patches (Optional[int]): Maximum patches after merge; triggers dynamic downsampling.
        num_image_tokens (int): Number of image token placeholders in prefix fusion.
        freeze_vision (bool): Whether to freeze the vision encoder parameters.
        freeze_llm (bool): Whether to freeze the language model parameters.
    """

    model_type = "vision_language"

    def __init__(
        self,
        vision_model_name: str = "openai/clip-vit-large-patch14",
        language_model_name: str = "meta-llama/Meta-Llama-3-8B",
        projector_type: str = "mlp2x_gelu",
        use_resampler: bool = False,
        compressor_type: Optional[str] = None,
        mm_spatial_pool_mode: str = "average",
        # mm_patch_merge_type: str = "maxpool2x2",
        mm_newline_position: str = "grid",
        freeze_vision: bool = True,
        freeze_llm: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Backbone model identifiers
        self.vision_model_name = vision_model_name
        self.language_model_name = language_model_name

        # Nested HF configs for each backbone
        self.vision_config = AutoConfig.from_pretrained(self.vision_model_name)
        self.vision_config = self.vision_config.vision_config if hasattr(self.vision_config, "vision_config") else self.vision_config
        self.language_config = AutoConfig.from_pretrained(self.language_model_name)
        self.language_config = self.language_config.text_config if hasattr(self.language_config, "text_config") else self.language_config
        

        # Projector and compressor settings
        self.projector_type = projector_type
        self.use_resampler = use_resampler
        self.compressor_type = compressor_type

        # Spatial pooling & patch merging
        self.mm_spatial_pool_mode = mm_spatial_pool_mode
        # self.mm_patch_merge_type = mm_patch_merge_type
        self.mm_newline_position = mm_newline_position

        # Prefix fusion
        self.img_size = self.vision_config.image_size

        # Parameter freezing
        self.freeze_vision = freeze_vision
        self.freeze_llm = freeze_llm

    def to_dict(self):
        """Serializes this instance to a dict, including nested backbone configs."""
        output = super().to_dict()
        extra = {
            "vision_model_name": self.vision_model_name,
            "language_model_name": self.language_model_name,
            "vision_config": self.vision_config.to_dict(),
            "language_config": self.language_config.to_dict(),
            "projector_type": self.projector_type,
            "use_resampler": self.use_resampler,
            "compressor_type": self.compressor_type,
            "mm_spatial_pool_mode": self.mm_spatial_pool_mode,
            # "mm_patch_merge_type": self.mm_patch_merge_type,
            "mm_newline_position": self.mm_newline_position,
            "img_size": self.img_size,
            "freeze_vision": self.freeze_vision,
            "freeze_llm": self.freeze_llm,
        }
        output.update(extra)
        return output
