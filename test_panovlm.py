from src.models.panovlm import PanoVLM
from src.models.panovlm_config import PanoVLMConfig
from torchinfo import summary

def test_panovlm_initialization():
    config = PanoVLMConfig(
        vision_model_name_or_path="facebook/dinov2-small",
        language_model_name_or_path="google/gemma-3-4b-it"
    )
    
    model = PanoVLM(config)
    print("Model summary:")
    print("===" * 20)
    print(model.language_model)
    print("===" * 20)
    print(model.vision_model)
    assert model.vision_model is not None, "Vision model should be initialized."
    assert model.language_model is not None, "Language model should be initialized."
    assert hasattr(model, 'projector'), "Projector should be defined."
    
    # Check if the language model is correctly extracted
    text_model = model.language_model
    assert text_model is not None, "Text model should be extracted from the full VLM."
    
    print("✅ PanoVLM initialization test passed.")
    
if __name__ == "__main__":
    test_panovlm_initialization()
