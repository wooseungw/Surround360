import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from src.models.panovlm import PanoVLM
from src.models.panovlm_config import PanoVLMConfig
from torchinfo import summary

def test_panovlm_initialization():
    """PanoVLM 모델의 초기화를 테스트합니다."""
    config = PanoVLMConfig(
        vision_model_name_or_path="facebook/dinov2-small",
        language_model_name_or_path="google/gemma-3-4b-it",
        use_fp16=True
    )
    
    print("🔄 모델 초기화 중...")
    model = PanoVLM(config)
    print("모델 초기화 완료!")
    
    print("\n📋 Model summary:")
    print("===" * 20)
    print(f"🧠 Language Model: {type(model.language_model).__name__}")
    print(f"👁️ Vision Model: {type(model.vision_model).__name__}")
    print(f"🔄 Projector: {model.projector}")
    print("===" * 20)
    
    assert model.vision_model is not None, "Vision model should be initialized."
    assert model.language_model is not None, "Language model should be initialized."
    assert hasattr(model, 'projector'), "Projector should be defined."
    
    # Check if the language model is correctly extracted
    text_model = model.language_model
    assert text_model is not None, "Text model should be extracted from the full VLM."
    
    print("✅ PanoVLM initialization test passed.")
    return model, config

def test_panovlm_forward(model, batch_size=1, num_patches=4, use_cuda=False):
    """모델의 forward pass를 테스트합니다."""
    print("\n🔄 Testing forward pass...")
    
    # 두 가지 형태의 입력 테스트
    print("1. 파노라마 이미지 입력 테스트 ([B, P, C, H, W])")
    
    # 예시 입력 텐서 생성 (파노라마 이미지 시뮬레이션)
    # [B, P, C, H, W] 형태의 텐서 생성 (배치, 패치, 채널, 높이, 너비)
    batch_size = batch_size  # 배치 크기
    num_patches = num_patches  # 파노라마 패치 수
    channels = 3  # RGB
    height = 224  # 높이
    width = 224  # 너비
    
    # 랜덤 픽셀 값 생성 및 정규화 (더 현실적인 값)
    pixel_values = torch.rand(batch_size, num_patches, channels, height, width)
    # 픽셀값 범위를 [-1, 1]로 조정 (일반적인 정규화)
    pixel_values = (pixel_values * 2) - 1
    
    # 예시 토크나이저 (실제 모델에 맞는 토크나이저 사용 필요)
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", trust_remote_code=True)
        # 텍스트 인코딩
        text = "<image> 이 파노라마 이미지에 대해 설명해주세요."
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        attention_mask = torch.ones_like(input_ids)
    except Exception as e:
        print(f"⚠️ 토크나이저 로드 실패: {e}. 임시 텐서로 대체합니다.")
        input_ids = torch.randint(0, 1000, (batch_size, 10))
        attention_mask = torch.ones_like(input_ids)
    
    # CUDA 사용 여부 확인
    if use_cuda and torch.cuda.is_available():
        model = model.to("cuda")
        pixel_values = pixel_values.to("cuda")
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        print("🔥 Using CUDA for inference")
    
    # 입출력 형태 확인용 출력
    print(f"📊 Input shapes:")
    print(f"  - Pixel values: {pixel_values.shape}")
    print(f"  - Input IDs: {input_ids.shape}")
    
    # Forward pass 실행
    try:
        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        print(f"📊 Output shape: {outputs.shape if hasattr(outputs, 'shape') else 'Not a tensor'}")
        print("✅ Forward pass succeeded!")
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

def test_panovlm_generate(model, batch_size=1, num_patches=4, use_cuda=False):
    """모델의 generate 함수를 테스트합니다."""
    print("\n🔄 Testing generate function...")
    
    # 파노라마 이미지 시뮬레이션
    pixel_values = torch.rand(batch_size, num_patches, 3, 224, 224)
    # 픽셀값 범위를 [-1, 1]로 조정 (일반적인 정규화)
    pixel_values = (pixel_values * 2) - 1
    
    # 프롬프트 설정
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", trust_remote_code=True)
        prompt = "이 파노라마 이미지에 대해 설명해주세요:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        attention_mask = torch.ones_like(input_ids)
    except Exception as e:
        print(f"⚠️ 토크나이저 로드 실패: {e}. 임시 텐서로 대체합니다.")
        input_ids = torch.randint(0, 1000, (batch_size, 10))
        attention_mask = torch.ones_like(input_ids)
    
    # CUDA 사용 여부 확인
    if use_cuda and torch.cuda.is_available():
        model = model.to("cuda")
        pixel_values = pixel_values.to("cuda")
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
    
    # 생성 설정
    gen_kwargs = {
        "max_new_tokens": 20,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    
    # Generate 실행
    try:
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs
            )
        
        print(f"📊 Generated IDs shape: {generated_ids.shape}")
        
        # 토크나이저로 디코딩 (가능한 경우)
        try:
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            print(f"📝 Generated text sample: {generated_text[0][:50]}...")
        except Exception as e:
            print(f"⚠️ 생성된 텍스트 디코딩 실패: {e}")
        
        print("✅ Generate function succeeded!")
        return True
    except Exception as e:
        print(f"❌ Generate function failed: {e}")
        return False

def test_with_real_image(model, image_path=None, use_cuda=False):
    """실제 이미지를 사용하여 모델 테스트를 수행합니다."""
    # 이미지가 제공되지 않은 경우 샘플 이미지 찾기
    if image_path is None:
        sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
        if os.path.exists(sample_dir):
            images = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            if images:
                image_path = os.path.join(sample_dir, images[0])
    
    if not image_path or not os.path.exists(image_path):
        print("❌ 테스트할 이미지를 찾을 수 없습니다. 실제 이미지 테스트를 건너뜁니다.")
        return False
    
    print(f"\n🖼️ Testing with real image: {image_path}")
    
    try:
        # 이미지 로드
        from PIL import Image
        import torchvision.transforms as T
        
        image = Image.open(image_path).convert("RGB")
        
        # 전처리
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 파노라마 이미지 시뮬레이션 - [B, P, C, H, W] 형태
        # 방법 1: 단일 이미지를 여러 패치로 복제 (샘플링된 파노라마 뷰 시뮬레이션)
        img_tensor = transform(image)
        patches = []
        for i in range(4):  # 4개의 패치/뷰 시뮬레이션
            patches.append(img_tensor)
        patches = torch.stack(patches, dim=0).unsqueeze(0)  # [1, 4, 3, 224, 224]
        
        # 토크나이저 및 프롬프트 설정
        try:
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", trust_remote_code=True)
            prompt = "이 파노라마 이미지에 무엇이 보이나요?"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            attention_mask = torch.ones_like(input_ids)
        except Exception as e:
            print(f"⚠️ 토크나이저 로드 실패: {e}. 임시 텐서로 대체합니다.")
            input_ids = torch.randint(0, 1000, (1, 10))
            attention_mask = torch.ones_like(input_ids)
        
        # CUDA 사용 여부 확인
        if use_cuda and torch.cuda.is_available():
            model = model.to("cuda")
            patches = patches.to("cuda")
            input_ids = input_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")
        
        # 생성
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=patches,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=30,
            
            )
        
        # 결과 디코딩
        try:
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            print(f"📝 Generated text: {generated_text[0]}")
        except Exception as e:
            print(f"⚠️ 생성된 텍스트 디코딩 실패: {e}")
        
        print("✅ Real image test succeeded!")
        return True
    except Exception as e:
        print(f"❌ Real image test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 PanoVLM 테스트 시작...")
    
    # 초기화 테스트
    model, config = test_panovlm_initialization()
    
    # 사용 가능한 CUDA 확인
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available, using CPU")
    
    # Forward pass 테스트
    success_forward = test_panovlm_forward(model, use_cuda=use_cuda)
    
    # Generate 함수 테스트
    if success_forward:  # Forward가 성공한 경우만 Generate 테스트
        success_generate = test_panovlm_generate(model, use_cuda=use_cuda)
    
    # 실제 이미지 테스트 (선택적)
    # 샘플 이미지 폴더가 있으면 자동으로 테스트
    test_with_real_image(model, use_cuda=use_cuda)
    
    print("\n🏁 PanoVLM 테스트 완료!")
