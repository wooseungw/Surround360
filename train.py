import argparse
import os
import yaml
from pprint import pprint
import torch

from transformers import (
    TrainingArguments,
    Trainer,
    Blip2Processor,
    Blip2Config,
    Blip2ForConditionalGeneration
)
from peft import get_peft_model, LoraConfig

from src.models.surroundblip import SurroundBlip
from src.dataset import QuIC360Dataset, data_collator

# ----------------------------------------------------------------
# 1. 설정 로딩 및 병합 유틸리티
# ----------------------------------------------------------------
def deep_merge_dict(base_dict, new_dict):
    for key, value in new_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = deep_merge_dict(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def load_and_merge_configs(stage_config_path: str, base_config_path: str = 'configs/base.yaml'):
    print(f"Loading base configuration from: {base_config_path}")
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loading stage configuration from: {stage_config_path}")
    with open(stage_config_path, 'r') as f:
        stage_config = yaml.safe_load(f)
    config = deep_merge_dict(config, stage_config)
    return config

# ----------------------------------------------------------------
# 2. 커스텀 Trainer
# ----------------------------------------------------------------
class StageAwareTrainer(Trainer):
    def __init__(self, *args, stage_name: str, custom_args: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_name = stage_name
        self.custom_args = custom_args if custom_args is not None else {}
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs["stage"] = self.stage_name
        inputs.update(self.custom_args.get('loss_specific_args', {}))
        
        outputs = model(**inputs)
        loss = outputs.get("loss")
        
        return (loss, outputs) if return_outputs else loss

# ----------------------------------------------------------------
# 3. 학습 가능한 파라미터 출력 유틸리티
# ----------------------------------------------------------------
def print_trainable_parameters(model):
    trainable_params, all_param = 0, 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || "
          f"trainable%: {100 * trainable_params / all_param:.2f}")

# ----------------------------------------------------------------
# 4. 메인 학습 함수
# ----------------------------------------------------------------
def train(config: dict):
    print("--- Configuration ---")
    pprint(config)
    print("-----------------------")

    # --- TF32 비활성화 (CUDA 호환성 문제 방지) ---
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # --- 프로세서 로딩 ---
    processor = Blip2Processor.from_pretrained(config['model']['model_name_or_path'])
    
    # --- 사전 학습된 모듈 이식 및 가중치 공유 ---
    print("Loading official BLIP-2 model to extract pre-trained components...")
    official_blip2_model = Blip2ForConditionalGeneration.from_pretrained(config['model']['model_name_or_path'])

    print("Initializing custom SurroundBlip model structure...")
    model_config = Blip2Config.from_pretrained(config['model']['model_name_or_path'])
    model = SurroundBlip(model_config)

    print("Transplanting pre-trained weights to custom model...")
    model.vision_model = official_blip2_model.vision_model
    model.qformer = official_blip2_model.qformer
    model.language_projection = official_blip2_model.language_projection
    
    print("Sharing weights from Q-Former to Text Encoder...")
    model.text_encoder.load_state_dict(model.qformer.state_dict(), strict=False)
    
    del official_blip2_model
    torch.cuda.empty_cache()

    # --- 스테이지별 체크포인트 로드 (이식 후) ---
    if config['model']['load_from_checkpoint']:
        print(f"Loading stage-specific checkpoint from: {config['model']['load_from_checkpoint']}")
        checkpoint_path = os.path.join(config['model']['load_from_checkpoint'], "pytorch_model.bin")
        if os.path.exists(checkpoint_path):
            stage_checkpoint = torch.load(checkpoint_path, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(stage_checkpoint, strict=False)
            print(f"Loaded checkpoint with missing keys: {missing_keys}")
            print(f"Loaded checkpoint with unexpected keys: {unexpected_keys}")
        else:
            print(f"Warning: Checkpoint path not found: {checkpoint_path}")

    # --- 모든 관련 임베딩 레이어 리사이즈 ---
    print("Synchronizing tokenizer and all model vocab sizes...")
    tokenizer_vocab_size = len(processor.tokenizer)

    if model.text_encoder.config.vocab_size != tokenizer_vocab_size:
        print(f"Resizing text_encoder's embeddings: {model.text_encoder.config.vocab_size} -> {tokenizer_vocab_size}")
        model.text_encoder.resize_token_embeddings(tokenizer_vocab_size)
    
    if model.get_input_embeddings().weight.shape[0] != tokenizer_vocab_size:
        print(f"Resizing language_model's embeddings: {model.get_input_embeddings().weight.shape[0]} -> {tokenizer_vocab_size}")
        model.resize_token_embeddings(tokenizer_vocab_size)
        
    print("All vocab sizes are synchronized.")
    
    # --- 모델 레이어 동결 ---
    freeze_modules = config['model'].get('freeze_modules', [])
    if freeze_modules:
        print(f"Freezing modules: {freeze_modules}")
    for name, param in model.named_parameters():
        is_trainable = True
        for freeze_name in freeze_modules:
            if freeze_name in name:
                is_trainable = False
                break
        param.requires_grad = is_trainable
    
    # --- LoRA/PEFT 적용 ---
    if config['model'].get('use_lora', False):
        print("Setting up LoRA/PEFT...")
        lora_config_dict = config['model']['lora_config']
        peft_config = LoraConfig(**lora_config_dict)
        model = get_peft_model(model, peft_config)

    print_trainable_parameters(model)

    # --- 데이터셋 로딩 ---
    train_dataset = QuIC360Dataset(config['data']['train_csv_path'], processor, max_length=config['data']['max_length'])
    eval_dataset = QuIC360Dataset(config['data']['valid_csv_path'], processor, max_length=config['data']['max_length'])

    # --- Trainer 설정 및 실행 ---
    print("Initializing Trainer...")
    training_args_dict = config.get('training_args', {})
    custom_args_dict = config.get('custom_args', {})
    training_args = TrainingArguments(**training_args_dict)
    
    trainer = StageAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        stage_name=config.get('stage_name', 'finetune'),
        custom_args=custom_args_dict
    )

    print(f"--- Starting Training for Stage: {trainer.stage_name} ---")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    # --- 최종 모델 및 프로세서 저장 ---
    print("Training finished. Saving final model and processor.")
    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    processor.save_pretrained(output_dir)
    trainer.save_model(output_dir)
    print(f"Model and processor saved to {output_dir}")

# ----------------------------------------------------------------
# 5. 스크립트 실행 블록
# ----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to the stage-specific yaml config file.")
    args = parser.parse_args()
    config = load_and_merge_configs(stage_config_path=args.config)
    train(config)