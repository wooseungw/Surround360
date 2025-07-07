# train.py

import argparse
import yaml
import os
from pprint import pprint

import torch
from transformers import (
    TrainingArguments,
    Trainer,
    Blip2Processor,
    AutoTokenizer,
)
from peft import get_peft_model, LoraConfig

# 현재 프로젝트의 모듈들을 임포트합니다.
# 파일 구조에 따라 경로를 수정해야 할 수 있습니다.
from src.models.surroundblip import SurroundBlip 
from transformers import Blip2Config
from dataset import QuIC360Dataset, data_collator

# ----------------------------------------------------------------
# 1. 설정 로딩 및 병합 유틸리티
# ----------------------------------------------------------------
def deep_merge_dict(base_dict, new_dict):
    """중첩된 딕셔너리를 재귀적으로 병합합니다."""
    for key, value in new_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = deep_merge_dict(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def load_and_merge_configs(stage_config_path: str, base_config_path: str = 'configs/base.yaml'):
    """기본 설정과 스테이지 설정을 불러와 병합합니다."""
    print(f"Loading base configuration from: {base_config_path}")
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loading stage configuration from: {stage_config_path}")
    with open(stage_config_path, 'r') as f:
        stage_config = yaml.safe_load(f)
        
    config = deep_merge_dict(config, stage_config)
    return config

# ----------------------------------------------------------------
# 2. 'stage' 인자를 모델에 전달하기 위한 커스텀 Trainer
# ----------------------------------------------------------------
class StageAwareTrainer(Trainer):
    """
    모델의 forward 메소드에 'stage' 인자를 전달하는 커스텀 Trainer.
    """
    def __init__(self, *args, stage_name: str, loss_specific_args: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_name = stage_name
        self.loss_specific_args = loss_specific_args if loss_specific_args is not None else {}

    def compute_loss(self, model, inputs, return_outputs=False):
        # inputs 딕셔너리에 stage와 loss 관련 인자를 추가하여 모델에 전달
        inputs["stage"] = self.stage_name
        inputs.update(self.loss_specific_args)
        
        outputs = model(**inputs)
        loss = outputs["loss"]
        
        return (loss, outputs) if return_outputs else loss

# ----------------------------------------------------------------
# 3. 학습 가능한 파라미터 출력 유틸리티
# ----------------------------------------------------------------
def print_trainable_parameters(model):
    """학습 가능한 파라미터의 수와 비율을 출력합니다."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )

# ----------------------------------------------------------------
# 4. 메인 학습 함수
# ----------------------------------------------------------------
def train(config: dict):
    """설정 딕셔너리를 받아 학습을 진행하는 메인 함수"""
    
    print("--- Configuration ---")
    pprint(config)
    print("-----------------------")

    # --- 모델 및 프로세서 로딩 ---
    print("Loading Processor and Model...")
    # processor는 BLIP-2의 것을 그대로 사용하거나, 필요시 커스텀
    processor = Blip2Processor.from_pretrained(config['model']['pretrain_name'])
    
    # 체크포인트 경로에 따라 모델 로딩
    if config['model']['load_from_checkpoint']:
        print(f"Loading model from checkpoint: {config['model']['load_from_checkpoint']}")
        model = SurroundBlip.from_pretrained(config['model']['load_from_checkpoint'])
    else:
        print("Initializing model from scratch based on config.")
        # 이 부분은 SurroundBlip이 Blip2Config를 받아 초기화되도록 구현되어 있어야 함
        model_config = Blip2Config.from_pretrained(config['model']['model_name_or_path'])
        model = SurroundBlip(model_config)

    # --- 모델 레이어 동결 (Freezing) ---
    if config['model'].get('freeze_modules'):
        print(f"Freezing modules: {config['model']['freeze_modules']}")
        for name, param in model.named_parameters():
            for freeze_name in config['model']['freeze_modules']:
                if freeze_name in name:
                    param.requires_grad = False
                    break

    # --- LoRA/PEFT 적용 (3단계) ---
    if config['model'].get('use_lora', False):
        print("Setting up LoRA/PEFT...")
        lora_config = config['model']['lora_config']
        peft_config = LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            target_modules=lora_config['target_modules'],
            lora_dropout=lora_config['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM", # 또는 "SEQ_2_SEQ_LM"
        )
        model = get_peft_model(model, peft_config)
    
    print_trainable_parameters(model)

    # --- 데이터셋 로딩 ---
    print("Loading Datasets...")
    train_dataset = QuIC360Dataset(
        csv_file=config['data']['train_csv_path'],
        processor=processor,
        split='train',
        max_length=config['data']['max_length'],
    )
    eval_dataset = QuIC360Dataset(
        csv_file=config['data']['valid_csv_path'],
        processor=processor,
        split='eval',
        max_length=config['data']['max_length'],
    )
    
    # --- Trainer 설정 및 실행 ---
    print("Initializing Trainer...")
    training_args = TrainingArguments(**config['training_args'])
    
    # 2. StageAwareTrainer를 초기화합니다.
    trainer = StageAwareTrainer(
        model=model,
        args=training_args, # 표준 인자 전달
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        stage_name=config['stage_name'],
        # [핵심] yaml의 'custom_args' 섹션에서 커스텀 인자를 가져옵니다.
        loss_specific_args=config.get('custom_args', {}).get('loss_specific_args', {})
    )

    print(f"--- Starting Training for Stage: {config['stage_name']} ---")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    print("Training finished. Saving final model.")
    model.save_pretrained(os.path.join(training_args.output_dir, "final_checkpoint"))


# ----------------------------------------------------------------
# 5. 스크립트 실행 블록
# ----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to the stage-specific yaml config file.")
    args = parser.parse_args()

    # 설정 로드 및 병합
    config = load_and_merge_configs(stage_config_path=args.config)
    
    # 학습 실행
    train(config)