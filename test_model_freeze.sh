#!/bin/bash
# PanoVLM 모델 파라미터 동결 설정 테스트

# 환경 변수 설정
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 실행 설정
CONFIG_PATH="config/panovlm_train.yaml"
DEBUG_MODE=true  # 디버그 모드 (실제 학습 실행 안 함)

# 디버그 모드일 때 실행 옵션
if [ "$DEBUG_MODE" = true ]; then
  # 간단한 모델 초기화만 수행하여 파라미터 동결 상태 확인
  python -c "
import yaml
import torch
from pathlib import Path
from src.models.panovlm import PanoVLM
from src.models.panovlm_config import PanoVLMConfig, ProjectorConfig
from transformers import AutoTokenizer

# 설정 로드
with open('$CONFIG_PATH', 'r') as f:
    config = yaml.safe_load(f)

print('설정 파일을 로드했습니다:', config)

# 모델 구성
print('\n모델을 초기화합니다...')
pano_config = PanoVLMConfig(
    vision_model_name_or_path=config['vision_model_name_or_path'],
    language_model_name_or_path=config['language_model_name_or_path'],
)

# Projector 설정
if 'projector_type' in config:
    print(f\"프로젝터 타입: {config['projector_type']}\")
    pano_config.projector_config = ProjectorConfig(
        type=config['projector_type'],
        in_features=config.get('projector_dim_in', 768),
        out_features=config.get('projector_dim_out', 4096)
    )

# 모델 초기화
model = PanoVLM(pano_config)

# 동결 설정
def freeze_model_part(model_part, freeze=True, except_layers=None):
    if not freeze:
        return
    
    for name, param in model_part.named_parameters():
        param.requires_grad = False
        
    if except_layers:
        for name, param in model_part.named_parameters():
            if any(layer in name for layer in except_layers):
                param.requires_grad = True

# Vision 모델 동결
vision_freeze = config.get('freeze_vision', False)
vision_except = config.get('freeze_vision_except', [])
freeze_model_part(model.vision_model, vision_freeze, vision_except)

# 언어 모델 동결
lang_freeze = config.get('freeze_language', True)
lang_except = config.get('freeze_language_except', [])
freeze_model_part(model.language_model, lang_freeze, lang_except)

# Projector 동결
if model.projector is not None and config.get('freeze_projector', False):
    for param in model.projector.parameters():
        param.requires_grad = False

# 각 모델 부분별 파라미터 요약
def count_parameters(model_part):
    total = sum(p.numel() for p in model_part.parameters())
    trainable = sum(p.numel() for p in model_part.parameters() if p.requires_grad)
    return total, trainable

# 전체 모델 파라미터
total_params, trainable_params = count_parameters(model)
print(f'\n===== 전체 모델 파라미터 요약 =====')
print(f'총 파라미터 수: {total_params:,}')
print(f'학습 가능 파라미터 수: {trainable_params:,}')
print(f'학습 불가 파라미터 수: {total_params - trainable_params:,}')
print(f'학습 가능 비율: {trainable_params/total_params*100:.2f}%')

# Vision Encoder 파라미터
vision_total, vision_trainable = count_parameters(model.vision_model)
print(f'\n===== Vision Encoder 파라미터 =====')
print(f'총 파라미터 수: {vision_total:,}')
print(f'학습 가능 파라미터 수: {vision_trainable:,}')
print(f'학습 가능 비율: {vision_trainable/vision_total*100:.2f}%')

# Language Model 파라미터
lang_total, lang_trainable = count_parameters(model.language_model)
print(f'\n===== Language Model 파라미터 =====')
print(f'총 파라미터 수: {lang_total:,}')
print(f'학습 가능 파라미터 수: {lang_trainable:,}')
print(f'학습 가능 비율: {lang_trainable/lang_total*100:.2f}%')

# Projector 파라미터
if model.projector is not None:
    proj_total, proj_trainable = count_parameters(model.projector)
    print(f'\n===== Projector 파라미터 =====')
    print(f'총 파라미터 수: {proj_total:,}')
    print(f'학습 가능 파라미터 수: {proj_trainable:,}')
    print(f'학습 가능 비율: {proj_trainable/proj_total*100:.2f}%')

# Vision 모델 각 레이어별 학습 가능 여부 확인
print('\n===== Vision 모델 레이어별 학습 가능 상태 =====')
for name, param in model.vision_model.named_parameters():
    if param.requires_grad:
        print(f'학습 가능: {name}')
"
else
  # 실제 학습 실행
  python train_panovlm.py --config "$CONFIG_PATH"
fi
