#!/bin/bash
# PanoVLM 모델 학습을 위한 스크립트

# 환경 변수 설정
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 기본 학습 설정
CONFIG_PATH="config/panovlm_train.yaml"

# 학습 시작
echo "PanoVLM 모델 학습 시작..."
python train_panovlm.py --config $CONFIG_PATH

echo "학습 완료!"
