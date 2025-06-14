#!/bin/bash
# 분산 학습을 위한 PanoVLM 학습 스크립트

# 환경 변수 설정
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 사용할 GPU 지정
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 기본 매개변수
CONFIG_PATH="config/panovlm_train.yaml"
NUM_NODES=1
NUM_GPUS_PER_NODE=4
MASTER_PORT=$(( RANDOM % 50000 + 10000 ))

# 명령행 인수 파싱
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --nodes)
      NUM_NODES="$2"
      shift 2
      ;;
    --gpus-per-node)
      NUM_GPUS_PER_NODE="$2"
      shift 2
      ;;
    *)
      echo "알 수 없는 옵션: $1"
      exit 1
      ;;
  esac
done

# DeepSpeed 설정
DEEPSPEED_CONFIG="config/zero3.json"

# 분산 학습 명령어
python -m torch.distributed.run \
  --nproc_per_node=$NUM_GPUS_PER_NODE \
  --nnodes=$NUM_NODES \
  --master_port=$MASTER_PORT \
  train_panovlm.py \
  --config $CONFIG_PATH

echo "학습 완료!"
