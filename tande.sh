export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974

python train.py --config config/train.yaml
python eval.py --config config/eval.yaml