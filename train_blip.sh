export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974
export TORCH_ZIPFILE_SERIALIZATION=legacy

python train_blip.py --cfg config/train.yaml
python train_blip.py --cfg config/eval_blip.yaml