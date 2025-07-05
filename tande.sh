export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974
export TORCH_ZIPFILE_SERIALIZATION=legacy

python train.py --config config/stage1_pretrain.yaml --stage 1
python train.py --config config/stage2_finetune.yaml --stage 2

python eval.py --config config/eval.yaml