export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY=9fd21364ed6c1c6677a250972c5e19a931171974
export TORCH_ZIPFILE_SERIALIZATION=legacy

python train.py --config configs/stage1_vision_pretrain.yaml
python train.py --config configs/stage2_qformer_pretrain.yaml 

python eval.py --config configs/eval.yaml