export CUDA_VISIBLE_DEVICES=1

python train.py --config config/train.yaml
python eval.py --config config/eval.yaml