# Surround360

## Setup Env
``` bash
conda create -n sur python=3.11 -y
conda activate sur 
```
### Install Torch

#### Linux or Window
``` 
# ROCM 6.0 (Linux only)
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/rocm6.0
# CUDA 11.8
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```
#### Mac OS 
``` 
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
```

#### Install DeepSpeed(Windows, Not recommended)
```
pip install https://github.com/daswer123/deepspeed-windows/releases/download/13.1/deepspeed-0.13.1+cu121-cp311-cp311-win_amd64.whl
```
#### Install Requirements
```
pip install -r requirements.txt
```
#### Install Surround360 (May not requirements work)
```
pip install transformers==4.51.3
pip install deepspeed
conda install -y -c conda-forge mpich
conda install -y cuda-cudart cuda-version=12
pip install mpi4py==4.0.3
conda install -y -c conda-forge cuda-nvcc=12.1
conda install -y -c conda-forge cudatoolkit-dev=12.1
pip install flash-attn --no-build-isolation
pip install git+https://github.com/jmhessel/pycocoevalcap.git@f76c47defb8eb646545147f913b7023bbfcfcabe
pip install openai-clip==1.0.1
pip install accelerate==1.6.0
pip install evaluate==0.4.3
pip install opencv-python==4.11.0.86
pip install pandas==2.2.3
```
