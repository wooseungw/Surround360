# Surround360

## Setup Env
``` bash
conda create -n sur python=3.12 -y
conda activate sur
pip install 
```
### Install Torch
#### Mac OS 
``` 
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
```
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
#### Install HuggingFace
```
pip install transformers==4.51.3
pip install datasets==3.5.1
```
