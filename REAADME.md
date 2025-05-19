# A code implemention of MoDeGPT
This repository is a code implemention of [MoDeGPT](https://arxiv.org/abs/2408.09632) in ICLR 2025

### 1. Set-up
1. Clone this repository
```bash
git clone https://github.com/XinruiXiong/MoDeGPT.git
cd MoDeGPT
```
2. Install Package
```Shell
conda create -n modegpt python=3.11
conda activate modegpt
pip install -r requirements.txt
```

### 2. Run compression
LLaMA2 example:
```bash
  python run_modegpt.py \
  --model meta-llama/Llama-2-7b-hf \
  --compression_ratio 0.8 \
  --calib_size 32 \
  --eval_size all \
  --output_dir ./compressed_output/llama2-7b_0.8 \
  --device 0
```

OPT example:
```bash
  python run_modegpt.py \
  --model facebook/opt-6.7b \
  --compression_ratio 0.8 \
  --calib_size 32 \
  --eval_size all \
  --output_dir ./compressed_output/opt-6.7b_0.8 \
  --device 0
```