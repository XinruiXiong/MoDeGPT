#!/bin/bash
#$ -V
#$ -N modegpt_run           # 任务名称
#$ -cwd                     # 使用当前工作目录
#$ -o logs/opt-125m.o$JOB_ID
#$ -e logs/opt-125m.e$JOB_ID
#$ -l h_rt=12:00:00         # 最长运行时间（12小时）
#$ -l h_vmem=16G            # 每个核的最大内存
#$ -pe sharedmem 4          # 请求 4 个 CPU 核
#$ -l gpu=1                 # 请求 1 块 GPU（如有）


# 确保激活你的 Python 虚拟环境
# source ~/.bashrc
# conda activate modegpt  # 或你自己的虚拟环境名称

echo "Running on node: $(hostname)"
echo "Python path: $(which python)"
python --version

# run MoDeGPT
python run_modegpt.py \
  --model facebook/opt-125m \
  --compression_ratio 0.9 \
  --calib_size 8 \
  --eval_size all \
  --output_dir compressed_output