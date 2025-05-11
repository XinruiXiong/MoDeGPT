#$ -N modegpt_67b_run
#$ -V
#$ -cwd
#$ -o logs/opt-6.7b/opt-6.7b.o$JOB_ID
#$ -e logs/opt-6.7b/opt-6.7b.e$JOB_ID
#$ -l h_rt=12:00:00
#$ -l h_vmem=32G
#$ -pe sharedmem 4
#$ -l A100,gpu,gpu_mem=80G,cuda=1

# python run_modegpt.py \
#   --model facebook/opt-6.7b \
#   --compression_ratio 0.5 \
#   --calib_size 8 \
#   --eval_size 128 \
#   --output_dir compressed_output/opt-6.7b

  python run_modegpt.py \
  --model facebook/opt-6.7b \
  --compression_ratio 0.5 \
  --calib_size 32 \
  --eval_size all \
  --output_dir /u/scratch/x/xxiong/compressed_output/opt-6.7b \
  --device 1