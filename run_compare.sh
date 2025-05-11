#$ -N modegpt_llama2_7b_run
#$ -V
#$ -cwd
#$ -o logs/compare.o$JOB_ID
#$ -e logs/compare.e$JOB_ID
#$ -l h_rt=12:00:00
#$ -l h_vmem=32G
#$ -pe sharedmem 4
#$ -l A100,gpu,gpu_mem=80G,cuda=1


python compare_vo_weights.py