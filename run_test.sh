#!/bin/bash
#$ -N opt_inference
#$ -cwd
#$ -V
#$ -o logs/opt_inference.o$JOB_ID
#$ -e logs/opt_inference.e$JOB_ID
#$ -l h_rt=24:00:00
#$ -l h_vmem=32G
#$ -pe sharedmem 4
#$ -l gpu,A100,cuda=1  

python test.py