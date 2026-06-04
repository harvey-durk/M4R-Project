#!/bin/bash
#PBS -N answer_analysis
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=00:30:00
#PBS -o /rds/general/user/hd722/home/Project/Script/outputs/answer_analysis.out
#PBS -e /rds/general/user/hd722/home/Project/Script/errors/answer_analysis.err

# Change to Project directory
cd /rds/general/user/hd722/home/Project

# Activate conda environment
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate statqa

echo "[i] Analyze model answers."
sleep 5s

python analyze_model_answer.py
python 'Model Answer'/'Task Performance'/summary_performance.py
python 'Model Answer'/'Task Performance'/error_type_analysis.py

echo "=== Analysis complete ==="
echo "Job finished on: $(date)"
