#!/bin/bash
#PBS -N figure_plots
#PBS -l select=1:ncpus=1:mem=8gb
#PBS -l walltime=00:10:00
#PBS -o /rds/general/user/hd722/home/Project/Script/outputs/figure_plots.out
#PBS -e /rds/general/user/hd722/home/Project/Script/errors/figure_plots.err

# Change to Project directory
cd /rds/general/user/hd722/home/Project

# Activate conda environment
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate statqa

echo "[i] Create figures of tuning sweep plots."
sleep 5s

python d_train_figure.py
python stateval_figure.py
python d_train_stateval_figure.py

echo "=== Figures created ==="
echo "Job finished on: $(date)"
