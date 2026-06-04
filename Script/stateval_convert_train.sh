#!/bin/bash
#PBS -N stateval_convert_train
#PBS -l select=1:ncpus=1:mem=16gb
#PBS -l walltime=00:30:00
#PBS -o /rds/general/user/hd722/home/Project/Script/outputs/stateval_convert_train.out
#PBS -e /rds/general/user/hd722/home/Project/Script/errors/stateval_convert_train.err

# Change to Project directory
cd /rds/general/user/hd722/home/Project

# Activate conda environment
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate statqa

echo "[i] Convert Foundational-knowledge.jsonl to llama-factory friendly .json."
sleep 5s

python stateval_convert_train.py --input "Foundational-knowledge.jsonl" --output "Finetuning/LLaMA-Factory/mini-stateval-foundational.json" --dataset-name "mini-stateval-foundational"
python deduplicate_json.py --input "Finetuning/LLaMA-Factory/mini-stateval-foundational.json"

echo "=== StatEval dataset convert completed ==="
echo "Job finished on: $(date)"
