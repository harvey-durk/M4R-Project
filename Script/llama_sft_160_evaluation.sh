#!/bin/bash
#PBS -N llama_sft_160_evaluation
#PBS -l select=1:ncpus=1:mem=32gb:ngpus=1
#PBS -l walltime=8:00:00
#PBS -o /rds/general/user/hd722/home/Project/Script/outputs/llama_sft_160_evaluation.out
#PBS -e /rds/general/user/hd722/home/Project/Script/errors/llama_sft_160_evaluation.err

# Change to Project directory
cd /rds/general/user/hd722/home/Project

# Activate conda environment
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate statqa_eval

export VLLM_ATTENTION_BACKEND=FLASHINFER

echo "[i] Perform evaluations on finetuned Llama-2-7b model."
sleep 5s

# llama-2-7b-sft
wait
python Evaluation/llama_sft_evaluation.py --model_type "2_7b_sft_160" --dataset_name "mini-StatQA" --trick "zero-shot"

echo "=== Evaluations complete ==="
echo "Job finished on: $(date)"
