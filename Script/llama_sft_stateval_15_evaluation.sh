#!/bin/bash
#PBS -N llama_sft_stateval_15_evaluation
#PBS -l select=1:ncpus=1:mem=32gb:ngpus=1
#PBS -l walltime=5:00:00
#PBS -o /rds/general/user/hd722/home/Project/Script/outputs/llama_sft_stateval_15_evaluation.out
#PBS -e /rds/general/user/hd722/home/Project/Script/errors/llama_sft_stateval_15_evaluation.err

# Change to Project directory
cd /rds/general/user/hd722/home/Project

# Activate conda environment
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate statqa_eval

export VLLM_ATTENTION_BACKEND=FLASHINFER

echo "[i] Perform evaluations on fine-tuned Llama models."
sleep 5s

# llama-2-7b-sft
wait
python Evaluation/llama_sft_stateval_evaluation.py --model_type "2_7b_sft_stateval_15" --dataset_name "mini-StatQA" --trick "zero-shot"

# llama-3-8b-sft
wait
python Evaluation/llama_sft_stateval_evaluation.py --model_type "3_8b_sft_stateval_15" --dataset_name "mini-StatQA" --trick "zero-shot"

# llama-3-8b-Instruct-sft
wait
python Evaluation/llama_sft_stateval_evaluation.py --model_type "3_8b_instruct_sft_stateval_15" --dataset_name "mini-StatQA" --trick "zero-shot"

echo "=== Evaluations complete ==="
echo "Job finished on: $(date)"
