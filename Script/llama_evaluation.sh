#!/bin/bash
#PBS -N llama_evaluation
#PBS -l select=1:ncpus=1:mem=24gb:ngpus=1
#PBS -l walltime=10:00:00
#PBS -o /rds/general/user/hd722/home/Project/Script/outputs/llama_evaluation.out
#PBS -e /rds/general/user/hd722/home/Project/Script/errors/llama_evaluation.err

# Change to Project directory
cd /rds/general/user/hd722/home/Project

# Activate conda environment
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate statqa_eval

# Force CUDA_VISIBLE_DEVICES to a simple integer index
export CUDA_VISIBLE_DEVICES=0

echo "[i] Perform evaluations on all Llama models."
sleep 5s

# llama-2-7b
wait
python Evaluation/llama_evaluation.py --model_type "2_7b" --dataset_name "mini-StatQA" --trick "zero-shot"

wait
python Evaluation/llama_evaluation.py --model_type "2_7b" --dataset_name "mini-StatQA" --trick "one-shot"

wait
python Evaluation/llama_evaluation.py --model_type "2_7b" --dataset_name "mini-StatQA" --trick "zero-shot-CoT"

wait
python Evaluation/llama_evaluation.py --model_type "2_7b" --dataset_name "mini-StatQA" --trick "one-shot-CoT"

wait
python Evaluation/llama_evaluation.py --model_type "2_7b" --dataset_name "mini-StatQA" --trick "stats-prompt"


# llama-2-13b
wait
python Evaluation/llama_evaluation.py --model_type "2_13b" --dataset_name "mini-StatQA" --trick "zero-shot"

wait
python Evaluation/llama_evaluation.py --model_type "2_13b" --dataset_name "mini-StatQA" --trick "one-shot"

wait
python Evaluation/llama_evaluation.py --model_type "2_13b" --dataset_name "mini-StatQA" --trick "zero-shot-CoT"

wait
python Evaluation/llama_evaluation.py --model_type "2_13b" --dataset_name "mini-StatQA" --trick "one-shot-CoT"

wait
python Evaluation/llama_evaluation.py --model_type "2_13b" --dataset_name "mini-StatQA" --trick "stats-prompt"


# llama-3-8b-Instruct
wait
python Evaluation/llama_evaluation.py --model_type "3_8b_instruct" --dataset_name "mini-StatQA" --trick "zero-shot"

wait
python Evaluation/llama_evaluation.py --model_type "3_8b_instruct" --dataset_name "mini-StatQA" --trick "one-shot"

wait
python Evaluation/llama_evaluation.py --model_type "3_8b_instruct" --dataset_name "mini-StatQA" --trick "zero-shot-CoT"

wait
python Evaluation/llama_evaluation.py --model_type "3_8b_instruct" --dataset_name "mini-StatQA" --trick "one-shot-CoT"

wait
python Evaluation/llama_evaluation.py --model_type "3_8b_instruct" --dataset_name "mini-StatQA" --trick "stats-prompt"


# llama-3-8b
wait
python Evaluation/llama_evaluation.py --model_type "3_8b" --dataset_name "mini-StatQA" --trick "zero-shot"

wait
python Evaluation/llama_evaluation.py --model_type "3_8b" --dataset_name "mini-StatQA" --trick "one-shot"

wait
python Evaluation/llama_evaluation.py --model_type "3_8b" --dataset_name "mini-StatQA" --trick "zero-shot-CoT"

wait
python Evaluation/llama_evaluation.py --model_type "3_8b" --dataset_name "mini-StatQA" --trick "one-shot-CoT"

wait
python Evaluation/llama_evaluation.py --model_type "3_8b" --dataset_name "mini-StatQA" --trick "stats-prompt"

echo "=== Evaluations complete ==="

echo "Job finished on: $(date)"
