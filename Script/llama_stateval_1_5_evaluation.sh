#!/bin/bash
#PBS -N llama_stateval_1_5_evaluation
#PBS -l select=1:ncpus=1:mem=32gb:ngpus=1
#PBS -l walltime=10:00:00
#PBS -o /rds/general/user/hd722/home/Project/Script/outputs/llama_stateval_1_5_evaluation.out
#PBS -e /rds/general/user/hd722/home/Project/Script/errors/llama_stateval_1_5_evaluation.err

# Change to Project directory
cd /rds/general/user/hd722/home/Project

# Activate conda environment
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate statqa_eval

export VLLM_ATTENTION_BACKEND=FLASHINFER

echo "[i] Perform evaluations on StatEval fine-tuned Llama models."
sleep 5s

# llama-2-7b-sft
wait
python Evaluation/llama_stateval_evaluation.py --model_type "2_7b_stateval_1_5" --dataset_name "mini-StatQA" --trick "zero-shot"

wait
python Evaluation/llama_stateval_evaluation.py --model_type "2_7b_stateval_1_5" --dataset_name "mini-StatQA" --trick "one-shot"

wait
python Evaluation/llama_stateval_evaluation.py --model_type "2_7b_stateval_1_5" --dataset_name "mini-StatQA" --trick "zero-shot-CoT"

wait
python Evaluation/llama_stateval_evaluation.py --model_type "2_7b_stateval_1_5" --dataset_name "mini-StatQA" --trick "one-shot-CoT"

wait
python Evaluation/llama_stateval_evaluation.py --model_type "2_7b_stateval_1_5" --dataset_name "mini-StatQA" --trick "stats-prompt"

# llama-3-8b-sft
wait
python Evaluation/llama_stateval_evaluation.py --model_type "3_8b_stateval_1_5" --dataset_name "mini-StatQA" --trick "zero-shot"

wait
python Evaluation/llama_stateval_evaluation.py --model_type "3_8b_stateval_1_5" --dataset_name "mini-StatQA" --trick "one-shot"

wait
python Evaluation/llama_stateval_evaluation.py --model_type "3_8b_stateval_1_5" --dataset_name "mini-StatQA" --trick "zero-shot-CoT"

wait
python Evaluation/llama_stateval_evaluation.py --model_type "3_8b_stateval_1_5" --dataset_name "mini-StatQA" --trick "one-shot-CoT"

wait
python Evaluation/llama_stateval_evaluation.py --model_type "3_8b_stateval_1_5" --dataset_name "mini-StatQA" --trick "stats-prompt"

# llama-3-8b-Instruct-sft
wait
python Evaluation/llama_stateval_evaluation.py --model_type "3_8b_instruct_stateval_1_5" --dataset_name "mini-StatQA" --trick "zero-shot"

wait
python Evaluation/llama_stateval_evaluation.py --model_type "3_8b_instruct_stateval_1_5" --dataset_name "mini-StatQA" --trick "one-shot"

wait
python Evaluation/llama_stateval_evaluation.py --model_type "3_8b_instruct_stateval_1_5" --dataset_name "mini-StatQA" --trick "zero-shot-CoT"

wait
python Evaluation/llama_stateval_evaluation.py --model_type "3_8b_instruct_stateval_1_5" --dataset_name "mini-StatQA" --trick "one-shot-CoT"

wait
python Evaluation/llama_stateval_evaluation.py --model_type "3_8b_instruct_stateval_1_5" --dataset_name "mini-StatQA" --trick "stats-prompt"

echo "=== Evaluations complete ==="
echo "Job finished on: $(date)"
