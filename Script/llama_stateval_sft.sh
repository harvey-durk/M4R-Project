#!/bin/bash
#PBS -N llama_stateval_sft
#PBS -l select=1:ncpus=1:mem=48gb:ngpus=1
#PBS -l walltime=12:00:00
#PBS -o /rds/general/user/hd722/home/Project/Script/outputs/llama_stateval_sft.out
#PBS -e /rds/general/user/hd722/home/Project/Script/errors/llama_stateval_sft.err

# Change to LLaMA-Factory directory
cd /rds/general/user/hd722/home/Project/LLaMA-Factory

# Activate conda environment
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate statqa

echo "[i] Finetune StatEval fine-tuned LLaMA models over D-train."
sleep 5s

# llama-2-7b
wait
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train ../Finetuning/StatEval_sft/llama2_7b_lora_stateval_sft_89.yaml
echo "-----------------------------------------------------------"
echo " [i] SFT: fine-tuning for LLaMA-2-7b is finished!"
echo "-----------------------------------------------------------"

# llama-3-8b
wait
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train ../Finetuning/StatEval_sft/llama3_8b_lora_stateval_sft_28.yaml
echo "-----------------------------------------------------------"
echo " [i] SFT: fine-tuning for LLaMA-3-8b is finished!"
echo "-----------------------------------------------------------"

# llama-3-8b-instruct
wait
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train ../Finetuning/StatEval_sft/llama3_8b_instruct_lora_stateval_sft_28.yaml
echo "-----------------------------------------------------------"
echo " [i] SFT: fine-tuning for LLaMA-3-8b-instruct is finished!"
echo "-----------------------------------------------------------"

echo "Job finished on: $(date)"
