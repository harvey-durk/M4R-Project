#!/bin/bash
#PBS -N llama_sft_stateval_0_5
#PBS -l select=1:ncpus=1:mem=32gb:ngpus=1
#PBS -l walltime=10:00:00
#PBS -o /rds/general/user/hd722/home/Project/Script/outputs/llama_sft_stateval_0_5.out
#PBS -e /rds/general/user/hd722/home/Project/Script/errors/llama_sft_stateval_0_5.err

# Change to LLaMA-Factory directory
cd /rds/general/user/hd722/home/Project/LLaMA-Factory

# Activate conda environment
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate statqa

echo "[i] Finetune sft LLaMA models over StatEval using LoRA with learning rate 5.0e-6."
sleep 5s

# llama-2-7b
wait
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train ../Finetuning/sft_StatEval/llama2_7b_lora_sft_stateval_0_5.yaml
echo "-----------------------------------------------------------"
echo " [i] SFT: fine-tuning for LLaMA-2-7b is finished!"
echo "-----------------------------------------------------------"

# llama-3-8b
wait
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train ../Finetuning/sft_StatEval/llama3_8b_lora_sft_stateval_0_5.yaml
echo "-----------------------------------------------------------"
echo " [i] SFT: fine-tuning for LLaMA-3-8b is finished!"
echo "-----------------------------------------------------------"

# llama-3-8b-instruct
wait
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train ../Finetuning/sft_StatEval/llama3_8b_instruct_lora_sft_stateval_0_5.yaml
echo "-----------------------------------------------------------"
echo " [i] SFT: fine-tuning for LLaMA-3-8b-instruct is finished!"
echo "-----------------------------------------------------------"

echo "Job finished on: $(date)"
