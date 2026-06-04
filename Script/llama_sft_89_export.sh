#!/bin/bash
#PBS -N llama_sft_89_export
#PBS -l select=1:ncpus=1:mem=16gb:ngpus=1
#PBS -l walltime=00:40:00
#PBS -o /rds/general/user/hd722/home/Project/Script/outputs/llama_sft_89_export.out
#PBS -e /rds/general/user/hd722/home/Project/Script/errors/llama_sft_89_export.err

# Change to LLaMA-Factory directory
cd /rds/general/user/hd722/home/Project/LLaMA-Factory

# Activate conda environment
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate statqa

echo "[i] Export finetuned LLaMA models to be used in evaluation."
sleep 5s

# llama-2-7b
wait
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export ../Finetuning/Sweep/llama2_7b_lora_sft_89_export.yaml
echo "-----------------------------------------------------------"
echo " [i] SFT: export of finetuned LLaMA-2-7b is finished!"
echo "-----------------------------------------------------------"

# llama-3-8b
wait
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export ../Finetuning/Sweep/llama3_8b_lora_sft_89_export.yaml
echo "-----------------------------------------------------------"
echo " [i] SFT: export of finetuned LLaMA-3-8b is finished!"
echo "-----------------------------------------------------------"

# llama-3-8b-instruct
wait
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export ../Finetuning/Sweep/llama3_8b_instruct_lora_sft_89_export.yaml
echo "-----------------------------------------------------------"
echo " [i] SFT: export of finetuned LLaMA-3-8b-instruct is finished!"
echo "-----------------------------------------------------------"

echo "Job finished on: $(date)"
