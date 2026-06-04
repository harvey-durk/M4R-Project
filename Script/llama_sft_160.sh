#!/bin/bash
#PBS -N llama_sft_160
#PBS -l select=1:ncpus=1:mem=48gb:ngpus=1
#PBS -l walltime=6:00:00
#PBS -o /rds/general/user/hd722/home/Project/Script/outputs/llama_sft_160.out
#PBS -e /rds/general/user/hd722/home/Project/Script/errors/llama_sft_160.err

# Change to LLaMA-Factory directory
cd /rds/general/user/hd722/home/Project/LLaMA-Factory

# Activate conda environment
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate statqa

echo "[i] Finetune LLaMA-2-7b model using LoRA with learning rate 1.6e-3."
sleep 5s

# llama-2-7b
wait
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train ../Finetuning/Sweep/llama2_7b_lora_sft_160.yaml
echo "-----------------------------------------------------------"
echo " [i] SFT: fine-tuning for LLaMA-2-7b is finished!"
echo "-----------------------------------------------------------"

echo "Job finished on: $(date)"
