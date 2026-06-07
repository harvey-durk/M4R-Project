# Hardware Platform

All experiments were conducted on the CX3 High-Performance Computing (HPC) cluster, via the [Imperial Research Computing Service](http://doi.org/10.14469/hpc/2232) . All fine-tuning, exportation and evaluation of models was carried out in a single GPU amd all other scripts were run on a single CPU.

# Environment Setup 

First, create the evaluation environment by running the following:
```
conda create --name statqa_eval python=3.11.15
conda activate statqa_eval
pip install -r requirements_statqa_eval.txt --extra-index-url https://download.pytorch.org/whl/cu121
```
Version 3.11.15 of python is used and all required libraries are detailed in `requirements_statqa_eval.txt`.

Next, create the general purpose environment by running:
```
conda create --name statqa python=3.11.14
conda activate statqa
pip install -r requirements_statqa.txt
cd LLaMA-Factory
pip install -e .[torch,metrics]
cp -r ../Finetuning/LLaMA-Factory/* data/
```
This uses python 3.11.14 and also requires one to install the LLaMA-Factory environment for fine-tuning, on top of the libraries in `requirements_statqa.txt`. Throughout, commit [f9f11dc](https://github.com/hiyouga/LlamaFactory/tree/f9f11dcb9762b75f784cec81a5d2155929f1eca9) of LLaMA-Factory will be used. 

# Data

The mini-StatQA benchmark is given in csv and JSON form in `Data/Integrated Dataset/Balanced Benchmark`, with the prompts for each prompting strategy given in `Data/Integrated Dataset/Dataset with Prompt/Test Set`. The fine-tuning datasets, $𝔻_\text{train}$ and mini-StatQA-foundational are given in `Finetuning/LLaMA-Factory` in JSON form, along with the dataset information file which must all be copied into the LLaMA-Factory environments as detailed above. mini-StatQA, its corresponding prompts and $𝔻_\text{train}$ were acquired directly from the [StatQA Repository](https://github.com/HKUSTDial/StatQA/tree/main) and mini-StatEval-foundational was obtained via running `stateval_convert_train.sh` to convert the [StatEval-Foundational-knowledge](https://huggingface.co/datasets/0v01111/StatEval-Foundational-knowledge/tree/main) JSONL file into LLaMA-Factory friendly JSON format whilst reformatting 6 answers that were given as dictionaries as opposed to strings and removing 3 duplicate examples. `stateval_convert_train.py` and  `deduplicate_json.py` are the python scripts that handle this.

# Models

The four Llama models of interest: [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf), [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) and [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) can be downloaded via hugging face. Throughout, it is assumed the base models are stored directly in `Models` with fine-tuned models being stored in their respective folders: `Models/Sweep` for $𝔻_\text{train}$ fine-tuned models; `Models/StatEval` for mini-StatQA-foundational fine-tuned models; `Models/sft_StatEval` for models fine-tuned on $𝔻_\text{train}$ followed by mini-StatQA-foundational; and `Models/StatEval_sft` for models fine-tuned on mini-StatQA-foundational followed by $𝔻_\text{train}$.

# Baseline Reproduction



# $𝔻_\text{train}$ LoRA Learning Rate Tuning Sweep



# StatEval Transfer Learning

## Fine-Tuning of Base Models on mini-StatEval-foundational



## Fine-Tuning on $𝔻_\text{train}$ followed by mini-StatEval-foundational



## Fine-Tuning on mini-StatEval-foundational followed by $𝔻_\text{train}$



# Neuro-Symbolic AI Framework



# Analysis of Answers



# Plotting for Figures
