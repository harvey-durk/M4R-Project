# Hardware Platform

All experiments were conducted on the CX3 High-Performance Computing (HPC) cluster, via the [Imperial Research Computing Service](http://doi.org/10.14469/hpc/2232) . All fine-tuning, exportation and evaluation of models was carried out in a single GPU amd all other scripts were run on a single CPU.

# Environment Setup 

First, the evaluation environment needs to be created by running the following:
```
conda create --name statqa_eval python=3.11.15
conda activate statqa_eval
pip install -r requirements_statqa_eval.txt --extra-index-url https://download.pytorch.org/whl/cu121
```
Version 3.11.15 of python is used and all required libraries are detailed in `requirements_statqa_eval.txt`.

Next, the general purpose environment needs to be created by running:
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

The mini-StatQA benchmark [1] is given in csv and JSON form in `Data/Integrated Dataset/Balanced Benchmark`, with the prompts for each prompting strategy given in `Data/Integrated Dataset/Dataset with Prompt/Test Set`. The fine-tuning datasets, $𝔻_\text{train}$ and mini-StatQA-foundational are given in `Finetuning/LLaMA-Factory` in JSON form, along with the dataset information file which must all be copied into the LLaMA-Factory environments as detailed above. mini-StatQA, its corresponding prompts and $𝔻_\text{train}$ were acquired directly from the [StatQA Repository](https://github.com/HKUSTDial/StatQA/tree/main) and mini-StatEval-foundational was obtained via running `stateval_convert_train.sh` to convert the [StatEval-Foundational-knowledge](https://huggingface.co/datasets/0v01111/StatEval-Foundational-knowledge/tree/main) [2] JSONL file into LLaMA-Factory friendly JSON format whilst reformatting 6 answers that were given as dictionaries as opposed to strings and removing 3 duplicate examples. `stateval_convert_train.py` and  `deduplicate_json.py` are the python scripts that handle this.

# Models

The four Llama models of interest: [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf), [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) and [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) can be downloaded via hugging face. Throughout, it is assumed the base models are stored directly in `Models` with fine-tuned models being stored in their respective folders: `Models/Sweep` for $𝔻_\text{train}$ fine-tuned models; `Models/StatEval` for mini-StatQA-foundational fine-tuned models; `Models/sft_StatEval` for models fine-tuned on $𝔻_\text{train}$ followed by mini-StatQA-foundational; and `Models/StatEval_sft` for models fine-tuned on mini-StatQA-foundational followed by $𝔻_\text{train}$.

# Baseline Reproduction

To establish a baseline for comparison, the experimental results of Zhu et al. [1] were reproduced using the four base Llama models and the $\mathbb{D}_{\text{train}}$-fine-tuned variants.

## Base Models

Each base model is evaluated on the mini-StatQA benchmark using the five prompting strategies: 0-shot, 0-shot-CoT, 1-shot, 1-shot-CoT, and 1-shot with domain knowledge. Evaluation is performed using the `llama_evaluation.sh` script, which calls the `Evaluation/llama_evaluation.py` script as in the following example:

```
python Evaluation/llama_evaluation.py --model_type "2_7b" --dataset_name "mini-StatQA" --trick "zero-shot"
```

These bash and python scripts are obtained directly from the [StatQA Repository](https://github.com/HKUSTDial/StatQA/tree/main).

## $𝔻_\text{train}$ Fine-tuned Models

The Llama-2-7b-chat-hf, Meta-Llama-3-8B and Meta-Llama-3-8B-Instruct models are then fine-tuned using the original hyperparameters used by Zhu et al. [1] (apart from 0.0 validation split used for the Llama 3 models as opposed to the original 0.1) using the `llama_sft_5_0.sh` which calls `Finetuning/Sweep/llama2_7b_lora_sft_5_0.yaml`, `Finetuning/Sweep/llama2_7b_lora_sft_5_0.yaml` and `Finetuning/Sweep/llama2_7b_lora_sft_5_0.yaml` using the LLaMA-Factory training interface:

```
cd LLaMA-Factory
llamafactory-cli train ../Finetuning/Sweep/llama2_7b_lora_sft_5_0.yaml
```

Rather than inference directly using LLaMA-Factory, the three models are exported using LLaMA-Factory using the `llama_sft_5_0_export.sh` via:

```
llamafactory-cli export ../Finetuning/Sweep/llama2_7b_lora_sft_5_0_export.yaml
```

Finally, the exported models can then evaluated using 0-shot with the `Evaluation/llama_sft_evaluation.py` script which follows the same structure as the original evaluation script with the fine-tuned model paths instead. This is used in `llama_sft_5_0_evaluation.sh` via:

```
python Evaluation/llama_sft_evaluation.py --model_type "2_7b_sft_5_0" --dataset_name "mini-StatQA" --trick "zero-shot"
```

# $𝔻_\text{train}$ LoRA Learning Rate Tuning Sweep

The same three-step process of fine-tuning, exportation and evaluation of models is carried our with all hyperparameters constant apart from learning rate for which a sweep is carried out over the values {1.6e-5, 2.8e-5, 5.0e-5, 8.9e-4, 1.6e-4, 2.8e-4, 5.0e-4, 8.9e-4} for all three base models used in fine-tuning with Llama-2-7b-chat-hf also being fine-tuned with learning rates of {1.6e-3, 2.8e-3}. The fine-tuning and exporting files for this sweep are also found in `Finetuning/Sweep/`, and evaluation using 0-shot prompting again uses the `Evaluation/llama_sft_evaluation.py` script.

# StatEval Transfer Learning

## Fine-Tuning of Base Models on mini-StatEval-foundational

Finetuning, exportation and evaluation of models over mini-StatEval-foundational follows a similar structure using the fine-tuning and exporting files in `Finetuning/StatEval/`. These files are adapted from those used in fine-tuning over $𝔻_\text{train}$, with all hyperparameters the same other than a 0.0 validation split used across all three models, 5 epochs ran increased from 3, and a learning rate sweep over {5.0e-6, 1.5e-5, 5.0e-5, 1.5e-4}. Of course the training dataset differs too with mini-StatEval-foundational being used here. Evaluation of these models using all five prompting strategies uses the `Evaluation/llama_stateval_evaluation.py` script via:

```
python Evaluation/llama_stateval_evaluation.py --model_type "2_7b_stateval_0_5" --dataset_name "mini-StatQA" --trick "zero-shot"
python Evaluation/llama_stateval_evaluation.py --model_type "2_7b_stateval_0_5" --dataset_name "mini-StatQA" --trick "one-shot"
python Evaluation/llama_stateval_evaluation.py --model_type "2_7b_stateval_0_5" --dataset_name "mini-StatQA" --trick "zero-shot-CoT"
python Evaluation/llama_stateval_evaluation.py --model_type "2_7b_stateval_0_5" --dataset_name "mini-StatQA" --trick "one-shot-CoT"
python Evaluation/llama_stateval_evaluation.py --model_type "2_7b_stateval_0_5" --dataset_name "mini-StatQA" --trick "stats-prompt"
```

## Fine-Tuning on $𝔻_\text{train}$ followed by mini-StatEval-foundational

The same hyperparameters and learning rate sweep are used as in the above section on fine-tuning of base models on mini-StatEval-foundational. Instead the best performing learning models from the $𝔻_\text{train}$ learning rate sweep are used as the models to fine-tune. This corresponds to the Llama-2-7b-chat-hf model fine-tuned over $𝔻_\text{train}$ using a learning rate of 8.9e-4, and the Meta-Llama-3-8b and Meta-Llama-3-8b-Instruct models fine-tuned using 2.8e-4. Once fine-tuned over mini-StatEval-foundational and exported using the scripts in `Finetuning/sft_StatEval/`, the models are evaluated using 0-shot prompting with `Evaluation/llama_sft_stateval_evaluation.py` via:

```
Evaluation/llama_sft_stateval_evaluation.py --model_type "2_7b_sft_stateval_0_5" --dataset_name "mini-StatQA" --trick "zero-shot"
```

## Fine-Tuning on mini-StatEval-foundational followed by $𝔻_\text{train}$

The learning rates corresponding to best performance using 0-shot for $𝔻_\text{train}$-only fine-tuned models (again, these are 8.9e-4 for Llama-2-7b-chat-hf and 2.8e-4 for Meta-Llama-3-8b and Meta-Llama-3-8b-Instruct) and mini-StatEval-foundational-only fine-tuned models (1.5e-5 for Llama-2-7b-chat-hf and Meta-Llama-3-8b, and 1.5e-4 for Meta-Llama-3-8b-Instruct) are used for fine-tuning of the base models over mini-StatEval-foundational followed by $𝔻_\text{train}. In practice, the best performing mini-StatEval-foundational fine-tuned models are further fine-tuned on $𝔻_\text{train}$ using the fine-tuning and export files in `Finetuning/StatEval_sft/`. The remaining hyperparameters remain the same as what is used in all other $𝔻_\text{train}$ fine-tuning and evaluation over mini-StatQA using 0-shot is carried out using `Evaluation/llama_stateval_sft_evaluation.py` via:

```
python Evaluation/llama_stateval_sft_evaluation.py --model_type "2_7b_stateval_sft" --dataset_name "mini-StatQA" --trick "zero-shot"
```

# Neuro-Symbolic AI Framework

The neuro-symbolic AI framework is implemented solely via the `Evaluation/llama_sft_logic_evaluation.py` script. The first section defines the functions needed for parsing of the prompt, parsing of the LLM's output, selecting task category from the model's chosen methods and applying the rule-based system for selecting applicable methods. These functions are then used in the second section which implements the neuro-symbolic framwork, by applying the rule-based system to the responses from the LLM, which are generated in the same way as in the original evaluation script: `Evaluation/llama_evaluation.py`. `Evaluation/llama_sft_logic_evaluation.py` is then used to generate responses of the neuro-symbolic framework used in conjunction with the best performing $𝔻_\text{train}-only fine-tuned models using 0-shot via:

```
python Evaluation/llama_sft_logic_evaluation.py --model_type "2_7b_sft_logic"
```

# Analysis of Answers

The `answer_analysis.sh` script is used to process the answers generated from each of the above models, stored in `Model Answer/Origin Answer/`. The `analyze_model_answer.py` and `Model Answer/Task Performance/summary_performance.py` scripts it uses, adapted from the [StatQA Repository](https://github.com/HKUSTDial/StatQA/tree/main), output the performance based on columns selection, methods selection and overall in their respective folders in `Model Answer/Task Performance/`. The key metric of interest is the overall accuracy, which is given for each model in `Model Answer/Task Performance/Selection Overall/overall_selection_summary_performance.csv` and also in `Results.xlsx` in an easier-to-read format. The other script, `Model Answer/Task Performance/error_type_analysis.py`, also adapted from the [StatQA Repository](https://github.com/HKUSTDial/StatQA/tree/main), used computes the distribution of errors made by each model summarised in `Model Answer/Task Performance/error_analysis_summary.csv`, and also creates the error analysis figures seen in the `Figures/` folder.

# Plotting for Figures

The remaining figures in the `Figures/` folder are generated by running the `figure_plots.sh` script utilising the `d_train_figure.py`, `d_train_stateval_figure.py` and `stateval_figure.py` scripts. These scripts plot the overall performances of the three tuning sweeps carried out, to allow for analysis and selection of best performing models.

# References

[1] Zhu, S. Du, B. Li, Y. Luo, and N. Tang. Are large language models good statisticians?, 2024.

[2]  Lu, R. Yang, Y. Zhang, S. Yu, R. Dai, Z. Wang, J. Xiang, W. E, S. Gao, X. Ruan, Y. Huang, C. Xi, H. Hu, Y. Fu, Q. Yu, X. Wei, J. Gu, R. Sun, J. Jia, and F. Zhou. Stateval: A comprehensive benchmark for large language models in statistics, 2025.

