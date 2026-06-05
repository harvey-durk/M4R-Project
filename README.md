# Experimental Setup

```
conda create --name statqa_eval python=3.11.15
conda activate statqa_eval
pip install -r requirements_statqa_eval.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

```
conda create --name statqa python=3.11.14
conda activate statqa
pip install -r requirements_statqa.txt
cd LLaMA-Factory
pip install -e .[torch,metrics]
cp -r ../Finetuning/LLaMA-Factory/* data/
```
