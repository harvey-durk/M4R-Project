# -*- coding: utf-8 -*-
"""
llama_sft_logic_evaluation.py
────────────────────────
Evaluation pipeline:
1. vllm  - LLM generates full answer (columns + methods)
2. Parse - extract columns and methods from LLM response
3. Infer - derive task category from LLM's predicted methods
4. Rules - re-select applicable methods for that category
           using column metadata from the prompt
5. Save  - write merged answer (LLM columns + rule methods) to CSV
"""

import sys
import os
import gc
import json
import time
import argparse
import re

from collections import defaultdict
from dataclasses import dataclass, field

import torch
import pandas as pd

main_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, main_folder_path)

os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")

from vllm import LLM, SamplingParams

# Paths
model_ans_path = "Model Answer/"
prompt_dataset_path_test = "Data/Integrated Dataset/Dataset with Prompt/Test Set/"

# Category and method vocabulary
category_map = {
    "Correlation Analysis":         ["Pearson Correlation Coefficient",
                                     "Spearman Correlation Coefficient",
                                     "Kendall Correlation Coefficient",
                                     "Partial Correlation Coefficient"],
    "Distribution Compliance Test": ["Anderson-Darling Test",
                                     "Shapiro-Wilk Test of Normality",
                                     "Kolmogorov-Smirnov Test for Normality",
                                     "Lilliefors Test",
                                     "Kolmogorov-Smirnov Test",
                                     "Kolmogorov-Smirnov Test for Uniform distribution",
                                     "Kolmogorov-Smirnov Test for Gamma distribution",
                                     "Kolmogorov-Smirnov Test for Exponential distribution"],
    "Contingency Table Test":       ["Chi-square Independence Test",
                                     "Fisher Exact Test",
                                     "Mantel-Haenszel Test"],
    "Descriptive Statistics":       ["Mean", "Median", "Mode", "Range", "Quartile",
                                     "Standard Deviation", "Skewness", "Kurtosis"],
    "Variance Test":                ["Mood Variance Test", "Levene Test",
                                     "Bartlett Test", "F-Test for Variance"],
}

category_names = list(category_map.keys())
method_to_cat = {m: c for c, ms in category_map.items() for m in ms}
cat_to_idx = {c: i for i, c in enumerate(category_names)}
idx_to_cat = {i: c for c, i in cat_to_idx.items()}

# Data structures
@dataclass
class ColumnInfo:
    header: str
    data_type: str
    num_rows: int
    is_normality: bool


@dataclass
class ParsedPrompt:
    columns: list = field(default_factory=list)
    selected_cols: list = field(default_factory=list)
    has_strata: bool = False
    question_text: str = ""


# Prompt parser functions
def parse_prompt(prompt_text: str, llm_columns: list = None) -> ParsedPrompt:
    result = ParsedPrompt()

    col_pattern = re.compile(
        r"column_header:\s*(.+?);\s*"
        r"data_type:\s*(\w+);\s*"
        r"num_of_rows:\s*(\d+);\s*"
        r"is_normality:\s*(True|False)",
        re.IGNORECASE
    )
    for match in col_pattern.finditer(prompt_text):
        result.columns.append(ColumnInfo(
            header=match.group(1).strip(),
            data_type=match.group(2).strip().lower(),
            num_rows=int(match.group(3)),
            is_normality=(match.group(4).strip().lower() == "true"),
        ))

    q_match = re.search(
        r"###\s*Statistical Question:\s*(.+?)(?=###|$)",
        prompt_text, re.DOTALL | re.IGNORECASE
    )
    if q_match:
        result.question_text = q_match.group(1).strip()

    if llm_columns:
        result.selected_cols = llm_columns
        result.has_strata = len(llm_columns) > 2

    return result


def get_selected_column_info(parsed: ParsedPrompt) -> list:
    col_map = {c.header.strip().lower(): c for c in parsed.columns}

    if parsed.selected_cols:
        info = []
        for name in parsed.selected_cols:
            key = name.strip().lower()
            if key in col_map:
                info.append(col_map[key])
            else:
                matches = [c for h, c in col_map.items() if key in h or h in key]
                if matches:
                    info.append(matches[0])
        return info if info else [c for c in parsed.columns if c.data_type != "id"]
    return [c for c in parsed.columns if c.data_type != "id"]


# Prerequisite helpers
def all_normal(cols):
    quant = [c for c in cols if c.data_type == "quantitative"]
    return bool(quant) and all(c.is_normality for c in quant)

def question_mentions(text, *keywords):
    t = text.lower()
    return any(kw.lower() in t for kw in keywords)


# Per-category rule functions (CTT and VT)
def contingency_methods(cols, parsed):
    methods  = []
    n_rows = cols[0].num_rows
    n_cols = len([c for c in cols if c.data_type == "categorical"])
    if n_cols == 2:
        if n_rows >= 40:
            methods.append("Chi-square Independence Test")
        elif n_rows < 500:
            methods.append("Fisher Exact Test")
    elif n_cols == 3:
        methods.append("Mantel-Haenszel Test")
    return methods

def variance_methods(cols, parsed):
    quant_cols = [c for c in cols if c.data_type == "quantitative"]
    n_cols = len(quant_cols)
    methods = []
    if n_cols >= 2:
        methods.append("Mood Variance Test")
        methods.append("Levene Test")
        if all_normal(quant_cols):
            methods.append("Bartlett Test")
            if n_cols == 2:
                methods.append("F-Test for Variance")
    return methods

category_rule_fn = {
    "Correlation Analysis": None,
    "Distribution Compliance Test": None,
    "Contingency Table Test": contingency_methods,
    "Descriptive Statistics": None,
    "Variance Test": variance_methods,
}

# LLM response parser function
def parse_llm_answer(response_text: str) -> dict:
    """
    Parse both 'columns' and 'methods' from an LLM response string.
    Handles CSV double-quote escaping, duplicate JSON objects, preamble text.

    Returns:
        {"columns": list[str], "methods": list[str]}
    """
    if not response_text or not isinstance(response_text, str):
        return {"columns": [], "methods": []}

    text = response_text.replace('""', '"').strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].replace('""', '"')

    def extract_fields(parsed: dict):
        cols = parsed.get("columns") or parsed.get("column") or []
        mths = parsed.get("methods") or parsed.get("method") or []
        cols = [str(c).strip() for c in cols if c] if isinstance(cols, list) else []
        mths = [str(m).strip() for m in mths if m] if isinstance(mths, list) else []
        return cols, mths

    # Try each JSON object found in the text
    for match in re.finditer(r'\{.*?\}', text, re.DOTALL):
        try:
            parsed = json.loads(match.group())
            cols, mths = extract_fields(parsed)
            if cols or mths:
                return {"columns": cols, "methods": mths}
        except json.JSONDecodeError:
            continue

    # Fallback: span between first { and last }
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start:end])
            cols, mths = extract_fields(parsed)
            return {"columns": cols, "methods": mths}
        except json.JSONDecodeError:
            pass

    return {"columns": [], "methods": []}


# Category inference from methods
def infer_category_from_methods(methods: list) -> int:
    """
    Derive task category index from LLM-predicted method names.
    Uses majority vote. Returns index of Descriptive Statistics (3)
    as fallback if no methods are recognised.
    """
    votes = defaultdict(int)
    for method in methods:
        cat = method_to_cat.get(method)
        if cat:
            votes[cat] += 1
    if not votes:
        return 3  # fallback: Descriptive Statistics
    best_cat = max(votes, key=votes.get)
    return cat_to_idx[best_cat]


# Rule-based method selection function
def predict_methods_from_category(category_idx: int,
                                   prompt_text: str,
                                   llm_columns: list = None) -> dict:
    category_name = idx_to_cat.get(category_idx, "Descriptive Statistics")
    rule_fn       = category_rule_fn[category_name]
    parsed        = parse_prompt(prompt_text, llm_columns)
    sel_cols      = get_selected_column_info(parsed)
    methods       = rule_fn(sel_cols, parsed)
    return {
        "predicted_category": category_name,
        "applicable_methods": methods,
    }


# Final answer builder function
def build_final_answer(llm_columns: list, rule_result: dict) -> str:
    """
    Assemble merged answer JSON:
      columns → from LLM
      methods → from rule-based selector (derived from LLM's category prediction)
    """
    return json.dumps({
        "columns": llm_columns,
        "methods": rule_result["applicable_methods"],
    })


# GPU cleanup function
def cleanup_gpu(obj):
    del obj
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# Model registry 
model_registry = {
    "2_7b_sft_logic": ("Models/Sweep/Llama-2-7b-chat-hf-sft-89", 1),
    "3_8b_sft_logic": ("Models/Sweep/Meta-Llama-3-8b-sft-28", 1),
    "3_8b_instruct_sft_logic": ("Models/Sweep/Meta-Llama-3-8b-Instruct-sft-28", 1),
}

# Main
def llama_answer_generation(
    model_type:  str,
    dataset_name: str,
    output_name:  str,
    trick:        str,
):
    if model_type not in model_registry:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         f"Choose from: {list(model_registry)}")

    model_path, parallel_num = model_registry[model_type]
    max_tokens = 1024 if trick in ("zero-shot-CoT", "one-shot-CoT") else 256
    file_path = prompt_dataset_path_test + dataset_name + " for " + trick + ".csv"
    output_path = (model_ans_path + "Origin Answer/" + output_name
                   + model_type + "_" + trick + ".csv")

    t0 = time.time()

    # Load vllm
    llm = LLM(
        model=model_path,
        tensor_parallel_size=parallel_num,
        gpu_memory_utilization=0.90,
    )
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=max_tokens
    )
    print(f"[i] vllm loaded: {model_path}")

    # Load dataset
    df = pd.read_csv(file_path)
    for col in ("llm_raw_answer", "llm_task", "model_answer"):
        if col not in df.columns:
            df[col] = ""
    print(f"[i] Dataset loaded: {len(df)} rows from {file_path}")

    load_time = time.time() - t0
    gen_t0 = time.time()
    batch_size = 10

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]

        # Track true dataframe indices of pending rows
        pending_indices = [idx for idx in batch_df.index]
        prompts = [df.at[idx, "prompt"] for idx in pending_indices]

        # Step 1: LLM generates columns + methods
        outputs = llm.generate(prompts, sampling_params)
        llm_responses = [o.outputs[0].text for o in outputs]

        # Steps 2–5: parse -> infer category -> rules -> merge
        for df_idx, llm_resp, prompt in zip(pending_indices, llm_responses, prompts):

            # Step 2: parse LLM response into columns and methods
            llm_answer = parse_llm_answer(llm_resp)
            llm_columns = llm_answer["columns"]
            llm_methods = llm_answer["methods"]

            # Step 3: infer task category from LLM's predicted methods
            cat_idx = infer_category_from_methods(llm_methods)
            category_name = idx_to_cat[cat_idx]

            # Step 4: build merged answer = LLM columns + rule methods
            if cat_idx in [0, 1, 3]:
                merged = llm_resp
            else:
                rule_result = predict_methods_from_category(
                    category_idx=cat_idx,
                    prompt_text=prompt,
                    llm_columns=llm_columns,
                )
                merged = build_final_answer(llm_columns, rule_result)

            df.at[df_idx, "llm_raw_answer"] = llm_resp
            df.at[df_idx, "llm_task"] = category_name
            df.at[df_idx, "model_answer"] = merged

        df.to_csv(output_path, index=False)
        print(f"[+] Batch {i}–{min(i+batch_size-1, len(df)-1)} done | "
              f"model={model_type} trick={trick}")

    gen_time = time.time() - gen_t0
    print(f"\n[i] Output → {output_path}")
    print("-" * 70)
    print(f"[i] Load: {load_time:.1f}s   Generation: {gen_time:.1f}s")
    print("-" * 70)

    cleanup_gpu(llm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StatQA: LLM columns + rule-based method selection (no MLP)")
    parser.add_argument("--model_type", type=str, default="3_8b_sft_logic", help=f"Model type. Choose from: {list(model_registry)}")
    parser.add_argument("--dataset_name", type=str, default="mini-StatQA")
    parser.add_argument("--output_name", type=str, default="llama")
    parser.add_argument("--trick", type=str, default="zero-shot")
    args = parser.parse_args()

    llama_answer_generation(
        model_type=args.model_type,
        dataset_name=args.dataset_name,
        output_name=args.output_name,
        trick=args.trick,
    )