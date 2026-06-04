"""
Convert StatEval JSONL to LLaMA-Factory Alpaca format
Supports all question types: Code, Calculation, Choice, Proof, Short Answer, 
Definition, Fill in the Blank, True/False
Recursively flattens dictionary responses into string format
"""

import json
import argparse
from pathlib import Path
import re
from collections import Counter


def clean_latex(text):
    """Clean LaTeX formatting for consistency"""
    if not isinstance(text, str):
        return text
    
    # Replace \( \) with $ $ for better compatibility
    text = re.sub(r'\\\(', '$', text)
    text = re.sub(r'\\\)', '$', text)
    
    # Replace \[ \] with $$ $$
    text = re.sub(r'\\\[', '$$', text)
    text = re.sub(r'\\\]', '$$', text)
    
    return text


def flatten_dict_to_string(obj, indent_level=0, prefix=""):
    """
    Recursively flatten a dictionary or list into a formatted string.
    
    Args:
        obj: The object to flatten (dict, list, str, or other)
        indent_level: Current indentation level for nested structures
        prefix: Optional prefix for the current section
    
    Returns:
        str: Flattened string representation
    """
    indent = "  " * indent_level
    next_indent = "  " * (indent_level + 1)
    lines = []
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            # Format the key as a header
            header = f"{indent}{key}:"
            lines.append(header)
            
            if isinstance(value, dict):
                # Recursively flatten nested dictionary
                lines.append(flatten_dict_to_string(value, indent_level + 1))
            elif isinstance(value, list):
                # Handle lists
                for i, item in enumerate(value, 1):
                    if isinstance(item, dict):
                        lines.append(f"{next_indent}[{i}]")
                        lines.append(flatten_dict_to_string(item, indent_level + 2))
                    else:
                        lines.append(f"{next_indent}- {item}")
            else:
                # Simple value
                lines.append(f"{next_indent}{value}")
    
    elif isinstance(obj, list):
        for i, item in enumerate(obj, 1):
            if isinstance(item, dict):
                lines.append(f"{indent}[Item {i}]")
                lines.append(flatten_dict_to_string(item, indent_level + 1))
            else:
                lines.append(f"{indent}- {item}")
    
    else:
        # Base case: string or other primitive
        lines.append(f"{indent}{obj}")
    
    # Join with newlines and handle nested returns
    result = "\n".join(lines)
    
    # If this is a nested call, return as is; otherwise ensure no trailing newline
    if indent_level > 0:
        return result
    else:
        return result.strip()


def flatten_dict_to_string(obj):
    """
    Recursively flatten a dictionary or list into a compact string representation.
    
    Args:
        obj: The object to flatten (dict, list, str, or other)
    
    Returns:
        str: Compact string representation in format {key1: value1, key2: value2, ...}
    """
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        
        # Process each key-value pair
        items = []
        for key, value in obj.items():
            # Recursively format the value
            formatted_value = flatten_dict_to_string(value)
            items.append(f"{key}: {formatted_value}")
        
        return "{" + ", ".join(items) + "}"
    
    elif isinstance(obj, str):
        # Return string with quotes
        # Clean the string to avoid issues with nested quotes
        cleaned = obj.replace('"', '\\"').replace('\n', ' ').strip()
        return f'"{cleaned}"'

    else:
        # Numbers and other primitives
        return str(obj)


def format_response(response):
    """
    Format the response for fine-tuning.
    If response is a string, return it cleaned.
    If response is a dictionary/list, recursively flatten it to a string.
    If response is None or empty, return empty string.
    
    Args:
        response: The response to format (string, dict, list, or None)
    
    Returns:
        str: Formatted response as a string
    """
    if response is None:
        return ""
    
    if isinstance(response, str):
        # Clean LaTeX in string responses
        return clean_latex(response.strip())
    
    if isinstance(response, (dict, list)):
        # Recursively flatten to string
        return flatten_dict_to_string(response)
    
    # For any other type (int, float, bool, etc.), convert to string
    return str(response)


def get_prompt_by_type(q_type):
    """
    Return appropriate instruction and system prompt based on question type
    """
    prompts = {
        "Code question": {
            "instruction": "Write code to solve the following statistical problem.",
            "system": "You are a statistician proficient in writing code for statistical analysis. Provide executable, well-commented code."
        },
        "Calculation question": {
            "instruction": "Calculate the answer to the following statistics problem.",
            "system": "You are a statistician. Show your step-by-step calculations and provide the final numerical answer."
        },
        "Choice question": {
            "instruction": "Select the correct answer for the following multiple-choice statistics question.",
            "system": "You are a statistician. Choose the correct option and explain why it is correct."
        },
        "Proof question": {
            "instruction": "Prove the following theorem with rigorous mathematical reasoning.",
            "system": "You are a mathematical statistician specializing in proofs. Provide clear, step-by-step proofs with proper mathematical notation."
        },
        "Short answer question": {
            "instruction": "Answer the following statistics question concisely and accurately.",
            "system": "You are a statistician. Provide a brief but complete answer."
        },
        "Short answer": {
            "instruction": "Answer the following statistics question concisely and accurately.",
            "system": "You are a statistician. Provide a brief but complete answer."
        },
        "Fill in the blank question": {
            "instruction": "Complete the following statement with the correct statistical term or value.",
            "system": "You are a statistician. Fill in the blank with the most appropriate term or value."
        },
        "True/False question": {
            "instruction": "Determine whether the following statement is true or false, and explain your reasoning.",
            "system": "You are a statistician. State whether the statement is true or false, then provide a brief justification."
        },
        "Definition question": {
            "instruction": "Define the following statistical term or concept clearly and precisely.",
            "system": "You are a statistician. Provide a clear, accurate definition with relevant examples if helpful."
        }
    }
    
    # Default fallback
    default = {
        "instruction": "Answer the following statistics question.",
        "system": "You are a statistician with expertise in probability theory, statistical inference, and mathematical proofs."
    }
    
    return prompts.get(q_type, default)


def format_question(item):
    """Format based on question type with specialized prompts"""
    q_type = item.get('type', '')
    question = clean_latex(item['question'])
    
    # Get the response (could be 'answer' or 'response' field)
    raw_response = item.get('answer') or item.get('response')
    
    # Format response (flatten dicts/lists, clean strings)
    formatted_response = format_response(raw_response)
    
    # Get specialized prompts
    prompts = get_prompt_by_type(q_type)
    
    return {
        "prompt": prompts["instruction"] + " " + question,
        "response": formatted_response,
    }


def convert_stateval_to_llamafactory(input_file, output_file, split_ratio=None, sample_size=None):
    """
    Convert StatEval JSONL to LLaMA-Factory format
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSON file
        split_ratio: If provided, split into train/val (e.g., 0.9 for 90% train)
        sample_size: If provided, only process this many examples
    """
    print(f"[+] Reading: {input_file}")
    
    # Read JSONL and process examples
    data = []
    dict_response_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                
                # Check response type for reporting
                response = item.get('answer') or item.get('response')
                
                if isinstance(response, dict):
                    dict_response_count += 1
                
                data.append(item)
                
            except json.JSONDecodeError as e:
                print(f"[!] Error at line {line_num}: {e}")
    
    print(f"\n[+] Loaded {len(data)} total examples")
    print(f"    - Dictionary responses (will be flattened): {dict_response_count}")
    
    if len(data) == 0:
        print("[!] No examples found. Exiting.")
        return None
    
    # Sample if requested
    if sample_size and sample_size < len(data):
        import random
        random.seed(42)
        data = random.sample(data, sample_size)
        print(f"[+] Sampled {sample_size} examples")
    
    # Count types
    type_counts = Counter()
    for item in data:
        t = item.get('type', 'unknown')
        type_counts[t] += 1
    
    print("\n[+] Question type distribution:")
    for t, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(data)) * 100
        print(f"    - {t}: {count} ({percentage:.1f}%)")
    
    # Convert to Alpaca format
    converted = [format_question(item) for item in data]
    
    # Split if requested
    if split_ratio and 0 < split_ratio < 1:
        import random
        random.seed(42)
        random.shuffle(converted)
        
        split_idx = int(len(converted) * split_ratio)
        train_data = converted[:split_idx]
        val_data = converted[split_idx:]
        
        # Save splits
        train_file = output_file.replace('.json', '_train.json')
        val_file = output_file.replace('.json', '_val.json')
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n[+] Split complete:")
        print(f"    Train: {len(train_data)} examples -> {train_file}")
        print(f"    Validation: {len(val_data)} examples -> {val_file}")
        
        return train_file, val_file
    
    else:
        # Save single file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)
        
        print(f"\n[+] Saved: {output_file}")
        return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert StatEval JSONL to LLaMA-Factory format')
    parser.add_argument('--input', type=str, required=True,
                       help='Input JSONL file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file')
    parser.add_argument('--split', type=float, default=None,
                       help='Split ratio for train/validation')
    parser.add_argument('--sample', type=int, default=None,
                       help='Sample N examples')
    parser.add_argument('--dataset-name', type=str, default='stateval',
                       help='Dataset name for dataset_info.json')
    
    args = parser.parse_args()
    
    # Convert
    result = convert_stateval_to_llamafactory(
        args.input, 
        args.output, 
        args.split,
        args.sample
    )
    
    if result is None:
        print("\n[!] No examples to convert. Exiting.")
        exit(1)
    
    # Generate helper files
    if isinstance(result, tuple):
        filename = result[0]  # Use train file for dataset info
    else:
        filename = result
    
    print("\n Conversion complete!")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Output files: {filename}")
    print("\n Next steps:")
    print("1. Copy the JSON file to LLaMA-Factory/data/")
    print("2. Add the dataset_info.json entry")