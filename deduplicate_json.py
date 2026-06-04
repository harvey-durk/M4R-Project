import json
import argparse
from pathlib import Path

def remove_duplicates_by_prompt(input_file, output_file=None, keep_first=True):
    """
    Remove duplicate entries from a JSON file based on the 'prompt' field.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (if None, overwrites input)
        keep_first: If True, keeps first occurrence; if False, keeps last
    
    Returns:
        Number of duplicates removed
    """
    # Load the JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of objects")
    
    original_count = len(data)
    
    # Remove duplicates based on prompt
    seen = {}
    unique_data = []
    
    for item in data:
        prompt = item.get('prompt', '')
        # Use prompt as key; if prompt is missing, use a placeholder
        if prompt not in seen:
            seen[prompt] = True
            unique_data.append(item)
        elif not keep_first:
            # If keeping last, replace the existing entry
            # Find and replace index (slower but preserves order)
            for i, existing in enumerate(unique_data):
                if existing.get('prompt', '') == prompt:
                    unique_data[i] = item
                    break
    
    # Alternative: simpler version if order doesn't matter and you want unique by prompt+response
    def get_key(item):
        return (item.get('prompt', ''), item.get('response', ''))
    
    # For checking duplicates by both prompt AND response (stricter)
    seen_both = {}
    unique_by_both = []
    for item in data:
        key = get_key(item)
        if key not in seen_both:
            seen_both[key] = True
            unique_by_both.append(item)
    
    duplicate_count_by_prompt = original_count - len(unique_data)
    duplicate_count_by_both = original_count - len(unique_by_both)
    
    # Determine which method to use (configurable)
    # Default: remove duplicates by prompt only
    final_data = unique_data
    duplicates_removed = duplicate_count_by_prompt
    
    # Save the result
    output_path = output_file if output_file else input_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"Original entries: {original_count}")
    print(f"Unique entries (by prompt): {len(unique_data)}")
    print(f"Duplicates removed (by prompt): {duplicate_count_by_prompt}")
    print(f"Would be removed if checking both prompt+response: {duplicate_count_by_both}")
    print(f"Saved to: {output_path}")
    
    return duplicates_removed


def remove_duplicates_by_both(input_file, output_file=None):
    """
    Remove duplicates based on both 'prompt' and 'response' fields.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of objects")
    
    original_count = len(data)
    seen = set()
    unique_data = []
    
    for item in data:
        key = (item.get('prompt', ''), item.get('response', ''))
        if key not in seen:
            seen.add(key)
            unique_data.append(item)
    
    output_path = output_file if output_file else input_file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, indent=2, ensure_ascii=False)
    
    print(f"Original entries: {original_count}")
    print(f"Unique entries (by prompt+response): {len(unique_data)}")
    print(f"Duplicates removed: {original_count - len(unique_data)}")
    print(f"Saved to: {output_path}")
    
    return original_count - len(unique_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove duplicate entries from JSON file")
    parser.add_argument("--input", help="Path to input JSON file")
    parser.add_argument("-o", "--output", help="Path to output JSON file (default: overwrite input)")
    parser.add_argument("--by-both", action="store_true", 
                        help="Remove duplicates based on both 'prompt' and 'response' (default: by 'prompt' only)")
    parser.add_argument("--keep-last", action="store_true",
                        help="Keep last occurrence of duplicate (default: keep first)")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: File {args.input} not found")
        exit(1)
    
    if args.by_both:
        remove_duplicates_by_both(args.input, args.output)
    else:
        remove_duplicates_by_prompt(args.input, args.output, 
                                   keep_first=not args.keep_last)