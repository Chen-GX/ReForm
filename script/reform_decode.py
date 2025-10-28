"""
Mathematical Problem Autoformalization

Translate mathematical problems from natural language to Lean 4 using vLLM.

Usage:
    python autoformalize.py -m <model_path> -d <data_path>
"""

import os
import re
import json
import argparse
from typing import Optional, List, Dict
from pathlib import Path

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

def load_data(file_path: str) -> List[Dict]:
    """Load JSONL data file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    return data


def save_data(data: List[Dict], file_path: str) -> None:
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def format_prompt(informal_statement: str, tokenizer) -> str:
    """Format input as chat prompt."""
    user_prompt = """Think step by step to translate the mathematical problem in natural language to Lean 4, and verify the consistency.
{informal_statement}
"""
    messages = [
        {"role": "user", "content": user_prompt.format(informal_statement=informal_statement)}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_lean4_code(text: str) -> Optional[str]:
    """Extract the last Lean 4 code block from ```lean4...``` tags."""
    pattern = r"```lean4\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1].strip() if matches else None
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Autoformalize mathematical problems to Lean 4")
    parser.add_argument("-m", "--model-path", required=True, help="Path to model")
    parser.add_argument("-d", "--data-path", required=True, help="Path to input JSONL file")
    parser.add_argument("-o", "--output-dir", default="output", help="Output directory")
    parser.add_argument("-n", "--n-samples", type=int, default=1, help="Number of samples per input")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}")
    data = load_data(args.data_path)
    print(f"Loaded {len(data)} items")
    
    # Initialize model
    print(f"Initializing model from {args.model_path}")
    torch.manual_seed(42)
    model = LLM(model=args.model_path, seed=42, trust_remote_code=True)
    tokenizer = model.get_tokenizer()
    
    # Configure sampling
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=32768,
        top_p=0.95,
        n=args.n_samples,
        skip_special_tokens=False,
        presence_penalty=1.5
    )
    
    # Prepare prompts
    print("Preparing prompts...")
    prompts = []
    valid_data = []
    for item in data:
        if item.get("informal_statement"):
            prompts.append(format_prompt(item["informal_statement"], tokenizer))
            valid_data.append(item)
    
    print(f"Generated {len(prompts)} prompts")
    
    # Generate responses
    print("Generating responses...")
    responses = model.generate(prompts, sampling_params)
    
    # Process results
    print("Processing results...")
    results = []
    for idx, item in enumerate(tqdm(valid_data)):
        formal_statements = []
        response_texts = []
        
        for output in responses[idx].outputs:
            response_text = output.text.strip()
            response_texts.append(response_text)
            
            lean_code = extract_lean4_code(response_text)
            if lean_code:
                formal_statements.append(lean_code)
        
        item["formal_statements"] = formal_statements
        item["llm_response"] = response_texts
        results.append(item)
    
    # Save results
    output_file = Path(args.output_dir) / f"{Path(args.data_path).stem}_formalized_n{args.n_samples}.jsonl"
    save_data(results, str(output_file))
    
    # Print summary
    print(f"Completed: {len(results)} items processed")
    print(f"Output: {output_file}")


if __name__ == '__main__':
    main()

