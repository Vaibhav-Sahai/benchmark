import argparse
import json
import os
import re
from vllm import LLM, SamplingParams

def extract_relevance(response):
    """Extract the relevance score (0 or 1) from the model's response."""
    pattern = r'<relevance>(\d+)</relevance>'
    match = re.search(pattern, response)
    if match:
        return int(match.group(1))
    return None

def extract_reasoning(response):
    """Extract the reasoning from the model's response."""
    pattern = r'<reasoning>(.*?)</reasoning>'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def truncate_prompt(prompt, max_length):
    """
    Truncate the prompt to ensure its length does not exceed max_length characters.
    Note: This is a simple string slicing approach and does not account for token boundaries.
    """
    if len(prompt) > max_length:
        prompt = prompt[:max_length]
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Generate relevance judgments using vLLM")
    parser.add_argument("--input", type=str, default="vllm_inputs.jsonl", help="Input JSONL file with prompts")
    parser.add_argument("--output", type=str, default="vllm_results.jsonl", help="Output file for results")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B", help="Model to use for generation")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file to resume from")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Load prompts from JSONL file
    print(f"Loading prompts from {args.input}...")
    prompts = []
    with open(args.input, 'r') as f:
        for line in f:
            prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} prompts")
    
    # Initialize vLLM model
    print(f"Initializing vLLM with model: {args.model}")
    llm = LLM(
        model=args.model,
        max_model_len=4096,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        trust_remote_code=True
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    
    # Load checkpoint if available
    processed_indices = set()
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        with open(args.checkpoint, 'r') as f:
            results = json.load(f)
            for result in results:
                processed_indices.add((result["query_id"], result["doc_id"]))
        print(f"Loaded {len(processed_indices)} processed items from checkpoint")
    else:
        results = []
    
    # Prepare the list of new prompts (and corresponding metadata) that haven't been processed yet.
    new_prompts = []
    meta_data = []  # Will store tuples: (query_id, doc_id, is_gold)
    for prompt_data in prompts:
        query_id = prompt_data["query_id"]
        doc_id = prompt_data["doc_id"]
        if (query_id, doc_id) in processed_indices:
            continue
        prompt = prompt_data["prompt"]
        # Truncate the prompt using simple string slicing
        prompt = truncate_prompt(prompt, 4096)
        new_prompts.append(prompt)
        meta_data.append((query_id, doc_id, prompt_data["is_gold"]))
    
    print(f"Generating outputs for {len(new_prompts)} prompts...")
    
    # Generate outputs for all new prompts in one call
    outputs = llm.generate(new_prompts, sampling_params)
    
    # Process each output and update the results
    for meta, output in zip(meta_data, outputs):
        query_id, doc_id, is_gold = meta
        response = output.outputs[0].text
        relevance = extract_relevance(response)
        reasoning = extract_reasoning(response)
        result = {
            "query_id": query_id,
            "doc_id": doc_id,
            "is_gold": is_gold,
            "relevance": relevance,
            "reasoning": reasoning,
            "full_response": response
        }
        results.append(result)
        processed_indices.add((query_id, doc_id))
    
    # Save final results
    print(f"Saving {len(results)} results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate statistics
    relevance_count = sum(1 for r in results if r["relevance"] == 1)
    gold_count = sum(1 for r in results if r["is_gold"])
    gold_relevance_count = sum(1 for r in results if r["is_gold"] and r["relevance"] == 1)
    
    print("Statistics:")
    print(f"  Total documents judged: {len(results)}")
    print(f"  Documents judged relevant: {relevance_count} ({relevance_count/len(results)*100:.1f}%)")
    print(f"  Gold documents: {gold_count}")
    print(f"  Gold documents judged relevant: {gold_relevance_count} ({gold_relevance_count/gold_count*100:.1f}%)")

if __name__ == "__main__":
    main()
