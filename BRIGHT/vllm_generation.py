import argparse
import json
import os
import re
import math
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def extract_relevance(response):
    """Extract the binary relevance from the response.
       Expected output format: <relevance>0</relevance> or <relevance>1</relevance>
    """
    pattern = r'<relevance>\s*(\d)\s*</relevance>'
    match = re.search(pattern, response)
    if match:
        # Return 1 if the digit is "1", otherwise 0.
        return 1 if match.group(1).strip() == "1" else 0
    return None

def extract_reasoning(response):
    """Extract the reasoning from the response, expecting it within <reasoning> tags."""
    pattern = r'<reasoning>(.*?)</reasoning>'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def truncate_prompt(prompt, max_length):
    """Truncate the prompt (simple string slicing)."""
    if len(prompt) > max_length:
        prompt = prompt[:max_length]
    return prompt

def main():
    parser = argparse.ArgumentParser(description="Generate relevance judgments using vLLM")
    parser.add_argument("--input", type=str, default="vllm_inputs.jsonl", help="Input JSONL file with prompts")
    parser.add_argument("--output", type=str, default="vllm_results.json", help="Output JSON file for results")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B", help="Model to use for generation")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file to resume from")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="Save checkpoint after processing this many items")
    args = parser.parse_args()

    # Load tokenizer to obtain token IDs (if needed for other purposes)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Load prompts from the input JSONL file
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
    
    # Configure sampling parameters – no stop tokens so that the model generates the full response.
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=args.max_tokens,
        logprobs=20  # Request logprobs for top tokens
    )
    
    # Load checkpoint if available
    processed_indices = set()
    results = []
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        with open(args.checkpoint, 'r') as f:
            results = json.load(f)
            for result in results:
                processed_indices.add((result["query_id"], result["doc_id"]))
        print(f"Loaded {len(processed_indices)} processed items from checkpoint")
    
    # Prepare new prompts and metadata for those not processed yet.
    new_prompts = []
    meta_data = []  # Each item: (query_id, doc_id, is_gold)
    for prompt_data in prompts:
        query_id = prompt_data["query_id"]
        doc_id = prompt_data["doc_id"]
        if (query_id, doc_id) in processed_indices:
            continue
        prompt = prompt_data["prompt"]
        prompt = truncate_prompt(prompt, 4096)
        new_prompts.append(prompt)
        meta_data.append((query_id, doc_id, prompt_data.get("is_gold", False)))
    
    print(f"Generating outputs for {len(new_prompts)} prompts...")
    
    # Generate outputs in one call.
    outputs = llm.generate(new_prompts, sampling_params)
    
    # Process each output and compute the confidence score.
    for meta, output in zip(meta_data, outputs):
        query_id, doc_id, is_gold = meta
        response = output.outputs[0].text
        print(f"\n---\nResponse for query_id: {query_id}, doc_id: {doc_id}\n{response}\n---\n")
        
        # Extract binary relevance and reasoning.
        relevance = extract_relevance(response)
        reasoning = extract_reasoning(response)
        
        # Default confidence_score
        confidence_score = 0.5
        try:
            # Iterate over the token-level logprobs to find the token corresponding to the expected digit.
            logprobs_list = output.outputs[0].logprobs  # List of dicts, one per token.
            token_ids = output.outputs[0].token_ids  # List of token IDs for each token.
            # Extract expected digit from the <relevance> tag in the response.
            match = re.search(r'<relevance>\s*(\d)\s*</relevance>', response)
            if match:
                expected_digit = match.group(1).strip()  # "0" or "1"
                found = False
                for i, token_id in enumerate(token_ids):
                    token_logprob_dict = logprobs_list[i]
                    token_info = token_logprob_dict.get(token_id)
                    if token_info is None:
                        continue
                    decoded = token_info.decoded_token.strip()
                    if decoded == expected_digit:
                        # Use the exponentiated logprob as the confidence score.
                        confidence_score = math.exp(token_info.logprob)
                        found = True
                        break
                if not found and relevance is not None:
                    confidence_score = float(relevance)
            else:
                if relevance is not None:
                    confidence_score = float(relevance)
        except Exception as e:
            print(f"Error extracting confidence score: {e}")
            if relevance is not None:
                confidence_score = float(relevance)
        
        result = {
            "query_id": query_id,
            "doc_id": doc_id,
            "is_gold": is_gold,
            "relevance": relevance,
            "reasoning": reasoning,
            "confidence_score": confidence_score,
            "full_response": response
        }
        results.append(result)
        processed_indices.add((query_id, doc_id))
        
        # Optionally, save checkpoint periodically.
        if args.checkpoint and len(results) % args.checkpoint_interval == 0:
            print(f"Saving checkpoint after processing {len(results)} results...")
            with open(args.checkpoint, 'w') as f:
                json.dump(results, f, indent=2)
    
    # Save final results.
    print(f"Saving {len(results)} results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    if args.checkpoint:
        with open(args.checkpoint, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Report statistics.
    total = len(results)
    gold_count = sum(1 for r in results if r["is_gold"])
    relevance_count = sum(1 for r in results if r["relevance"] == 1)
    null_count = sum(1 for r in results if r["relevance"] is None)
    print("Statistics:")
    print(f"  Total documents judged: {total}")
    print(f"  Documents judged relevant: {relevance_count} ({relevance_count/total*100:.1f}%)")
    print(f"  Documents with null judgments: {null_count} ({null_count/total*100:.1f}%)")
    if gold_count > 0:
        gold_relevance_count = sum(1 for r in results if r["is_gold"] and r["relevance"] == 1)
        print(f"  Gold documents: {gold_count}")
        print(f"  Gold documents judged relevant: {gold_relevance_count} ({gold_relevance_count/gold_count*100:.1f}%)")

if __name__ == "__main__":
    main()
