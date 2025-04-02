import json
import argparse
import os
from datasets import load_dataset
from retrievers import calculate_retrieval_metrics

def convert_results(results_list):
    """
    Converts a list of result objects (each with query_id, doc_id, relevance, etc.)
    into a nested dictionary in the format:
    {
      "query_id_1": {
        "doc_id_1": score,
        "doc_id_2": score,
        ...
      },
      ...
    }
    
    Uses confidence_score for ranking if available, otherwise falls back to binary relevance.
    """
    ideal_scores = {}
    for entry in results_list:
        qid = entry["query_id"]
        doc_id = entry["doc_id"]
        
        # First try to use confidence_score (continuous value between 0 and 1)
        if "confidence_score" in entry and entry["confidence_score"] is not None:
            score = float(entry["confidence_score"])
        # Otherwise fall back to relevance (0 or 1)
        elif entry["relevance"] is not None:
            score = float(entry["relevance"])
        # Handle null relevance values
        else:
            score = 0.0  # Treat null as not relevant
            
        if qid not in ideal_scores:
            ideal_scores[qid] = {}
        ideal_scores[qid][doc_id] = score
    return ideal_scores

def calculate_gold_document_recall(results_list):
    """
    Calculate the recall for gold documents, i.e., the percentage of gold documents 
    that were correctly identified as relevant.
    
    Returns:
    - correct_gold: int - number of gold documents correctly marked as relevant
    - total_gold: int - total number of gold documents
    - recall: float - recall percentage
    """
    correct_gold = sum(1 for entry in results_list if entry["is_gold"] and entry.get("relevance") == 1)
    total_gold = sum(1 for entry in results_list if entry["is_gold"])
    recall = (correct_gold / total_gold) * 100 if total_gold > 0 else 0
    
    return {
        "correct_gold": correct_gold,
        "total_gold": total_gold,
        "gold_document_recall": recall
    }

def generate_ground_truth(task, long_context, gt_filename):
    """
    Generates ground truth (qrels) from the BRIGHT dataset examples.
    Uses the "gold_ids_long" field if long_context is True,
    otherwise uses the "gold_ids" field.
    """
    print(f"Generating ground truth for task: {task}, long_context: {long_context}")
    key = "gold_ids_long" if long_context else "gold_ids"
    examples = load_dataset("xlangai/BRIGHT", "examples")[task]
    ground_truth = {}
    for e in examples:
        qid = e["id"]
        # Relevant documents are marked as 1.
        ground_truth[qid] = {gid: 1 for gid in e[key]}
        # Optionally, mark excluded docs as 0 if provided.
        for did in e.get("excluded_ids", []):
            if did != "N/A":  # Skip placeholder values
                ground_truth[qid][did] = 0
    # Save the generated ground truth file
    with open(gt_filename, "w") as f:
        json.dump(ground_truth, f, indent=2)
    print(f"Ground truth saved to {gt_filename}")
    return ground_truth

def main():
    parser = argparse.ArgumentParser(
        description="Convert list-based retrieval results to nested dictionary format, generate ground truth if needed, and run evaluation."
    )
    parser.add_argument("--results_file", type=str, required=True,
                        help="Path to the JSON file containing the list of result objects.")
    parser.add_argument("--ground_truth_file", type=str, default="ground_truth.json",
                        help="Path to the ground truth (qrels) JSON file. If not found, it will be generated.")
    parser.add_argument("--task", type=str, default=None,
                        help="The task/dataset to evaluate (e.g., biology, economics, etc.). Required if ground truth is to be generated.")
    parser.add_argument("--long_context", action="store_true",
                        help="Flag indicating whether to use the long-context ground truth (gold_ids_long).")
    parser.add_argument("--converted_output", type=str, default="converted_results.json",
                        help="File to save the converted nested dictionary results.")
    parser.add_argument("--output_metrics", type=str, default="evaluation_metrics.json",
                        help="File to save the evaluation metrics.")
    args = parser.parse_args()
    
    # Load the list-based results
    with open(args.results_file, "r") as f:
        results_list = json.load(f)
    
    # Print statistics about the results
    null_count = sum(1 for entry in results_list if entry["relevance"] is None)
    total_count = len(results_list)
    print(f"Found {null_count} out of {total_count} entries with null relevance ({null_count/total_count*100:.2f}%)")
    
    # Check if we have confidence scores
    has_confidence = any("confidence_score" in entry for entry in results_list)
    if has_confidence:
        print("Using confidence scores for ranking")
        # Calculate statistics on confidence scores
        avg_confidence = sum(entry.get("confidence_score", 0) for entry in results_list) / total_count
        print(f"Average confidence score: {avg_confidence:.4f}")
    else:
        print("No confidence scores found, using binary relevance for ranking")
    
    # Calculate gold document recall
    gold_recall_metrics = calculate_gold_document_recall(results_list)
    print(f"Gold document recall: {gold_recall_metrics['correct_gold']}/{gold_recall_metrics['total_gold']} ({gold_recall_metrics['gold_document_recall']:.2f}%)")
    
    # Convert to nested dictionary format
    converted_results = convert_results(results_list)
    with open(args.converted_output, "w") as f:
        json.dump(converted_results, f, indent=2)
    print(f"Converted results saved to {args.converted_output}")
    
    # Load ground truth if file exists; otherwise, generate it.
    if os.path.exists(args.ground_truth_file):
        with open(args.ground_truth_file, "r") as f:
            ground_truth = json.load(f)
        print(f"Loaded existing ground truth from {args.ground_truth_file}")
    else:
        if args.task is None:
            raise ValueError("Ground truth file not found and --task not provided; cannot generate ground truth.")
        ground_truth = generate_ground_truth(args.task, args.long_context, args.ground_truth_file)
    
    # Ensure every query in the ground truth is present in the converted results.
    for qid in ground_truth.keys():
        if qid not in converted_results:
            converted_results[qid] = {}
    
    # Run the evaluation using the calculate_retrieval_metrics function.
    standard_metrics = calculate_retrieval_metrics(results=converted_results, qrels=ground_truth)
    
    # Combine all metrics
    all_metrics = {**standard_metrics, **gold_recall_metrics}
    
    # Save all metrics to file
    if args.output_metrics:
        with open(args.output_metrics, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Metrics saved to {args.output_metrics}")
    
    print("Evaluation Metrics:")
    print(json.dumps(standard_metrics, indent=2))
    
    # Print additional gold document stats
    if gold_recall_metrics["total_gold"] > 0:
        null_gold = sum(1 for entry in results_list if entry["is_gold"] and entry["relevance"] is None)
        if null_gold > 0:
            print(f"WARNING: {null_gold} gold documents have null relevance judgments")

    # Print top-k correctness for various k values
    for k in [1, 5, 10, 25, 50, 100]:
        top_k_total = 0
        top_k_correct = 0
        
        for qid in ground_truth:
            if qid not in converted_results:
                continue
                
            # Sort documents by score for this query
            sorted_docs = sorted(converted_results[qid].items(), key=lambda x: x[1], reverse=True)[:k]
            
            # Count how many are actually relevant according to ground truth
            for doc_id, _ in sorted_docs:
                if doc_id in ground_truth[qid] and ground_truth[qid][doc_id] == 1:
                    top_k_correct += 1
                top_k_total += 1
                
        if top_k_total > 0:
            precision = top_k_correct / top_k_total
            print(f"Top-{k} precision: {precision:.4f} ({top_k_correct}/{top_k_total})")

if __name__ == "__main__":
    main()