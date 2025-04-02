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
    """
    ideal_scores = {}
    for entry in results_list:
        qid = entry["query_id"]
        doc_id = entry["doc_id"]
        # Force conversion to float
        score = float(entry["relevance"])
        if qid not in ideal_scores:
            ideal_scores[qid] = {}
        ideal_scores[qid][doc_id] = score
    return ideal_scores

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
    args = parser.parse_args()
    
    # Load the list-based results
    with open(args.results_file, "r") as f:
        results_list = json.load(f)
    
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
    metrics = calculate_retrieval_metrics(results=converted_results, qrels=ground_truth)
    
    print("Evaluation Metrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
