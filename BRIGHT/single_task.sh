#!/bin/bash

# This script processes a single BRIGHT benchmark task
# Usage: ./process_task.sh task_name [model_path] [ft]
# If "ft" is provided as the third argument, files will be saved with "-ft" suffix

if [ -z "$1" ]; then
    echo "Error: Task name is required"
    echo "Usage: ./process_task.sh task_name [model_path] [ft]"
    exit 1
fi

TASK="$1"
MODEL="${2:-../qwen2.5-7b-reasoning-merged/}"
FT_SUFFIX=""

# Check if "ft" argument is provided
if [ "$3" = "ft" ]; then
    FT_SUFFIX="-ft"
    echo "Fine-tuned model mode enabled. Files will be saved with -ft suffix."
fi

echo "================================================"
echo "Processing task: $TASK"
echo "Using model: $MODEL"
echo "================================================"

mkdir -p "json/$TASK"

# Skip if the inputs file already exists (same for both base and ft)
INPUTS_FILE="json/$TASK/vllm_inputs.jsonl"
if [ -f "$INPUTS_FILE" ]; then
    echo "vLLM inputs file already exists at $INPUTS_FILE. Skipping preparation step."
else
    echo "Preparing vLLM inputs for $TASK..."
    python vllm_prep.py --task "$TASK" --top_k 100 --output "$INPUTS_FILE"
fi

# Step 2: Run vLLM generation
echo "Running vLLM generation for $TASK..."
python vllm_generation.py \
    --input "$INPUTS_FILE" \
    --output "json/$TASK/vllm_results${FT_SUFFIX}.json" \
    --model "$MODEL" \
    --checkpoint "json/$TASK/checkpoint${FT_SUFFIX}.json"

# Step 3: Calculate metrics
echo "Calculating metrics for $TASK..."
python calculate_ans.py \
    --results_file "json/$TASK/vllm_results${FT_SUFFIX}.json" \
    --task "$TASK" \
    --ground_truth_file "json/$TASK/ground_truth.json" \
    --converted_output "json/$TASK/converted_results${FT_SUFFIX}.json" \
    --output_metrics "json/$TASK/metrics${FT_SUFFIX}.json"

echo "Completed processing for $TASK"