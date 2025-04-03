#!/bin/bash

# List of tasks to run
TASKS=("earth_science" "economics" "psychology" "robotics" "stackoverflow" "sustainable_living" "leetcode" "pony" "aops" "theoremqa")

# Model paths
BASE_MODEL="Qwen/Qwen2.5-7B"
FT_MODEL="../qwen2.5-7b-reasoning-merged/"

# Loop through each task
for TASK in "${TASKS[@]}"; do
    echo "================================================"
    echo "Starting processing for task: $TASK"
    echo "================================================"

    # Run non-ft version on GPUs 2 and 3
    (CUDA_VISIBLE_DEVICES=2,3 ./single_task.sh "$TASK" "$BASE_MODEL") &

    # Run ft version on GPUs 0 and 1
    (CUDA_VISIBLE_DEVICES=0,1 ./single_task.sh "$TASK" "$FT_MODEL" ft) &

    # Wait for both to finish before continuing
    wait

    echo "------------------------------------------------"
    echo "Finished processing task: $TASK"
    echo "------------------------------------------------"
done

echo "All tasks completed!"
