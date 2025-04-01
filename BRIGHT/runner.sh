#!/bin/bash

# Configuration
MODEL="qwen2.5"  # Set your model
ENCODE_BATCH_SIZE=1024  # Set your batch size
LONG_CONTEXT=false  # Set to true if you want to use long context mode
OUTPUT_BASE_DIR="outputs"  # Base directory for outputs
LOG_FILE="run_all_tasks_$(date +%Y%m%d_%H%M%S).log"  # Create a timestamped log file

# Define all tasks
TASKS=(
    "earth_science"
    "economics"
    "pony"
    "psychology"
    "robotics"
    "stackoverflow"
    "sustainable_living"
    "aops"
    "leetcode"
    "theoremqa_theorems"
    "theoremqa_questions"
)

# Display run configuration
echo "===========================================" | tee -a "$LOG_FILE"
echo "Running all tasks with:" | tee -a "$LOG_FILE"
echo "  Model: $MODEL" | tee -a "$LOG_FILE"
echo "  Encode Batch Size: $ENCODE_BATCH_SIZE" | tee -a "$LOG_FILE"
echo "  Long Context: $LONG_CONTEXT" | tee -a "$LOG_FILE"
echo "  Output Base Dir: $OUTPUT_BASE_DIR" | tee -a "$LOG_FILE"
echo "  Total tasks: ${#TASKS[@]}" | tee -a "$LOG_FILE"
echo "  Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "===========================================" | tee -a "$LOG_FILE"

# Process each task
for task in "${TASKS[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "===========================================" | tee -a "$LOG_FILE"
    echo "Processing task: $task" | tee -a "$LOG_FILE"
    echo "Start time: $(date)" | tee -a "$LOG_FILE"
    echo "===========================================" | tee -a "$LOG_FILE"
    
    # Set up long context flag if enabled
    long_flag=""
    if [ "$LONG_CONTEXT" = true ]; then
        long_flag="--long_context"
    fi
    
    # Construct the command
    cmd="python run.py --task $task --model $MODEL --encode_batch_size $ENCODE_BATCH_SIZE $long_flag --output_dir $OUTPUT_BASE_DIR"
    
    # Run the command
    echo "Running: $cmd" | tee -a "$LOG_FILE"
    eval $cmd 2>&1 | tee -a "$LOG_FILE"
    
    # Check if the command was successful
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "Task '$task' completed successfully." | tee -a "$LOG_FILE"
    else
        echo "ERROR: Task '$task' failed!" | tee -a "$LOG_FILE"
    fi
    
    echo "End time: $(date)" | tee -a "$LOG_FILE"
    echo "===========================================" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "===========================================" | tee -a "$LOG_FILE"
echo "All tasks completed" | tee -a "$LOG_FILE"
echo "End time: $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "===========================================" | tee -a "$LOG_FILE"
