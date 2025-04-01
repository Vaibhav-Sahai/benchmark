from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
import os

# Define paths
base_model_id = "Qwen/Qwen2.5-7B"
adapter_id = "sumukshashidhar-testing/reasoning-v0.2-qwen2.5-7b"
output_path = "./qwen2.5-7b-reasoning-merged"

print(f"Loading base model from {base_model_id}...")
# Load the base model using the same settings as in your training
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,  # Using bf16 as in your config
    return_dict=True,
    low_cpu_mem_usage=True,
    device_map="auto",
    trust_remote_code=True,  # Required for Qwen models as in your config
)

print(f"Loading tokenizer from {base_model_id}...")
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    trust_remote_code=True
)

print(f"Loading adapter from {adapter_id}...")
# Load the adapter onto the base model
peft_model = PeftModel.from_pretrained(base_model, adapter_id)

print("Merging adapter with base model...")
# Merge the adapter weights with the base model
merged_model = peft_model.merge_and_unload()

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

print(f"Saving merged model to {output_path}...")
# Save the merged model and tokenizer
merged_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print("Merge completed successfully!")