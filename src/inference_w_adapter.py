from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Base model
BASE_MODEL = "Qwen/Qwen2.5-7B"

# Create the LLM with LoRA enabled
llm = LLM(model=BASE_MODEL, 
          enable_lora=True, 
          tensor_parallel_size=2,
          max_model_len = 2048,
          gpu_memory_utilization = 0.9,
          max_lora_rank=256,
          trust_remote_code=True)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0,
    top_p=0.95,
    max_tokens=1024
)

# Your prompts
prompts = [
'''
You are tasked with determining whether a given document is relevant to answering a specific query. Follow these steps carefully:

1. You will be presented with a query and a document containing a title and content.
2. First, examine the query.
3. Next, examine the document.

4. Analyze the query carefully. Determine what specific information or answer it is seeking. Consider the key concepts, entities, or relationships mentioned in the query.

5. Carefully read and analyze the document content. Pay attention to the main topics, key information, and any details that might be relevant to the query.

6. Reason about the relevance of the document to the query. Consider the following:
- Does the document contain information that directly answers the query?
- Does it provide context or background information that would be helpful in understanding or answering the query?
- Are there any significant matches between key terms or concepts in the query and the document?
- Even if the document doesn't fully answer the query, does it contain partial information that could contribute to an answer?

7. Based on your analysis, determine whether the document is relevant or not relevant to answering the query.

8. Provide your reasoning and verdict in the following format:
<reasoning>
[Explain your thought process here, discussing why you believe the document is or is not relevant to the query. Provide specific examples or quotes from the document if applicable.]
</reasoning>
<relevance>[Insert either 0 for not relevant or 1 for relevant]</relevance>

Remember, the content within the <relevance> tags must be either 0 or 1, with no other text or explanation.

<question>
How does the use of the STEP system facilitate the procurement process for consulting firms and individual consultants in the Kiribati Health Systems Strengthening Project?
</question>

<document>
9 of the “World Bank Procurement Regulations for IPF Borrowers” (November 2020) (“Procurement Regulations”) the Bank’s Systematic Tracking and Exchanges in Procurement (STEP) system will be used to prepare, clear and update Procurement Plans and conduct all procurement transactions for the Project. This textual part along with the Procurement Plan tables in STEP constitute the Procurement Plan for the Project. The following conditions apply to all procurement activities in the Procurement Plan. The other elements of the Procurement Plan as required under paragraph 4. 4 of the Procurement Regulations are set forth in STEP. The Bank’s Standard Procurement Documents: shall be used for all contracts subject to international competitive procurement and those contracts as specified in the Procurement Plan tables in STEP. National Procurement Arrangements: In accordance with paragraph 5.
</document>
'''
]

# Create a LoRA request with your adapter
adapter_path = "sumukshashidhar-testing/reasoning-v0.2-qwen2.5-7b"
outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest("reasoning_adapter", 1, adapter_path)
)

# Print results
for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Output: {output.outputs[0].text}")
    print("-" * 40)