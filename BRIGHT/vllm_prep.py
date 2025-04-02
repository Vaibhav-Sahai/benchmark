import argparse
from datasets import load_dataset
from tqdm import tqdm
import json
import numpy as np
import os
from rank_bm25 import BM25Okapi

def simple_tokenize(text):
    """Simple, robust tokenization function."""
    if not isinstance(text, str):
        print(f"Warning: Expected string but got {type(text)}")
        text = str(text)
    
    # Convert to lowercase and split on whitespace and punctuation
    text = text.lower()
    # Replace punctuation with spaces
    for char in '.,;:!?"\'()[]{}':
        text = text.replace(char, ' ')
    # Split on whitespace and filter empty tokens
    return [token for token in text.split() if token]

def format_prompt(query, document):
    """Format the prompt for relevance classification."""
    return f"""You are tasked with determining whether a given document is relevant to answering a specific query. Follow these steps carefully:
1. You will be presented with a query and a document.
2. First, examine the query.
3. Next, examine the document.
4. Analyze the query carefully to determine what specific information or answer it is seeking.
5. Carefully read and analyze the document content.
6. Reason about the relevance of the document to the query. Consider if the document provides information that directly answers the query or offers valuable context.
7. Based on your analysis, determine whether the document is relevant or not relevant.
8. Provide your reasoning and final verdict in the following format:
<reasoning>
[Explain your thought process here...]
</reasoning>
<relevance>[Insert either 0 for not relevant or 1 for relevant]</relevance>
Remember, the content within the <relevance> tags must be either 0 or 1 with no extra text.
<question>
Query: {query.strip()}
</question>
<document>
Document: {document.strip()}
</document>
Answer:"""

def main():
    parser = argparse.ArgumentParser(description="Prepare BRIGHT dataset for vLLM using BM25")
    parser.add_argument("--task", type=str, default="biology", help="Task from BRIGHT dataset")
    parser.add_argument("--output", type=str, default="vllm_inputs.jsonl", help="Output file path")
    parser.add_argument("--top_k", type=int, default=100, help="Number of top documents per query")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Load data from BRIGHT dataset
    print(f"Loading examples for {args.task}...")
    examples = load_dataset('xlangai/BRIGHT', 'examples')[args.task]
    
    print(f"Loading documents for {args.task}...")
    document_dataset = load_dataset('xlangai/BRIGHT', 'documents')[args.task]
    
    # Debug dataset structure
    print("Example structure:", examples[0])
    print("Document structure:", document_dataset[0])
    
    # Create a dictionary to map doc_id to document content
    doc_id_to_content = {}
    for doc in document_dataset:
        try:
            doc_id_to_content[doc["id"]] = doc["content"]
        except (KeyError, TypeError) as e:
            print(f"Error accessing document data: {e}")
            print(f"Document structure: {doc}")
            # Try alternative access methods if the dataset structure is different
            if isinstance(doc, dict):
                print("Available keys:", doc.keys())
            elif isinstance(doc, (list, tuple)):
                print("Document is a sequence of length:", len(doc))
    
    if not doc_id_to_content:
        raise ValueError("Could not build document content mapping. Check dataset structure.")
    
    # Tokenize all documents for BM25
    print("Tokenizing documents for BM25...")
    tokenized_docs = []
    doc_ids = []
    
    # Process documents directly
    for doc in tqdm(document_dataset):
        try:
            if isinstance(doc, dict):
                doc_id = doc.get("id")
                content = doc.get("content")
            else:
                # If document is not a dict, try to access it as a sequence
                doc_id = doc[0] if len(doc) > 0 else None
                content = doc[1] if len(doc) > 1 else None
                
            if doc_id is None or content is None:
                raise ValueError(f"Could not extract id or content from document: {doc}")
                
            tokenized_docs.append(simple_tokenize(content))
            doc_ids.append(doc_id)
            
        except Exception as e:
            print(f"Error processing document: {e}")
            print(f"Document data: {doc}")
            continue
    
    if not tokenized_docs:
        raise ValueError("No documents were successfully tokenized.")
    
    print(f"Successfully tokenized {len(tokenized_docs)} documents")
    
    # Create BM25 index
    print("Creating BM25 index...")
    try:
        # Use custom parameters similar to the reference implementation
        bm25 = BM25Okapi(tokenized_docs, k1=0.9, b=0.4)
    except Exception as e:
        print(f"Error creating BM25 index: {e}")
        print("Trying with default parameters...")
        bm25 = BM25Okapi(tokenized_docs)
    
    # Prepare prompts
    prompts = []
    
    print("Processing queries...")
    for example in tqdm(examples):
        try:
            if isinstance(example, dict):
                query = example.get("query")
                query_id = example.get("id")
                gold_ids = example.get("gold_ids", [])
                excluded_ids = example.get("excluded_ids", [])
            else:
                # Try to access as a sequence
                query = example[0] if len(example) > 0 else None
                query_id = example[1] if len(example) > 1 else None
                gold_ids = example[2] if len(example) > 2 else []
                excluded_ids = example[3] if len(example) > 3 else []
            
            if query is None or query_id is None:
                raise ValueError(f"Could not extract query or id from example: {example}")
            
            # Skip empty queries
            if not query.strip():
                print(f"Skipping empty query with id {query_id}")
                continue
                
            # Tokenize query
            tokenized_query = simple_tokenize(query)
            
            # Skip empty tokenized queries
            if not tokenized_query:
                print(f"Skipping query with no tokens: '{query}' (id: {query_id})")
                continue
            
            # Get BM25 scores
            doc_scores = bm25.get_scores(tokenized_query)
            
            # Create a dictionary of {doc_id: score}
            id_score_dict = dict(zip(doc_ids, doc_scores))
            
            # Remove excluded documents
            for doc_id in excluded_ids:
                if doc_id in id_score_dict and doc_id != "N/A":
                    id_score_dict.pop(doc_id)
            
            # Sort documents by score and get top-k
            top_doc_ids = [doc_id for doc_id, _ in sorted(id_score_dict.items(), 
                                                         key=lambda x: x[1], 
                                                         reverse=True)[:args.top_k]]
            
            # Make sure all gold documents are included
            for gold_id in gold_ids:
                if gold_id not in top_doc_ids and gold_id not in excluded_ids:
                    top_doc_ids.append(gold_id)
            
            # Create prompts for vLLM
            for doc_id in top_doc_ids:
                if doc_id in doc_id_to_content:
                    prompts.append({
                        "query_id": query_id,
                        "doc_id": doc_id,
                        "is_gold": doc_id in gold_ids,
                        "prompt": format_prompt(query, doc_id_to_content[doc_id])
                    })
        except Exception as e:
            print(f"Error processing example: {e}")
            continue
    
    # Save prompts to file
    print(f"Saving {len(prompts)} prompts to {args.output}...")
    with open(args.output, "w") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + "\n")
    
    print(f"Done! Prompts saved to {args.output}")
    if len(examples) > 0 and len(document_dataset) > 0:
        print(f"Reduced from {len(examples) * len(document_dataset)} to {len(prompts)} prompts")

if __name__ == "__main__":
    main()