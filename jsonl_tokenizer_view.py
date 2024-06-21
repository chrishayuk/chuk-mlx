import json
import argparse
from transformers import AutoTokenizer
import torch

def load_and_view_dataset(file_path, model_name="mistralai/Mistral-7B-v0.2"):
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Contents of '{file_path}':")
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            print(f"\nEntry {i + 1}:")
            data = json.loads(line)
            
            for key, value in data.items():
                print(f"Key: {key}")
                if isinstance(value, str):
                    # Tokenize the text
                    tokens = tokenizer.encode(value, return_tensors="pt")
                    print("Tokenized tensor:")
                    print(tokens)
                    print(f"Decoded: {value}")
                else:
                    print(value)
            
            if i >= 4:  # Limit to 5 entries for brevity
                print("\n... (showing first 5 entries)")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View contents of a JSONL dataset file")
    parser.add_argument("--batch_file", default="./sample_data/calvin_scale_llama/train.jsonl", help="Path to the JSONL batch file")
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.2", help="Name of the model for tokenizer")
    
    args = parser.parse_args()
    
    load_and_view_dataset(args.batch_file, args.model_name)