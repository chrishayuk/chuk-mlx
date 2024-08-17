import json
import argparse
import os
from utils.tokenizer_loader import load_tokenizer

def load_and_view_dataset(file_path, tokenizer_path, num_entries=5):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    # Initialize the tokenizer using the provided tokenizer path
    try:
        tokenizer = load_tokenizer(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer from '{tokenizer_path}': {e}")
        return

    print(f"Contents of '{file_path}':")
    
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                print(f"\nEntry {i + 1}:")
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {i + 1}: {e}")
                    continue
                
                for key, value in data.items():
                    print(f"Key: {key}")
                    if isinstance(value, str):
                        try:
                            # Tokenize the text
                            tokens = tokenizer.encode(value, return_tensors="pt")
                            print("Tokenized tensor:")
                            print(tokens)
                            decoded_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
                            print(f"Decoded back: {decoded_text}")
                        except Exception as e:
                            print(f"Error during tokenization: {e}")
                    else:
                        print(value)
                
                if i + 1 >= num_entries:  # Limit to a specified number of entries
                    print(f"\n... (showing first {num_entries} entries)")
                    break
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View contents of a JSONL dataset file")
    parser.add_argument("--batch_file", default="./sample_data/calvin_scale_llama/train.jsonl", help="Path to the JSONL batch file")
    parser.add_argument("--tokenizer", default=None, help="Path to the tokenizer; defaults to model path if not specified")
    parser.add_argument("--model", default="ibm-granite/granite-3b-code-instruct", help="Path to the model or tokenizer")
    parser.add_argument("--num_entries", type=int, default=5, help="Number of entries to display")
    
    args = parser.parse_args()
    
    # Determine which path to use for the tokenizer
    tokenizer_path = args.tokenizer if args.tokenizer else args.model

    # Load the dataset and view its contents
    load_and_view_dataset(args.batch_file, tokenizer_path, args.num_entries)
