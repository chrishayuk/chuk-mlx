import numpy as np
import argparse
import os
import sys

# Add the parent directory of the tools directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# imports
from core.utils.tokenizer_loader import load_tokenizer

def format_tensor(tensor):
    tensor_str = np.array2string(tensor, separator=', ', threshold=np.inf)
    tensor_str = tensor_str.replace('\n', '\n           ')  # Indent continuation lines
    return f"tensor([{tensor_str[1:-1]}])"

def load_and_view_dataset(file_path, tokenizer="mistralai/Mistral-7B-v0.1"):
    data = np.load(file_path)
    tokenizer = load_tokenizer(tokenizer)

    print(f"Contents of '{file_path}':")
    
    # Iterate through all keys in the dataset
    for key in data.files:
        print(f"\nKey: {key}")
        for i in range(len(data[key])):
            print(f"\nEntry {i + 1}:")
            print(f"Tensor ({key}):")
            print(format_tensor(data[key][i]))

            # Decode the tensor
            decoded_text = tokenizer.decode(data[key][i][data[key][i] != 0])
            print(f"Decoded: {decoded_text}")
            
            if i >= 1:  # Limit to 2 entries per tensor for brevity
                print("\n... (showing first 2 entries)")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View contents of an NPZ dataset file")
    parser.add_argument("--batch_file", default="output/calvin/batches/calvin_batch_0189.npz", help="Path to the NPZ batch file")
    parser.add_argument("--tokenizer", default="mistralai/Mistral-7B-Instruct-v0.2", help="Name of the model for tokenizer")
    
    args = parser.parse_args()
    
    load_and_view_dataset(args.batch_file, args.tokenizer)
