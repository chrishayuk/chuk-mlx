import numpy as np
import argparse
from transformers import AutoTokenizer

def format_tensor(tensor):
    tensor_str = np.array2string(tensor, separator=', ', threshold=np.inf)
    tensor_str = tensor_str.replace('\n', '\n           ')  # Indent continuation lines
    return f"tensor([{tensor_str[1:-1]}])"

def load_and_view_dataset(file_path, model_name="mistralai/Mistral-7B-v0.1"):
    data = np.load(file_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Contents of '{file_path}':")
    
    for i in range(len(data['input_tensor'])):
        print(f"\nEntry {i + 1}:")
        print("Key: text")
        print("Tokenized tensor:")
        print(format_tensor(data['input_tensor'][i]))
        
        decoded_text = tokenizer.decode(data['input_tensor'][i][data['input_tensor'][i] != 0])
        print(f"Decoded: {decoded_text}")
        
        if i >= 1:  # Limit to 2 entries for brevity
            print("\n... (showing first 2 entries)")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View contents of an NPZ dataset file")
    parser.add_argument("--batch_file", default="output/calvin/batches/calvin_batch_0189.npz", help="Path to the NPZ batch file")
    parser.add_argument("--model_name", default="mistralai/Mistral-7B-Instruct-v0.2", help="Name of the model for tokenizer")
    
    args = parser.parse_args()
    
    load_and_view_dataset(args.batch_file, args.model_name)