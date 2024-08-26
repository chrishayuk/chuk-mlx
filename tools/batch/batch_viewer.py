import argparse
import numpy as np
import os
import sys

# Add the parent directory of the tools directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# imports
from core.utils.tokenizer_loader import load_tokenizer
from core.batch.sequence_visualizer import visualize_sequences

def analyze_batch_file(batch_file, tokenizer_name, tensor_type='both', num_rows=None):
    # Load the tokenizer
    tokenizer = load_tokenizer(tokenizer_name) if tokenizer_name else None

    # Load the batch
    batch_tensor = np.load(batch_file)

    # Analyze the specified tensor type or both
    if tensor_type == 'input':
        tensor_key = 'input_tensor'
    elif tensor_type == 'target':
        tensor_key = 'target_tensor'
    elif tensor_type == 'attention_mask_tensor':
        tensor_key = 'attention_mask_tensor'
    elif tensor_type == 'both':
        input_tensor_key = 'input_tensor'
        target_tensor_key = 'target_tensor'
        attention_mask_tensor_key = 'attention_mask_tensor'
    else:
        raise ValueError("Invalid value for 'tensor_type'. Use 'input', 'target', or 'both'.")

    if tensor_type == 'both':
        if input_tensor_key not in batch_tensor or target_tensor_key not in batch_tensor:
            raise KeyError("Either 'input_tensor' or 'target_tensor' is missing in the .npz file")

        input_tensor = batch_tensor[input_tensor_key]
        target_tensor = batch_tensor[target_tensor_key]
        attention_mask_tensor = batch_tensor[attention_mask_tensor_key]

        # Get the maximum sequence lengths
        max_input_sequence_length = input_tensor.shape[1]
        max_target_sequence_length = target_tensor.shape[1]

        # Slice the batch tensors to the specified number of rows if num_rows is specified
        if num_rows:
            sliced_input_tensor = input_tensor[:num_rows]
            sliced_target_tensor = target_tensor[:num_rows]
            sliced_attention_mask_tensor = attention_mask_tensor[:num_rows]
        else:
            sliced_input_tensor = input_tensor
            sliced_target_tensor = target_tensor
            sliced_attention_mask_tensor = attention_mask_tensor

        # Visualize input sequences
        print("\nVisualizing Input Tensor:\n\n")
        visualize_sequences(sliced_input_tensor, tokenizer, max_input_sequence_length)
        # Visualize target sequences
        print("\nVisualizing Target Tensor:\n\n")
        visualize_sequences(sliced_target_tensor, tokenizer, max_target_sequence_length)
        # Visualize attention sequences
        print("\nVisualizing Attention Mask Tensor:\n\n")
        visualize_sequences(sliced_attention_mask_tensor, tokenizer, max_target_sequence_length)
    else:
        if tensor_key not in batch_tensor:
            raise KeyError(f"'{tensor_key}' not found in the .npz file")

        tensor = batch_tensor[tensor_key]

        # Get the maximum sequence length
        max_sequence_length = tensor.shape[1]

        # Slice the batch tensor to the specified number of rows if num_rows is specified
        if num_rows:
            sliced_batch_tensor = tensor[:num_rows]
        else:
            sliced_batch_tensor = tensor

        # Visualize sequences
        print(f"Visualizing {tensor_type.capitalize()} Tensor:")
        visualize_sequences(sliced_batch_tensor, tokenizer, max_sequence_length)

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Analyze a .npz batch file.')

    # Define the arguments
    parser.add_argument('--batch_file', type=str, help='Path to the .npz batch file.', required=True)
    parser.add_argument('--tokenizer', type=str, required=False, default=None, help='Name or path of the tokenizer')
    parser.add_argument('--tensor_type', type=str, choices=['input', 'target', 'both'], default='both', help='Type of tensor to analyze. Default: analyze both tensors separately')
    parser.add_argument('--rows', type=int, default=None, help='Number of rows to display (default: all rows)')

    # Parse the arguments
    args = parser.parse_args()

    # Check if the batch file exists
    if not os.path.isfile(args.batch_file):
        print(f"Error: Batch file '{args.batch_file}' does not exist.")
        return

    # Call the batch analysis function with the provided batch file, tokenizer, tensor type, and number of rows
    analyze_batch_file(args.batch_file, args.tokenizer, args.tensor_type, args.rows)

if __name__ == '__main__':
    main()
