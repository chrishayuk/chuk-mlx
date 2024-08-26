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
    elif tensor_type == 'attention_mask':
        tensor_key = 'attention_mask_tensor'
    elif tensor_type == 'both':
        input_tensor_key = 'input_tensor'
        target_tensor_key = 'target_tensor'
        attention_mask_tensor_key = 'attention_mask_tensor'
    else:
        raise ValueError("Invalid value for 'tensor_type'. Use 'input', 'target', 'attention_mask', or 'both'.")

    if tensor_type == 'both':
        input_tensor = batch_tensor.get(input_tensor_key)
        target_tensor = batch_tensor.get(target_tensor_key)
        attention_mask_tensor = batch_tensor.get(attention_mask_tensor_key)

        if input_tensor is None:
            print(f"Warning: '{input_tensor_key}' is missing in the .npz file.")
        if target_tensor is None:
            print(f"Warning: '{target_tensor_key}' is missing in the .npz file.")
        if attention_mask_tensor is None:
            print(f"Warning: '{attention_mask_tensor_key}' is missing in the .npz file.")

        # If any tensors are present, visualize them
        if input_tensor is not None:
            print("\nVisualizing Input Tensor:\n")
            max_input_sequence_length = input_tensor.shape[1]
            visualize_sequences(input_tensor[:num_rows], tokenizer, max_input_sequence_length)
        
        if target_tensor is not None:
            print("\nVisualizing Target Tensor:\n")
            max_target_sequence_length = target_tensor.shape[1]
            visualize_sequences(target_tensor[:num_rows], tokenizer, max_target_sequence_length)

        if attention_mask_tensor is not None:
            print("\nVisualizing Attention Mask Tensor:\n")
            max_attention_mask_sequence_length = attention_mask_tensor.shape[1]
            visualize_sequences(attention_mask_tensor[:num_rows], tokenizer, max_attention_mask_sequence_length)
    else:
        tensor = batch_tensor.get(tensor_key)
        if tensor is None:
            print(f"Warning: '{tensor_key}' is missing in the .npz file.")
            return

        # Get the maximum sequence length
        max_sequence_length = tensor.shape[1]

        # Visualize sequences
        print(f"Visualizing {tensor_type.capitalize()} Tensor:")
        visualize_sequences(tensor[:num_rows], tokenizer, max_sequence_length)

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Analyze a .npz batch file.')

    # Define the arguments
    parser.add_argument('--batch_file', type=str, help='Path to the .npz batch file.', required=True)
    parser.add_argument('--tokenizer', type=str, required=False, default=None, help='Name or path of the tokenizer')
    parser.add_argument('--tensor_type', type=str, choices=['input', 'target', 'attention_mask', 'both'], default='both', help='Type of tensor to analyze. Default: analyze both tensors separately')
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
