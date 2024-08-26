import argparse
import os
import sys
import numpy as np

# Add the parent directory of the tools directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# imports
from core.utils.tokenizer_loader import load_tokenizer

def visualize_single_sequence(input_sequence, target_sequence, attention_sequence, tokenizer, row_number=None):
    """
    Visualizes a single sequence for input, target, and attention, showing all tokens in a simple grid format.
    """
    if input_sequence is not None:
        input_tokens = [tokenizer.decode([token_id]).strip() for token_id in input_sequence]
        input_token_ids = [str(token_id) for token_id in input_sequence]
    else:
        input_tokens = []
        input_token_ids = []

    if target_sequence is not None:
        target_tokens = [tokenizer.decode([token_id]).strip() for token_id in target_sequence]
        target_token_ids = [str(token_id) for token_id in target_sequence]
    else:
        target_tokens = []
        target_token_ids = []

    attention_token_ids = [str(token_id) for token_id in attention_sequence] if attention_sequence is not None else []

    # Display the row number if provided
    if row_number is not None:
        print(f"\nRow {row_number}:")
    
    # Display input tokens and IDs if available
    if input_sequence is not None:
        print("Input:")
        print(f"Tokens: {' '.join(input_tokens)}")
        print(f"IDs:    {' '.join(input_token_ids)}\n")
    
    # Display target tokens and IDs if available
    if target_sequence is not None:
        print("Target:")
        print(f"Tokens: {' '.join(target_tokens)}")
        print(f"IDs:    {' '.join(target_token_ids)}\n")

    # Display attention IDs only if attention_sequence is provided
    if attention_sequence is not None:
        print("Attention:")
        print(f"IDs:    {' '.join(attention_token_ids)}\n")

def analyze_batch(batch_file, tokenizer_name, row_number=None):
    # Load the tokenizer
    tokenizer = load_tokenizer(tokenizer_name) if tokenizer_name else None

    # Load the batch
    batch_tensor = np.load(batch_file)

    input_tensor = batch_tensor.get('input_tensor')
    target_tensor = batch_tensor.get('target_tensor')
    attention_mask_tensor = batch_tensor.get('attention_mask_tensor')

    if input_tensor is None:
        print("Warning: 'input_tensor' is missing in the .npz file. Proceeding without it.")
    if target_tensor is None:
        print("Warning: 'target_tensor' is missing in the .npz file. Proceeding without it.")
    if attention_mask_tensor is None:
        print("Warning: 'attention_mask_tensor' is missing in the .npz file. Proceeding without it.")

    # If a specific row number is provided, visualize that row only
    if row_number is not None:
        if input_tensor is not None and row_number >= len(input_tensor):
            print(f"Error: Row number {row_number} exceeds the number of rows in the input tensor.")
            return
        input_sequence = input_tensor[row_number] if input_tensor is not None else None
        target_sequence = target_tensor[row_number] if target_tensor is not None else None
        attention_sequence = attention_mask_tensor[row_number] if attention_mask_tensor is not None else None
        visualize_single_sequence(input_sequence, target_sequence, attention_sequence, tokenizer, row_number)
    else:
        # If no row number is provided, visualize all rows
        num_rows = len(input_tensor) if input_tensor is not None else len(target_tensor)
        for i in range(num_rows):
            input_sequence = input_tensor[i] if input_tensor is not None else None
            target_sequence = target_tensor[i] if target_tensor is not None else None
            attention_sequence = attention_mask_tensor[i] if attention_mask_tensor is not None else None
            visualize_single_sequence(input_sequence, target_sequence, attention_sequence, tokenizer, i)

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Visualize rows from a batch file.')

    # Define the arguments
    parser.add_argument('--batch_file', type=str, required=True, help='Path to the .npz batch file.')
    parser.add_argument('--tokenizer', type=str, required=False, default=None, help='Name or path of the tokenizer')
    parser.add_argument('--row_number', type=int, help='Row number to visualize. If not provided, all rows are displayed.')

    # Parse the arguments
    args = parser.parse_args()

    # Check if the batch file exists
    if not os.path.isfile(args.batch_file):
        print(f"Error: Batch file '{args.batch_file}' does not exist.")
        return

    # Visualize the specified row or all rows
    analyze_batch(args.batch_file, args.tokenizer, args.row_number)

if __name__ == '__main__':
    main()
