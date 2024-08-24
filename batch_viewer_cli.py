import argparse
import os
import numpy as np
from core.utils.tokenizer_loader import load_tokenizer

def visualize_single_sequence(input_sequence, target_sequence, tokenizer, row_number=None):
    """
    Visualizes a single sequence for both input and target, showing all tokens in the sequence in a simple grid format.
    """
    input_tokens = [tokenizer.decode([token_id]).strip() for token_id in input_sequence]
    input_token_ids = [str(token_id) for token_id in input_sequence]

    target_tokens = [tokenizer.decode([token_id]).strip() for token_id in target_sequence]
    target_token_ids = [str(token_id) for token_id in target_sequence]

    # Display the row number if provided
    if row_number is not None:
        print(f"\nRow {row_number}:")
    
    # Display input tokens and IDs
    print("Input:")
    print(f"Tokens: {' '.join(input_tokens)}")
    print(f"IDs:    {' '.join(input_token_ids)}\n")
    
    # Display target tokens and IDs
    print("Target:")
    print(f"Tokens: {' '.join(target_tokens)}")
    print(f"IDs:    {' '.join(target_token_ids)}\n")

def analyze_batch(batch_file, tokenizer_name, row_number=None):
    # Load the tokenizer
    tokenizer = load_tokenizer(tokenizer_name) if tokenizer_name else None

    # Load the batch
    batch_tensor = np.load(batch_file)

    if 'input_tensor' not in batch_tensor or 'target_tensor' not in batch_tensor:
        raise KeyError("'input_tensor' or 'target_tensor' not found in the .npz file")

    input_tensor = batch_tensor['input_tensor']
    target_tensor = batch_tensor['target_tensor']

    # If a specific row number is provided, visualize that row only
    if row_number is not None:
        if row_number >= len(input_tensor):
            print(f"Error: Row number {row_number} exceeds the number of rows in the batch.")
            return
        input_sequence = input_tensor[row_number]
        target_sequence = target_tensor[row_number]
        visualize_single_sequence(input_sequence, target_sequence, tokenizer, row_number)
    else:
        # If no row number is provided, visualize all rows
        for i, (input_sequence, target_sequence) in enumerate(zip(input_tensor, target_tensor)):
            visualize_single_sequence(input_sequence, target_sequence, tokenizer, i)

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
