import argparse
import os
import numpy as np
from utils.tokenizer_loader import load_tokenizer
from batches.sequence_visualizer import visualize_sequences

def analyze_batch_file(batch_file, tokenizer_name, num_rows):
    # Load the tokenizer
    tokenizer = load_tokenizer(tokenizer_name) if tokenizer_name else None

    # Load the batch
    batch_tensor = np.load(batch_file)

    # Determine the correct tensor key
    if 'input_tensor' in batch_tensor:
        tensor = batch_tensor['input_tensor']
    elif 'target_tensor' in batch_tensor:
        tensor = batch_tensor['target_tensor']
    else:
        raise KeyError("Neither 'input_tensor' nor 'target_tensor' found in the .npz file")

    # Get the maximum sequence length
    max_sequence_length = tensor.shape[1]

    # Slice the batch tensor to the specified number of rows if num_rows is specified
    if num_rows:
        sliced_batch_tensor = tensor[:num_rows]
    else:
        sliced_batch_tensor = tensor

    # Visualize input sequences
    visualize_sequences(sliced_batch_tensor, tokenizer, max_sequence_length)

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Analyze a .npz batch file.')

    # Define the arguments
    parser.add_argument('--batch_file', type=str, help='Path to the .npz batch file.', required=True)
    parser.add_argument('--tokenizer', type=str, required=False, default=None, help='Name or path of the tokenizer')
    parser.add_argument('--rows', type=int, default=None, help='Number of rows to display (default: all rows)')

    # Parse the arguments
    args = parser.parse_args()

    # Check if the batch file exists
    if not os.path.isfile(args.batch_file):
        print(f"Error: Batch file '{args.batch_file}' does not exist.")
        return

    # Call the batch analysis function with the provided batch file and number of rows
    analyze_batch_file(args.batch_file, args.tokenizer, args.rows)

if __name__ == '__main__':
    main()
