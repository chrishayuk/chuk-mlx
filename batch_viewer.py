import argparse
import os
import numpy as np
from utils.tokenizer_loader import load_tokenizer
from batches.sequence_visualizer import visualize_sequences

def analyze_batch_file(batch_file, tokenizer_name, num_rows):
    # load the tokenizer
    tokenizer = load_tokenizer(tokenizer_name)

    # Load the batch
    batch_tensor = np.load(batch_file)

    # Get the maximum sequence length
    max_sequence_length = batch_tensor.shape[1]

    # Slice the batch tensor to the specified number of rows
    sliced_batch_tensor = batch_tensor[:num_rows]

    # Visualize input sequences
    visualize_sequences(sliced_batch_tensor, tokenizer, max_sequence_length)

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Analyze a .npy batch file.')

    # Define the arguments
    parser.add_argument('--batch_file', type=str, help='Path to the .npy batch file.')
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