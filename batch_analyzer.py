import argparse
import numpy as np
import os
from utils.tokenizer_loader import load_tokenizer
from batches.summary_utility import generate_batch_analysis_summary_table

def analyze_batch_file(batch_file, tokenizer_name):
    # check we have a tokenizer
    if tokenizer_name:
        # load the tokenizer
        tokenizer = load_tokenizer(tokenizer_name)

        # TODO: handle this in load tokenizer if a llama based tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        pad_token_id = tokenizer.pad_token_id
    else:
        pad_token_id = 0

    # Load the batch data from the .npy file
    batch_data = np.load(batch_file)

    # Generate the summary table using the utility method
    summary_table = generate_batch_analysis_summary_table(batch_data, batch_file, pad_token_id)

    # Print the summary table
    print(summary_table)

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Analyze a .npy batch file.')

    # Define the arguments
    parser.add_argument('--batch_file', type=str, help='Path to the .npy batch file.')
    parser.add_argument('--tokenizer', type=str, required=False, default=None, help='Name or path of the tokenizer')

    # Parse the arguments
    args = parser.parse_args()

    # Check if the batch file exists
    if not os.path.isfile(args.batch_file):
        print(f"Error: Batch file '{args.batch_file}' does not exist.")
        return

    # Call the batch analysis function with the provided batch file
    analyze_batch_file(args.batch_file, args.tokenizer)

if __name__ == '__main__':
    main()