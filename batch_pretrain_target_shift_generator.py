import argparse
import os
import numpy as np
from batch_generation.pretrain_target_batch_generator import create_target_batch

def process_batches(input_directory, batch_prefix, individual_batch=None):
    if individual_batch:
        # Process a single batch if an individual batch is specified
        batch_files = [individual_batch]
    else:
        # Get all batch files with the specified prefix and .npz extension
        batch_files = [
            f for f in os.listdir(input_directory)
            if f.startswith(batch_prefix) and f.endswith('.npz')
        ]

    for batch_file in batch_files:
        # Load the input batch file
        input_batch_path = os.path.join(input_directory, batch_file)
        input_batch_data = np.load(input_batch_path)
        input_tensor = input_batch_data['input_tensor']
        lengths = input_batch_data.get('lengths')  # Get lengths if available, otherwise None

        # Determine the maximum sequence length and pad token ID
        max_seq_length = input_tensor.shape[1]
        pad_token_id = input_tensor[0][-1]

        # Create the target batch using the input batch
        target_tensor, target_lengths = create_target_batch(input_tensor, pad_token_id, max_seq_length)

        # If lengths are not provided in the input batch, use lengths derived from the target sequences
        if lengths is None:
            lengths = target_lengths

        # Save the target batch to the same file (replace the existing one) if it exists
        target_batch_file = batch_file
        target_batch_path = os.path.join(input_directory, target_batch_file)
        np.savez(target_batch_path, input_tensor=input_tensor, target_tensor=target_tensor, lengths=lengths)

        print(f"Replaced target batch: {target_batch_file}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Create target batches by shifting input batches.')

    # Define required and optional arguments
    parser.add_argument('--input_directory', type=str, required=True, help='Directory containing the input batch files')
    parser.add_argument('--batch_prefix', type=str, required=True, help='Prefix of the batch files')
    parser.add_argument('--individual_batch', type=str, help='Name of an individual batch file to process')

    # Parse the arguments
    args = parser.parse_args()

    # Process the batches based on the provided arguments
    process_batches(args.input_directory, args.batch_prefix, args.individual_batch)

if __name__ == '__main__':
    # Execute the main function
    main()
