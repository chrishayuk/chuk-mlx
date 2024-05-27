import argparse
import os
import numpy as np
from batches.pretrain_target_batch_generator import create_target_batch

def process_batches(input_directory, batch_prefix, individual_batch=None):
    if individual_batch:
        # Process a single batch
        batch_files = [individual_batch]
    else:
        # Get all batch files with the specified prefix
        batch_files = [f for f in os.listdir(input_directory) if f.startswith(batch_prefix) and f.endswith('.npz') and not f.endswith('_target.npz')]

    for batch_file in batch_files:
        # Load the input batch
        input_batch_path = os.path.join(input_directory, batch_file)
        input_batch_data = np.load(input_batch_path)
        input_batch = input_batch_data['input_tensor']

        # Get the maximum sequence length and pad token ID from the input batch
        max_seq_length = input_batch.shape[1]
        pad_token_id = input_batch[0][-1]

        # Create the target batch
        target_batch = create_target_batch(input_batch, pad_token_id, max_seq_length)

        # Save the target batch with the postfix '_target.npz'
        target_batch_file = batch_file.replace('.npz', '_target.npz')
        target_batch_path = os.path.join(input_directory, target_batch_file)
        np.savez(target_batch_path, target_tensor=target_batch)

        print(f"Created target batch: {target_batch_file}")
        
def main():
    # setup arg parser
    parser = argparse.ArgumentParser(description='Create target batches by shifting input batches.')

    # set the arguments
    parser.add_argument('--input_directory', type=str, required=True, help='Directory containing the input batch files')
    parser.add_argument('--batch_prefix', type=str, required=True, help='Prefix of the batch files')
    parser.add_argument('--individual_batch', type=str, help='Name of an individual batch file to process')

    # parse the arguments
    args = parser.parse_args()

    # process the batches
    process_batches(args.input_directory, args.batch_prefix, args.individual_batch)

if __name__ == '__main__':
    # call main
    main()
