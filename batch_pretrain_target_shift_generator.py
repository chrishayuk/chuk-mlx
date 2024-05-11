import argparse
import os
import numpy as np

def create_target_batch(input_batch, pad_token_id, max_seq_length):
    target_indices = []
    for seq in input_batch:
        if isinstance(pad_token_id, list):
            target_seq = seq[1:].tolist() + pad_token_id
        else:
            target_seq = seq[1:].tolist() + [pad_token_id]
        
        # Pad or truncate the target sequence to match the input sequence length
        if len(target_seq) < max_seq_length:
            target_seq += [pad_token_id] * (max_seq_length - len(target_seq))
        else:
            target_seq = target_seq[:max_seq_length]
        
        target_indices.append(target_seq)
    
    return np.array(target_indices, dtype=np.int32)

def process_batches(input_directory, batch_prefix, individual_batch=None):
    if individual_batch:
        # Process a single batch
        batch_files = [individual_batch]
    else:
        # Get all batch files with the specified prefix
        batch_files = [f for f in os.listdir(input_directory) if f.startswith(batch_prefix) and f.endswith('.npy')]

    for batch_file in batch_files:
        # Load the input batch
        input_batch_path = os.path.join(input_directory, batch_file)
        input_batch = np.load(input_batch_path)

        # Get the maximum sequence length and pad token ID from the input batch
        max_seq_length = input_batch.shape[1]
        pad_token_id = input_batch[0][-1]

        # Create the target batch
        target_batch = create_target_batch(input_batch, pad_token_id, max_seq_length)

        # Save the target batch with the postfix '_target.npy'
        target_batch_file = batch_file.replace('.npy', '_target.npy')
        target_batch_path = os.path.join(input_directory, target_batch_file)
        np.save(target_batch_path, target_batch)

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