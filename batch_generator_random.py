import argparse
import time
import numpy as np
import os
import shutil

def clear_output_directory(output_directory):
    """Clear the output directory."""
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

def generate_npz_batches(output_directory, file_prefix, max_sequence_length=8192, batch_size=1024, num_batches=1, vocab_size=32000):
    """
    Generates multiple .npz files, each representing a batch containing random data simulating input sequences.

    :param output_directory: Directory where batch files will be saved.
    :param file_prefix: Prefix for output batch files.
    :param max_sequence_length: Maximum length of each sequence.
    :param batch_size: Number of sequences per batch.
    :param num_batches: Total number of batches to generate.
    :param vocab_size: The size of the vocabulary (range of token IDs).
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for batch_idx in range(num_batches):
        # Generate random data for this batch
        input_tensor = np.random.randint(0, vocab_size, (batch_size, max_sequence_length)).astype(np.int32)
        target_tensor = np.random.randint(0, vocab_size, (batch_size, max_sequence_length)).astype(np.int32)
        lengths = np.random.randint(1, max_sequence_length + 1, (batch_size,))

        # Create the batch file name with the specified prefix and save the data
        file_path = os.path.join(output_directory, f'{file_prefix}_batch_{batch_idx + 1:04d}.npz')
        np.savez(file_path, input_tensor=input_tensor, target_tensor=target_tensor, lengths=lengths)

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == num_batches:
            print(f"Generated {file_path} with {batch_size} sequences of length {max_sequence_length} using vocab size {vocab_size}")

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Generate .npz batch files simulating tokenized input sequences.')

    # Define the arguments with some defaults
    parser.add_argument('--output_directory', type=str, default='./output', help='Output directory to store the generated .npz batch files. Default: ./output')
    parser.add_argument('--file_prefix', type=str, default='chunk_random', help='Prefix for the output batch files. Default: chunk_random')
    parser.add_argument('--max_sequence_length', type=int, default=8192, help='Maximum length of each input sequence. Default: 8192')
    parser.add_argument('--batch_size', type=int, default=1024, help='Number of sequences per batch. Default: 1024')
    parser.add_argument('--num_batches', type=int, default=1, help='Total number of batches to generate. Default: 1')
    parser.add_argument('--vocab_size', type=int, default=32000, help='Size of the vocabulary (range of token IDs). Default: 32000')

    # Parse the arguments
    args = parser.parse_args()

    # Clear the output directory
    clear_output_directory(args.output_directory)

    # Record the start time
    start_time = time.time()

    # Call the batch generation function with parsed arguments
    generate_npz_batches(args.output_directory, args.file_prefix, args.max_sequence_length, args.batch_size, args.num_batches, args.vocab_size)

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Calculate tokens per batch and total tokens generated
    tokens_per_batch = args.max_sequence_length * args.batch_size
    total_tokens_generated = tokens_per_batch * args.num_batches

    # Calculate batches generated per second and tokens generated per second
    batches_per_second = args.num_batches / elapsed_time
    tokens_per_second = total_tokens_generated / elapsed_time

    # Print the summary table
    print("\nSummary Table:")
    print("=" * 50)
    print(f"{'Number of batches generated:':<30} {args.num_batches:>20,}")
    print(f"{'Tokens per batch:':<30} {tokens_per_batch:>20,}")
    print(f"{'Total tokens generated:':<30} {total_tokens_generated:>20,}")
    print(f"{'Batches generated per second:':<30} {batches_per_second:>20.2f}")
    print(f"{'Tokens generated per second:':<30} {tokens_per_second:>20,.2f}")
    print(f"{'Start time:':<30} {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)):>20}")
    print(f"{'End time:':<30} {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)):>20}")
    print(f"{'Elapsed time:':<30} {elapsed_time:>20.2f} seconds")
    print("=" * 50)

if __name__ == '__main__':
    main()
