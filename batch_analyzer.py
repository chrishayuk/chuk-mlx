import argparse
import numpy as np
import os

def analyze_batch_file(batch_file):
    # Load the batch data from the .npy file
    batch_data = np.load(batch_file)

    # Get the number of rows and sequence length
    num_rows, max_sequence_length = batch_data.shape

    # Count the number of real tokens and padding tokens in each row
    real_tokens_per_row = np.sum(batch_data != 0, axis=1)
    padding_tokens_per_row = max_sequence_length - real_tokens_per_row

    # Calculate the total number of real tokens and padding tokens in the batch
    total_real_tokens = np.sum(real_tokens_per_row)
    total_padding_tokens = np.sum(padding_tokens_per_row)

    # Calculate the average number of real tokens and padding tokens per row
    avg_real_tokens_per_row = total_real_tokens / num_rows
    avg_padding_tokens_per_row = total_padding_tokens / num_rows

    # Calculate the memory usage for real tokens and padding tokens (assuming 4 bytes per token)
    memory_usage_real_tokens = total_real_tokens * 4
    memory_usage_padding_tokens = total_padding_tokens * 4

    # Print the summary table
    print("\nBatch Analysis Summary:")
    print("=" * 50)
    print(f"{'Batch File:':<35} {batch_file}")
    print(f"{'Number of Rows:':<35} {num_rows:>15,}")
    print(f"{'Number of Tokens:':<35} {num_rows * max_sequence_length:>15,}")
    print(f"{'Max Sequence Length:':<35} {max_sequence_length:>15,}")
    print(f"{'Average Real Tokens per Row:':<35} {avg_real_tokens_per_row:>15.2f}")
    print(f"{'Average Padding Tokens per Row:':<35} {avg_padding_tokens_per_row:>15.2f}")
    print(f"{'Total Real Tokens in Batch:':<35} {total_real_tokens:>15,}")
    print(f"{'Total Padding Tokens in Batch:':<35} {total_padding_tokens:>15,}")
    print(f"{'Memory Usage for Real Tokens:':<35} {memory_usage_real_tokens:>15,} bytes")
    print(f"{'Memory Usage for Padding Tokens:':<35} {memory_usage_padding_tokens:>15,} bytes")
    print("=" * 50)

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Analyze a .npy batch file.')

    # Define the arguments
    parser.add_argument('--batch_file', type=str, help='Path to the .npy batch file.')

    # Parse the arguments
    args = parser.parse_args()

    # Check if the batch file exists
    if not os.path.isfile(args.batch_file):
        print(f"Error: Batch file '{args.batch_file}' does not exist.")
        return

    # Call the batch analysis function with the provided batch file
    analyze_batch_file(args.batch_file)

if __name__ == '__main__':
    main()