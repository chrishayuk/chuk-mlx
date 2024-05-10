import argparse
import numpy as np
from utils.batch_utility import calculate_memory_per_batch, format_memory_size

def main():
    parser = argparse.ArgumentParser(description='Calculate estimated memory usage per batch and across multiple batches, or calculate required batches for a specified number of tokens.')

    # Arguments needed for memory calculation
    parser.add_argument('--max_sequence_length', type=int, default=8192, help='Maximum length of each input sequence. Default: 8192')
    parser.add_argument('--batch_size', type=int, default=1024, help='Number of sequences per batch. Default: 1024')
    parser.add_argument('--dtype', type=str, default='int32', help='Data type (e.g., int32, float32, etc.). Default: int32')
    parser.add_argument('--num_tokens', type=int, default='1000000000000', help='Specify a total number of tokens to calculate required batches.')

    args = parser.parse_args()

    # Convert string data type to NumPy dtype
    try:
        dtype = np.dtype(args.dtype)
    except TypeError:
        raise ValueError(f"Invalid data type: {args.dtype}. Ensure it's a valid NumPy dtype (e.g., 'int32', 'float32').")

    # Calculate memory usage per batch
    memory_per_batch = calculate_memory_per_batch(args.max_sequence_length, args.batch_size, dtype)

    # Calculate how many batches are required to cover the specified number of tokens
    tokens_per_batch = args.max_sequence_length * args.batch_size
    required_batches = (args.num_tokens + tokens_per_batch - 1) // tokens_per_batch  # Ceiling division

    total_memory = memory_per_batch * required_batches
    formatted_total_memory = format_memory_size(total_memory)

    # Print results for the required batches
    print(f"Estimated number of batches required for {args.num_tokens} tokens: {required_batches}")
    print(f"Estimated memory usage for {required_batches} batches: {formatted_total_memory}")

if __name__ == '__main__':
    main()
