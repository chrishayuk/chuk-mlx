import argparse
import numpy as np
import os
import sys

# Add the parent directory of the tools directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# imports
from core.batch.calculate_batch_memory import calculate_memory_per_batch, format_memory_size

def main():
    parser = argparse.ArgumentParser(description='Calculate estimated memory usage per batch and across multiple batches.')

    # Arguments needed for memory calculation
    parser.add_argument('--max_sequence_length', type=int, default=8192, help='Maximum length of each input sequence. Default: 8192')
    parser.add_argument('--batch_size', type=int, default=1024, help='Number of sequences per batch. Default: 1024')
    parser.add_argument('--dtype', type=str, default='int32', help='Data type (e.g., int32, float32, etc.). Default: int32')
    parser.add_argument('--num_batches', type=int, default=1, help='Number of batches to calculate total memory usage. Default: 1')

    args = parser.parse_args()

    # Convert string data type to NumPy dtype
    try:
        dtype = np.dtype(args.dtype)
    except TypeError:
        raise ValueError(f"Invalid data type: {args.dtype}. Ensure it's a valid NumPy dtype (e.g., 'int32', 'float32').")

    # Calculate memory usage per batch
    memory_per_batch = calculate_memory_per_batch(args.max_sequence_length, args.batch_size, dtype)
    formatted_memory_per_batch = format_memory_size(memory_per_batch)

    # Calculate total memory usage for all batches
    total_memory = memory_per_batch * args.num_batches
    formatted_total_memory = format_memory_size(total_memory)

    # Print memory usage information
    print(f"Estimated memory usage per batch: {formatted_memory_per_batch}")
    print(f"Estimated memory usage across {args.num_batches} batches: {formatted_total_memory}")

if __name__ == '__main__':
    main()
