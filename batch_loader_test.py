import argparse
import time
import os
import numpy as np
import mlx.core as mx

def load_batches(output_directory, file_prefix, num_batches):
    """
    Load batches using MLX and measure performance.

    :param output_directory: Directory containing batch files.
    :param file_prefix: Prefix of batch files.
    :param num_batches: Number of batches to load.
    :return: List of loaded MLX arrays.
    """
    loaded_batches = []
    load_times = []

    for batch_idx in range(num_batches):
        file_path = os.path.join(output_directory, f"{file_prefix}_batch_{batch_idx + 1:04d}.npy")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        # Measure loading time
        start_time = time.time()

        # load mlx
        mlx_array = mx.load(file_path)

        # Measure finish time
        end_time = time.time()

        # set the load times and batches
        load_times.append(end_time - start_time)
        loaded_batches.append(mlx_array)

    return loaded_batches, load_times

def measure_memory_usage(arrays):
    """
    Measure the memory usage of loaded MLX arrays.

    :param arrays: List of MLX arrays.
    :return: Total memory usage in bytes.
    """
    total_memory_usage = sum(array.nbytes for array in arrays)
    return total_memory_usage

def format_memory_size(memory_in_bytes):
    """
    Converts a memory size in bytes to a more readable format (KB, MB, GB).

    :param memory_in_bytes: Memory size in bytes.
    :return: Formatted string representing the memory size.
    """
    if memory_in_bytes < 1024:
        return f"{memory_in_bytes} B"
    elif memory_in_bytes < 1024 ** 2:
        return f"{memory_in_bytes / 1024:.2f} KB"
    elif memory_in_bytes < 1024 ** 3:
        return f"{memory_in_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{memory_in_bytes / (1024 ** 3):.2f} GB"

def main():
    parser = argparse.ArgumentParser(description='Benchmark batch loading performance and memory usage.')
    parser.add_argument('--output_directory', type=str, default='./output', help='Directory containing batch files. Default: ./output')
    parser.add_argument('--file_prefix', type=str, default='chuk_random', help='Prefix for batch files. Default: chuk_random')
    parser.add_argument('--num_batches', type=int, default=1, help='Number of batches to load. Default: 1')
    args = parser.parse_args()

    # Load the batches
    batches, load_times = load_batches(args.output_directory, args.file_prefix, args.num_batches)
    if not batches:
        print("No batches were loaded.")
        return

    # Calculate tokens per batch based on the loaded data
    tokens_per_batch = batches[0].shape[0] * batches[0].shape[1]

    # Calculate total memory usage
    total_memory_usage = measure_memory_usage(batches)
    formatted_total_memory = format_memory_size(total_memory_usage)

    # Calculate total loading time
    total_load_time = sum(load_times)

    # Calculate average loading time per batch
    avg_load_time = total_load_time / len(load_times)

    # Calculate loading speed in batches per second and gigabytes per second
    batches_per_second = len(batches) / total_load_time
    gigabytes_per_second = total_memory_usage / (total_load_time * 1024 ** 3)

    # Calculate total tokens and tokens per second
    total_tokens = tokens_per_batch * len(batches)
    tokens_per_second = total_tokens / total_load_time

    # Print the summary table
    print("\nBatch Loading Summary:")
    print("=" * 50)
    print(f"{'Total Batches Loaded:':<35} {len(batches):>15,}")
    print(f"{'Tokens per Batch:':<35} {tokens_per_batch:>15,}")
    print(f"{'Total Tokens:':<35} {total_tokens:>15,}")
    print(f"{'Total Memory Usage:':<35} {formatted_total_memory:>15}")
    print(f"{'Total Loading Time:':<35} {total_load_time:>15.2f} seconds")
    print(f"{'Average Loading Time per Batch:':<35} {avg_load_time:>15.4f} seconds")
    print(f"{'Loading Speed (Batches/second):':<35} {batches_per_second:>15.2f}")
    print(f"{'Loading Speed (Tokens/second):':<35} {tokens_per_second:>15,.2f}")
    print(f"{'Loading Speed (GB/second):':<35} {gigabytes_per_second:>15.2f}")
    print("=" * 50)

if __name__ == '__main__':
    main()
