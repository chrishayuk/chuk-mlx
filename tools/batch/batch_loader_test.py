import argparse
import time
import os
import numpy as np

def measure_memory_usage(arrays):
    """
    Measure the memory usage of loaded numpy arrays.

    :param arrays: List of numpy arrays.
    :return: Total memory usage in bytes.
    """
    total_memory_usage = sum(array.nbytes for array in arrays if array is not None)
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

def load_batches(output_directory, file_prefix, num_batches, tensor_type='all'):
    """
    Load batches using numpy and measure performance.

    :param output_directory: Directory containing batch files.
    :param file_prefix: Prefix of batch files.
    :param num_batches: Number of batches to load.
    :param tensor_type: Type of tensor to load, either 'input', 'target', 'attention_mask', or 'all'.
    :return: List of loaded numpy arrays.
    """
    loaded_batches = []
    load_times = []

    for batch_idx in range(num_batches):
        file_path = os.path.join(output_directory, f"{file_prefix}_batch_{batch_idx + 1:04d}.npz")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        # Measure loading time
        start_time = time.time()

        # Load npz file
        batch_data = np.load(file_path)

        batch_content = {}

        if tensor_type == 'input' or tensor_type == 'all':
            if 'input_tensor' in batch_data:
                batch_content['input'] = batch_data['input_tensor']
            else:
                print(f"Warning: No 'input_tensor' found in the .npz file: {file_path}")

        if tensor_type == 'target' or tensor_type == 'all':
            if 'target_tensor' in batch_data:
                batch_content['target'] = batch_data['target_tensor']
            else:
                print(f"Warning: No 'target_tensor' found in the .npz file: {file_path}")
        
        if tensor_type == 'attention_mask' or tensor_type == 'all':
            if 'attention_mask_tensor' in batch_data:
                batch_content['attention_mask'] = batch_data['attention_mask_tensor']
            else:
                print(f"Warning: No 'attention_mask_tensor' found in the .npz file: {file_path}")

        # Only append the batch if any tensor was loaded
        if batch_content:
            loaded_batches.append(batch_content)

        # Measure finish time
        end_time = time.time()

        # Store the load times
        load_times.append(end_time - start_time)

    return loaded_batches, load_times


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='Benchmark batch loading performance and memory usage.')
    parser.add_argument('--output_directory', type=str, default='./output', help='Directory containing batch files. Default: ./output')
    parser.add_argument('--file_prefix', type=str, default='chuk_random', help='Prefix for batch files. Default: chuk_random')
    parser.add_argument('--num_batches', type=int, default=1, help='Number of batches to load. Default: 1')
    parser.add_argument('--tensor_type', type=str, default='all', choices=['input', 'target', 'attention_mask', 'all'], help="Type of tensor to load. Default: 'all'")
    args = parser.parse_args()

    # Load the batches
    batches, load_times = load_batches(args.output_directory, args.file_prefix, args.num_batches, args.tensor_type)
    if not batches:
        print("No batches were loaded.")
        return

    # Initialize tensor counts
    input_count = 0
    target_count = 0
    attention_mask_count = 0

    # Get the first tensor (either input, target, or attention_mask) to calculate tokens per batch
    first_tensor = None
    for tensor_name in ['input', 'target', 'attention_mask']:
        if tensor_name in batches[0]:
            first_tensor = batches[0][tensor_name]
            break

    if first_tensor is None:
        print("No valid tensors found in the loaded batches.")
        return

    tokens_per_batch = first_tensor.shape[0] * first_tensor.shape[1]

    # Calculate total memory usage and count tensors
    total_memory_usage = 0
    for batch in batches:
        if 'input' in batch:
            input_count += 1
            total_memory_usage += batch['input'].nbytes
        if 'target' in batch:
            target_count += 1
            total_memory_usage += batch['target'].nbytes
        if 'attention_mask' in batch:
            attention_mask_count += 1
            total_memory_usage += batch['attention_mask'].nbytes

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

    print(f"\nSummary for {args.tensor_type} tensor(s):")
    print("-" * 50)
    print(f"{'Total Batches Loaded:':<35} {len(batches):>15,}")
    print(f"{'Tokens per Batch:':<35} {tokens_per_batch:>15,}")
    print(f"{'Total Tokens:':<35} {total_tokens:>15,}")
    print(f"{'Total Memory Usage:':<35} {formatted_total_memory:>15}")
    print(f"{'Total Loading Time:':<35} {total_load_time:>15.2f} seconds")
    print(f"{'Average Loading Time per Batch:':<35} {avg_load_time:>15.4f} seconds")
    print(f"{'Loading Speed (Batches/second):':<35} {batches_per_second:>15.2f}")
    print(f"{'Loading Speed (Tokens/second):':<35} {tokens_per_second:>15,.2f}")
    print(f"{'Loading Speed (GB/second):':<35} {gigabytes_per_second:>15.2f}")
    print("-" * 50)
    print(f"{'Input Tensors Loaded:':<35} {input_count:>15}")
    print(f"{'Target Tensors Loaded:':<35} {target_count:>15}")
    print(f"{'Attention Mask Tensors Loaded:':<35} {attention_mask_count:>15}")
    print("-" * 50)


if __name__ == '__main__':
    main()
