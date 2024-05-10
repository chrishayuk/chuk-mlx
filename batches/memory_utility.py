import numpy as np

def calculate_memory_per_batch(max_sequence_length, batch_size, dtype=np.int32):
    """
    Calculates the memory usage for one batch of data.

    :param max_sequence_length: Maximum length of each sequence.
    :param batch_size: Number of sequences per batch.
    :param dtype: Data type of each element, default is np.int32.
    :return: Approximate memory usage per batch in bytes.
    """
    # Size of each element in bytes
    element_size = np.dtype(dtype).itemsize

    # Total number of elements in one batch
    total_elements = max_sequence_length * batch_size

    # Total memory in bytes
    memory_in_bytes = total_elements * element_size

    return memory_in_bytes

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
