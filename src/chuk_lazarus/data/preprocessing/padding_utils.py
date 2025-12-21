import numpy as np


def pad_sequences(sequences, pad_value, max_length=None, dtype=np.int32):
    """
    Pads a list of sequences to the length of the longest sequence or a specified max_length.

    Args:
        sequences (list of list): A list of sequences where each sequence is a list of tokens/integers.
        pad_value (int): The value to use for padding the sequences.
        max_length (int): The maximum length to which sequences will be padded. If None, it will pad to the length of the longest sequence.
        dtype (np.dtype): The data type of the output numpy array. Default is np.int32.

    Returns:
        np.ndarray: A numpy array of padded sequences.
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    padded_sequences = []
    for seq in sequences:
        seq = list(seq)  # Ensure seq is a list to handle list operations
        padded_seq = seq + [pad_value] * (max_length - len(seq))
        padded_sequences.append(padded_seq)

    return np.array(padded_sequences, dtype=dtype)
