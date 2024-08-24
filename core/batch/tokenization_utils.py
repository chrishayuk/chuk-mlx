def tokenize_and_pad(seq, tokenizer, seq_length):
    if seq_length < 1:
        raise ValueError("sequence_length must be at least 1")
    
    # flatten the sequence
    flat_seq = [item if isinstance(item, int) else item[0] for item in seq]

    # check if truncated
    if len(flat_seq) > seq_length:
        flat_seq = flat_seq[:seq_length]

    # check if padded
    if len(flat_seq) < seq_length:
        padding_needed = seq_length - len(flat_seq)
        flat_seq += [tokenizer.pad_token_id] * padding_needed

    # return the sequence
    return flat_seq


def batch_tokenize_and_pad(batch_data, tokenizer, max_sequence_length):
    # Ensure batch_data is not empty
    if not batch_data:
        raise ValueError("batch_data cannot be empty")

    # If batch_data contains tuples (input, target), extract the input sequences
    if isinstance(batch_data[0], tuple):
        batch_data = [seq[0] for seq in batch_data]

    # Calculate the lengths of the sequences
    sequence_lengths = [len(seq) for seq in batch_data]

    # Find the length of the longest sequence in the batch, but don't exceed max_sequence_length
    max_seq_length = min(max(sequence_lengths), max_sequence_length)

    # Tokenize and pad each input sequence to the max length
    processed_batch = [tokenize_and_pad(seq, tokenizer, max_seq_length) for seq in batch_data]

    # return the batch
    return processed_batch


