import mlx.core as mx
import logging

logger = logging.getLogger(__name__)

def pad_sequences(sequences, max_length, padding_value=0):
    padded_sequences = []
    for seq in sequences:
        seq_length = seq.shape[0]
        if seq_length < max_length:
            padding = [padding_value] * (max_length - seq_length)
            padded_seq = mx.array(seq.tolist() + padding)
        else:
            padded_seq = seq[:max_length]
        padded_sequences.append(padded_seq)
    return mx.stack(padded_sequences)

def process_concatenated_batch(concatenated_tensor, lengths, tokenizer, batch_index):
    input_tensors = []
    target_tensors = []

    for i in range(concatenated_tensor.shape[0]):
        seq_length = int(lengths[i].item())
        seq_tensor = concatenated_tensor[i, :seq_length]

        sep_index = -1
        for idx in range(seq_length):
            if int(seq_tensor[idx].item()) == tokenizer.sep_token_id:
                sep_index = idx
                break

        if sep_index == -1:
            logger.warning(f"No separator found in sequence {i} of batch {batch_index}. Skipping this sequence.")
            continue

        input_tensor = seq_tensor[:sep_index + 1]
        target_tensor = seq_tensor[sep_index + 1:seq_length]

        if target_tensor.shape[0] < input_tensor.shape[0]:
            padding_length = input_tensor.shape[0] - target_tensor.shape[0]
            padding = [tokenizer.pad_token_id] * padding_length
            target_tensor = mx.array(target_tensor.tolist() + padding)

        input_tensors.append(input_tensor)
        target_tensors.append(target_tensor)

    return input_tensors, target_tensors

def process_non_concatenated_batch(batch, batch_index, tokenizer):
    input_tensors = []
    target_tensors = []

    for i, tensor in enumerate(batch):
        if len(tensor.shape) == 2:
            input_tensors.append(tensor)
            target_tensors.append(tensor)
        elif len(tensor.shape) == 1:
            input_tensor = tensor.reshape(-1, 1)
            input_tensors.append(input_tensor)
            target_tensors.append(input_tensor)
        else:
            logger.error(f"Skipping tensor in sequence {i} of batch {batch_index}. Expected 1D or 2D tensor, but got: {tensor.shape}")

    return input_tensors, target_tensors

