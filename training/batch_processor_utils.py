import numpy as np
import mlx.core as mx
import logging

logger = logging.getLogger(__name__)

def pad_sequences(sequences, max_batch_size, max_seq_length, padding_value=0):
    padded_sequences = []
    for seq in sequences:
        seq_length = seq.shape[1]
        batch_size = seq.shape[0]
        
        if seq_length < max_seq_length:
            padding = np.full((batch_size, max_seq_length - seq_length), padding_value)
            padded_seq = np.concatenate((seq, padding), axis=1)
        else:
            padded_seq = seq[:, :max_seq_length]
        
        if batch_size < max_batch_size:
            padding = np.full((max_batch_size - batch_size, max_seq_length), padding_value)
            padded_seq = np.concatenate((padded_seq, padding), axis=0)
        
        padded_sequences.append(padded_seq)
    
    #logger.debug(f"Padded sequences shapes: {[seq.shape for seq in padded_sequences]}")
    return padded_sequences

def process_concatenated_batch(concatenated_tensor, lengths, tokenizer, batch_index):
    logger.debug(f"Processing concatenated batch {batch_index}")
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

        input_tensors.append(input_tensor)
        target_tensors.append(target_tensor)

    #logger.debug(f"Processed input tensors: {input_tensors}")
    #logger.debug(f"Processed target tensors: {target_tensors}")
    return input_tensors, target_tensors

def process_non_concatenated_batch(batch, batch_index, tokenizer):
    logger.debug(f"Processing non-concatenated batch {batch_index}")
    input_tensors = []
    target_tensors = []

    for i, tensor in enumerate(batch):
        if len(tensor.shape) == 2:
            input_tensors.append(tensor)
            target_tensors.append(tensor)
        elif len(tensor.shape) == 1:
            input_tensor = tensor.reshape(1, -1)
            input_tensors.append(input_tensor)
            target_tensors.append(input_tensor)
        else:
            logger.error(f"Skipping tensor in sequence {i} of batch {batch_index}. Expected 1D or 2D tensor, but got: {tensor.shape}")

    #logger.debug(f"Processed input tensors: {input_tensors}")
    #logger.debug(f"Processed target tensors: {target_tensors}")
    return input_tensors, target_tensors
