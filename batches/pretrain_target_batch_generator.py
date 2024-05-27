import numpy as np

def create_target_batch(input_batch, pad_token_id, max_seq_length):
    target_indices = []
    lengths = []
    for seq in input_batch:
        if isinstance(pad_token_id, list):
            target_seq = seq[1:].tolist() + pad_token_id
        else:
            target_seq = seq[1:].tolist() + [pad_token_id]
        
        # Pad or truncate the target sequence to match the input sequence length
        if len(target_seq) < max_seq_length:
            target_seq += [pad_token_id] * (max_seq_length - len(target_seq))
        else:
            target_seq = target_seq[:max_seq_length]
        
        target_indices.append(target_seq)
        lengths.append(len(target_seq))
    
    return np.array(target_indices, dtype=np.int32), np.array(lengths, dtype=np.int32)

