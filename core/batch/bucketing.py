def add_to_buckets(buckets, input_tokens, target_tokens):
    """ Add input and target tokens to the appropriate bucket based on the combined sequence length. """
    # Combine lengths of input and target sequences
    input_length = len(input_tokens)
    target_length = len(target_tokens)
    combined_length = input_length + target_length
    
    # Create a new bucket if it doesn't exist
    if combined_length not in buckets:
        buckets[combined_length] = []
    
    # Add the sequence pair to the corresponding bucket
    buckets[combined_length].append((input_tokens, target_tokens))



def split_large_buckets(buckets, max_batch_size):
    """ Split buckets that exceed the maximum batch size into smaller buckets. """
    split_buckets = {}
    
    # Iterate over each bucket
    for seq_length, bucket in buckets.items():
        # If the bucket is larger than max_batch_size, split it
        if len(bucket) > max_batch_size:
            num_splits = (len(bucket) + max_batch_size - 1) // max_batch_size
            for i in range(num_splits):
                split_key = (seq_length, i)  # Create a unique key for the split bucket
                start_idx = i * max_batch_size
                end_idx = min((i + 1) * max_batch_size, len(bucket))
                
                # Assign the split portion to a new bucket
                split_buckets[split_key] = bucket[start_idx:end_idx]
        else:
            # If no split is needed, retain the original bucket
            split_buckets[seq_length] = bucket
    
    return split_buckets


def merge_small_buckets(buckets, max_batch_size):
    """Merge small buckets to create larger batches, ensuring they do not exceed the maximum batch size."""
    merged_buckets = {}
    current_bucket = []
    current_seq_length = None

    for seq_length, bucket in sorted(buckets.items(), key=lambda x: x[0] if isinstance(x[0], tuple) else (x[0], 0)):
        for pair in bucket:
            if len(current_bucket) + 1 > max_batch_size:
                if current_seq_length is not None:
                    merged_buckets[current_seq_length] = current_bucket
                current_bucket = []
                current_seq_length = None

            current_bucket.append(pair)
            if current_seq_length is None:
                current_seq_length = seq_length

    if current_bucket:
        merged_buckets[current_seq_length] = current_bucket

    return merged_buckets

def get_batch_from_buckets(buckets, max_batch_size):
    """Retrieve a batch from buckets, ensuring it meets the maximum batch size."""
    for bucket_key, bucket in list(buckets.items()):
        if len(bucket) >= max_batch_size:
            batch = bucket[:max_batch_size]
            buckets[bucket_key] = bucket[max_batch_size:]
            if not buckets[bucket_key]:
                del buckets[bucket_key]  # Remove the bucket if it's empty
            return batch
        elif len(bucket) > 0:
            batch = bucket[:]
            del buckets[bucket_key]  # Remove the bucket completely since it's all used up
            return batch
    return None


