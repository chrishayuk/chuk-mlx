def add_to_buckets(buckets, input_tokens, target_tokens):
    """Add input and target tokens to the appropriate bucket based on sequence length."""
    input_length = len(input_tokens)
    target_length = len(target_tokens)
    seq_length = max(input_length, target_length)
    if seq_length not in buckets:
        buckets[seq_length] = []
    buckets[seq_length].append((input_tokens, target_tokens))

def split_large_buckets(buckets, max_batch_size):
    """Split buckets that exceed the maximum batch size into smaller buckets."""
    split_buckets = {}
    for seq_length, bucket in buckets.items():
        if len(bucket) > max_batch_size:
            num_splits = (len(bucket) + max_batch_size - 1) // max_batch_size
            for i in range(num_splits):
                split_key = (seq_length, i)
                start_idx = i * max_batch_size
                end_idx = min((i + 1) * max_batch_size, len(bucket))
                split_buckets[split_key] = bucket[start_idx:end_idx]
        else:
            split_buckets[seq_length] = bucket
    return split_buckets

def merge_small_buckets(buckets, max_batch_size):
    """Merge small buckets to create larger batches, ensuring they do not exceed the maximum batch size."""
    merged_buckets = {}
    current_bucket = []
    current_seq_length = None

    for seq_length, bucket in sorted(buckets.items(), key=lambda x: x[0] if isinstance(x[0], tuple) else (x[0], 0)):
        if current_seq_length is None:
            current_seq_length = seq_length

        if len(current_bucket) + len(bucket) <= max_batch_size:
            current_bucket.extend(bucket)
        else:
            merged_buckets[current_seq_length] = current_bucket
            current_bucket = bucket
            current_seq_length = seq_length

    if current_bucket:
        merged_buckets[current_seq_length] = current_bucket

    return merged_buckets

def get_batch_from_buckets(buckets, max_batch_size):
    """Retrieve a batch from buckets, ensuring it meets the maximum batch size."""
    for bucket_key, bucket in buckets.items():
        if len(bucket) >= max_batch_size:
            batch = bucket[:max_batch_size]
            buckets[bucket_key] = bucket[max_batch_size:]
            return batch
    return None
