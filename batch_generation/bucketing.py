def add_to_buckets(buckets, tokens):
    seq_length = len(tokens)
    if seq_length not in buckets:
        buckets[seq_length] = []
    buckets[seq_length].append(tokens)

def split_large_buckets(buckets, max_batch_size):
    split_buckets = {}
    for seq_length, bucket in buckets.items():
        if len(bucket) > max_batch_size:
            num_splits = (len(bucket) + max_batch_size - 1) // max_batch_size
            for i in range(num_splits):
                split_key = (seq_length, i)
                split_buckets[split_key] = bucket[i * max_batch_size: (i + 1) * max_batch_size]
        else:
            split_buckets[seq_length] = bucket
    return split_buckets

def merge_small_buckets(buckets, max_batch_size):
    merged_buckets = {}
    sorted_keys = sorted(buckets.keys(), key=lambda x: x[0] if isinstance(x, tuple) else x)

    i = 0
    while i < len(sorted_keys):
        current_bucket = buckets[sorted_keys[i]]
        j = i + 1
        while j < len(sorted_keys) and len(current_bucket) + len(buckets[sorted_keys[j]]) <= max_batch_size:
            current_bucket.extend(buckets[sorted_keys[j]])
            j += 1
        merged_buckets[sorted_keys[i]] = current_bucket
        i = j

    return merged_buckets

def get_batch_from_buckets(buckets, max_batch_size):
    for bucket_key, bucket in buckets.items():
        if len(bucket) >= max_batch_size:
            batch = bucket[:max_batch_size]
            buckets[bucket_key] = bucket[max_batch_size:]
            return batch
    return None