import pytest
from core.batch.bucketing import add_to_buckets, split_large_buckets, merge_small_buckets, get_batch_from_buckets

def test_add_to_buckets():
    buckets = {}
    input_tokens = [1, 2, 3, 4]
    target_tokens = [5, 6, 7, 8]
    
    add_to_buckets(buckets, input_tokens, target_tokens)
    
    combined_length = len(input_tokens) + len(target_tokens)
    
    assert len(buckets) == 1, "Bucket was not created."
    assert combined_length in buckets, "Bucket with the correct combined length key was not created."
    assert buckets[combined_length] == [(input_tokens, target_tokens)], "Tokens were not added to the correct bucket."

def test_add_to_buckets_varied_lengths():
    buckets = {}
    input_tokens_short = [1, 2]
    target_tokens_long = [5, 6, 7, 8]
    
    add_to_buckets(buckets, input_tokens_short, target_tokens_long)
    
    combined_length = len(input_tokens_short) + len(target_tokens_long)
    
    assert len(buckets) == 1, "Bucket was not created for varied length sequences."
    assert combined_length in buckets, "Bucket with the correct combined length key was not created for varied length sequences."
    assert buckets[combined_length] == [(input_tokens_short, target_tokens_long)], "Tokens were not added to the correct bucket for varied length sequences."

def test_split_large_buckets():
    buckets = {8: [([1, 2, 3, 4], [5, 6, 7, 8])] * 10}
    max_batch_size = 3
    
    split_buckets = split_large_buckets(buckets, max_batch_size)
    
    assert len(split_buckets) == 4, "Buckets were not split correctly."
    assert all(len(bucket) <= max_batch_size for bucket in split_buckets.values()), "A split bucket exceeds max_batch_size."

def test_merge_small_buckets():
    buckets = {
        6: [([1, 2, 3], [4, 5, 6])] * 2,
        8: [([1, 2, 3, 4], [5, 6, 7, 8])] * 2,
    }
    max_batch_size = 5
    
    merged_buckets = merge_small_buckets(buckets, max_batch_size)
    
    assert len(merged_buckets) == 1, "Buckets were not merged correctly."
    assert len(merged_buckets[6]) == 4, "The merged bucket does not contain all sequences."

def test_get_batch_from_buckets():
    buckets = {
        8: [([1, 2, 3, 4], [5, 6, 7, 8])] * 5,
        6: [([1, 2, 3], [4, 5, 6])] * 2,
    }
    max_batch_size = 3
    
    batch = get_batch_from_buckets(buckets, max_batch_size)
    
    assert batch is not None, "No batch was returned."
    assert len(batch) == max_batch_size, "The returned batch does not have the correct size."
    
    # Check if the bucket has been updated correctly
    assert len(buckets[8]) == 2, "The bucket was not updated correctly after batch extraction."

def test_empty_buckets():
    buckets = {}
    max_batch_size = 3
    
    batch = get_batch_from_buckets(buckets, max_batch_size)
    
    assert batch is None, "Batch should be None when there are no sequences."

def test_split_and_merge_combination():
    buckets = {
        6: [([1, 2, 3], [4, 5, 6])] * 10,
    }
    max_batch_size = 4
    
    split_buckets = split_large_buckets(buckets, max_batch_size)
    assert len(split_buckets) == 3, "Buckets were not split correctly."
    
    merged_buckets = merge_small_buckets(split_buckets, max_batch_size)
    assert len(merged_buckets) == 3, "Buckets were not merged correctly."
    assert all(len(bucket) <= max_batch_size for bucket in merged_buckets.values()), "Merged bucket exceeds max_batch_size."

def test_partial_batch_after_merging():
    buckets = {
        6: [([1, 2, 3], [4, 5, 6])] * 2,
        8: [([1, 2, 3, 4], [5, 6, 7, 8])] * 1,
    }
    max_batch_size = 4
    
    merged_buckets = merge_small_buckets(buckets, max_batch_size)
    
    # Try to retrieve a batch
    batch = get_batch_from_buckets(merged_buckets, max_batch_size)
    
    assert batch is not None, "No batch was returned."
    assert len(batch) == 3, "The batch size should match the total number of sequences in the merged buckets."

def test_uneven_split_merge():
    buckets = {
        6: [([1, 2, 3], [4, 5, 6])] * 7,
    }
    max_batch_size = 4
    
    split_buckets = split_large_buckets(buckets, max_batch_size)
    assert len(split_buckets) == 2, "Buckets were not split correctly for uneven numbers."
    
    merged_buckets = merge_small_buckets(split_buckets, max_batch_size)
    assert len(merged_buckets) == 2, "Buckets were not merged correctly for uneven numbers."
    assert all(len(bucket) <= max_batch_size for bucket in merged_buckets.values()), "Merged bucket exceeds max_batch_size."

if __name__ == "__main__":
    pytest.main()
