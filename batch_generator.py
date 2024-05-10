import argparse
import json
import os
import numpy as np
from utils.tokenizer_loader import load_tokenizer
from utils.sequence_utility import SequenceUtility
import time

def tokenize_and_batch(input_files, tokenizer_name, output_directory, file_prefix, max_sequence_length, batch_size):
    # load the tokenizer
    tokenizer = load_tokenizer(tokenizer_name)

    # TODO: handle this in load tokenizer if a llama based tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    pad_token_id = tokenizer.pad_token_id
    
    # create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    batch_idx = 0
    current_batch = []
    total_tokens = 0
    total_batches = 0
    
    for input_file in input_files:
        with open(input_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                text = data['text']
                
                tokens = tokenizer.encode(text, max_length=max_sequence_length, truncation=True, add_special_tokens=False)
                
                current_batch.append(tokens)
                
                # check that we're at the end of the batch
                if len(current_batch) == batch_size:
                    # TODO: refactor this into save batch
                    # get sequence utility
                    seq_util = SequenceUtility(max_seq_length=max_sequence_length, padding_value=pad_token_id)
                    
                    # pad the batch
                    padded_batch = seq_util.batch_sequences(current_batch)
                    
                    # load the padded batch
                    batch_data = np.array(padded_batch, dtype=np.int32)

                    # save the batch
                    file_path = os.path.join(output_directory, f'{file_prefix}_batch_{batch_idx + 1:04d}.npy')
                    np.save(file_path, batch_data)
                    
                    # generate the summary table for the batch
                    batch_start_time = time.time()
                    summary_table = generate_summary_table(file_path, batch_data, pad_token_id)
                    batch_end_time = time.time()
                    batch_generation_time = batch_end_time - batch_start_time
                    batch_tokens_per_second = np.sum(batch_data != pad_token_id) / batch_generation_time
                    
                    print(f"Batch {batch_idx + 1} Summary:")
                    print(summary_table)
                    print(f"Batch Generation Time: {batch_generation_time:.2f} seconds")
                    print(f"Batch Tokens per Second: {batch_tokens_per_second:.2f}")
                    print("=" * 50)
                    
                    # next batch
                    current_batch = []
                    batch_idx += 1
                    total_batches += 1
    
    if current_batch:
        # TODO: refactor this into save batch
        # get sequence utility
        seq_util = SequenceUtility(max_seq_length=max_sequence_length, padding_value=pad_token_id)
        
        # pad the batch
        padded_batch = seq_util.batch_sequences(current_batch)
        
        # save the batch
        batch_data = np.array(padded_batch, dtype=np.int32)
        file_path = os.path.join(output_directory, f'{file_prefix}_batch_{batch_idx + 1:04d}.npy')
        np.save(file_path, batch_data)
        total_batches += 1
        
        # generate the summary table for the last batch
        batch_start_time = time.time()
        summary_table = generate_summary_table(file_path, batch_data, pad_token_id)
        batch_end_time = time.time()
        batch_generation_time = batch_end_time - batch_start_time
        batch_tokens_per_second = np.sum(batch_data != pad_token_id) / batch_generation_time
        
        print(f"Last Batch Summary:")
        print(summary_table)
        print(f"Batch Generation Time: {batch_generation_time:.2f} seconds")
        print(f"Batch Tokens per Second: {batch_tokens_per_second:.2f}")
        print("=" * 50)

def generate_summary_table(file_path, batch_data, pad_token_id):
    if batch_data.ndim == 1:
        batch_data = batch_data.reshape(1, -1)
    
    num_rows, max_seq_length = batch_data.shape
    
    real_tokens_per_row = np.sum(batch_data != pad_token_id, axis=1)
    num_tokens = np.sum(real_tokens_per_row)
    avg_real_tokens_per_row = np.mean(real_tokens_per_row)
    
    padding_tokens_per_row = np.sum(batch_data == pad_token_id, axis=1)
    avg_padding_tokens_per_row = np.mean(padding_tokens_per_row)
    
    total_real_tokens = np.sum(real_tokens_per_row)
    total_padding_tokens = np.sum(padding_tokens_per_row)
    
    memory_usage_real_tokens = total_real_tokens * 4  # Each token is stored as int32 (4 bytes)
    memory_usage_padding_tokens = total_padding_tokens * 4
    
    summary_table = f"""
Batch Analysis Summary:
==================================================
Batch File: {file_path}
--------------------------------------------------
Number of Rows: {num_rows}
Number of Tokens: {num_tokens}
Max Sequence Length: {max_seq_length}
--------------------------------------------------
Average Real Tokens per Row: {avg_real_tokens_per_row:.2f}
Average Padding Tokens per Row: {avg_padding_tokens_per_row:.2f}
--------------------------------------------------
Total Real Tokens in Batch: {total_real_tokens}
Total Padding Tokens in Batch: {total_padding_tokens}
--------------------------------------------------
Memory Usage for Real Tokens: {memory_usage_real_tokens} bytes
Memory Usage for Padding Tokens: {memory_usage_padding_tokens} bytes
==================================================
"""   
    return summary_table

def main():
    parser = argparse.ArgumentParser(description='Tokenize JSONL scripts into batches.')
    parser.add_argument('--input_files', type=str, nargs='+', required=True, help='Input JSONL files')
    parser.add_argument('--tokenizer', type=str, required=True, help='Name or path of the tokenizer')
    parser.add_argument('--output_directory', type=str, default='./output', help='Output directory for tokenized batches')
    parser.add_argument('--file_prefix', type=str, default='tokenized', help='Prefix for output batch files')
    parser.add_argument('--max_sequence_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of sequences per batch')
    
    args = parser.parse_args()
    
    tokenize_and_batch(args.input_files, args.tokenizer, args.output_directory, args.file_prefix,
                       args.max_sequence_length, args.batch_size)

if __name__ == '__main__':
    main()