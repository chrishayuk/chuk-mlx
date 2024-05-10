import argparse
import json
import os
import time
import numpy as np
from utils.tokenizer_loader import load_tokenizer
from utils.sequence_utility import SequenceUtility
from batches.summary_utility import generate_batch_analysis_summary_table, generate_batch_generation_summary

def save_batch(batch_data, file_path, max_sequence_length, pad_token_id):
    # get sequence utility
    seq_util = SequenceUtility(max_seq_length=max_sequence_length, padding_value=pad_token_id)
    
    # pad the batch
    padded_batch = seq_util.batch_sequences(batch_data)
    
    # load the padded batch
    batch_data = np.array(padded_batch, dtype=np.int32)

    # save the batch
    np.save(file_path, batch_data)
    
    return batch_data

def process_batch(batch_idx, batch_data, file_path, max_sequence_length, pad_token_id):
        # start the batch timer
        batch_start_time = time.time()

        # save the batch
        batch_data = save_batch(batch_data, file_path, max_sequence_length, pad_token_id)

        # capture batch end time
        batch_end_time = time.time()

        # calculate the batch generation time
        summary_table = generate_batch_analysis_summary_table(batch_data, file_path, pad_token_id)
        generation_stats = generate_batch_generation_summary(batch_idx, batch_data, batch_start_time, batch_end_time, pad_token_id)
        
        # print out the batch summary
        print(f"Batch {batch_idx + 1} Summary:")
        print(generation_stats)
        print(summary_table)
        
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
    total_batches = 0
    
    for input_file in input_files:
        with open(input_file, 'r') as file:
            for line in file:
                data = json.loads(line)
                text = data['text']
                
                tokens = tokenizer.encode(text, max_length=max_sequence_length, truncation=True, add_special_tokens=False)
                
                current_batch.append(tokens)
                
                # check if the current batch is full
                if len(current_batch) == batch_size:
                    # get the file path
                    file_path = os.path.join(output_directory, f'{file_prefix}_batch_{batch_idx + 1:04d}.npy')

                    # process the batch
                    process_batch(batch_idx, current_batch, file_path, max_sequence_length, pad_token_id)
                    
                    # next batch
                    current_batch = []
                    batch_idx += 1
                    total_batches += 1

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