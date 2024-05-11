import json
import os
import time
import numpy as np
from utils.tokenizer_loader import load_tokenizer
from utils.sequence_utility import SequenceUtility
from .batch_generation_summary import generate_batch_generation_summary
from .batch_analysis_summary import generate_batch_analysis_summary_table

def get_line_text(line):
    line = line.strip()
    
    if line.startswith('{'):  # JSONL format
        data = json.loads(line)
        if 'text' in data:
            return data['text']
        elif 'content' in data:
            return data['content']
        else:
            raise ValueError(f"No 'text' or 'content' field found in JSONL: {line}")
    else:  # Plain text format
        return line

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

def process_batch(batch_idx, batch_data, file_path, max_sequence_length, pad_token_id, print_summaries):
        # start the batch timer
        batch_start_time = time.time()

        # save the batch
        batch_data = save_batch(batch_data, file_path, max_sequence_length, pad_token_id)

        # capture batch end time
        batch_end_time = time.time()

        # calculate the batch generation time
        summary_table = generate_batch_analysis_summary_table(batch_data, file_path, pad_token_id)
        generation_stats = generate_batch_generation_summary(batch_idx, batch_data, batch_start_time, batch_end_time, pad_token_id)
        
        if print_summaries:
            # print out the batch summary
            print(f"Batch {batch_idx + 1} Summary:")
            print(generation_stats)
            print(summary_table)
        

def tokenize_and_batch(input_files, tokenizer_name, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries):
    # load the tokenizer
    tokenizer = load_tokenizer(tokenizer_name)
    
    # create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # start at the beginning for the batch
    batch_idx = 0
    current_batch = []
    total_batches = 0
    
    for input_file in input_files:
        with open(input_file, 'r') as file:
            for line in file:
                # get the text for the current line
                text = get_line_text(line)
                
                # tokenize
                tokens = tokenizer.encode(text, max_length=max_sequence_length, truncation=True, add_special_tokens=False)
                
                # add the tokens to the batch
                current_batch.append(tokens)
                
                # check if the current batch is full
                if len(current_batch) == batch_size:
                    # get the file path
                    file_path = os.path.join(output_directory, f'{file_prefix}_batch_{batch_idx + 1:04d}.npy')

                    # process the batch
                    process_batch(batch_idx, current_batch, file_path, max_sequence_length, tokenizer.pad_token_id, print_summaries)
                    
                    # next batch
                    current_batch = []
                    batch_idx += 1
                    total_batches += 1
        
    # check if there are any remaining samples in the current batch
    if current_batch:
        # get the file path for the last batch
        file_path = os.path.join(output_directory, f'{file_prefix}_batch_{batch_idx + 1:04d}.npy')

        # process the last batch
        process_batch(batch_idx, current_batch, file_path, max_sequence_length, tokenizer.pad_token_id, print_summaries)
        total_batches += 1