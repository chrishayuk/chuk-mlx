import json
import os
import time
import numpy as np
from utils.tokenizer_loader import load_tokenizer
from batch_generation.sequence_utility import SequenceUtility
from batch_generation.batch_generation_summary import generate_batch_generation_summary
from batch_generation.batch_analysis_summary import generate_batch_analysis_summary_table

def get_line_text(line):
    line = line.strip()
    if line.startswith('{'):  # JSONL format
        try:
            data = json.loads(line)
            if 'text' in data:
                return data['text']
            elif 'content' in data:
                return data['content']
            else:
                raise ValueError(f"No 'text' or 'content' field found in JSONL: {line}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSONL line: {line}, error: {e}")
    else:  # Plain text format
        return line

def save_batch(batch_data, file_path, max_sequence_length, pad_token_id, initial_pad_token_id, eos_token_id):
    # get sequence utility
    seq_util = SequenceUtility(max_seq_length=max_sequence_length, padding_value=pad_token_id, initial_pad_token=initial_pad_token_id)
        
    # pad the batch
    padded_batch = seq_util.batch_sequences(batch_data)
    
    # Append the EOS token if needed and pad again if required
    processed_batch = []
    for seq in padded_batch:
        stripped_seq = [token for token in seq if token != eos_token_id and token != 0]
        stripped_seq.append(eos_token_id)
        if len(stripped_seq) < max_sequence_length:
            padded_seq = stripped_seq + [0] * (max_sequence_length - len(stripped_seq))
        else:
            padded_seq = stripped_seq[:max_sequence_length]
        processed_batch.append(padded_seq)
    
    # convert the padded batch to a numpy array
    batch_data = np.array(processed_batch, dtype=np.int32)

    # save the batch to a .npz file
    np.savez(file_path, input_tensor=batch_data)
    
    return batch_data

def process_batch(batch_idx, batch_data, file_path, max_sequence_length, pad_token_id, initial_pad_token_id, eos_token_id, print_summaries):
    # start the batch timer
    batch_start_time = time.time()

    # save the batch
    batch_data = save_batch(batch_data, file_path, max_sequence_length, pad_token_id, initial_pad_token_id, eos_token_id)

    # capture batch end time
    batch_end_time = time.time()

    # generate summaries
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
    
    # initialize batch variables
    batch_idx = 0
    current_batch = []
    total_batches = 0
    
    for input_file in input_files:
        try:
            with open(input_file, 'r') as file:
                for line in file:
                    # get the text for the current line
                    text = get_line_text(line)
                    
                    # tokenize the text
                    tokens = tokenizer.encode(text, max_length=max_sequence_length, truncation=True, add_special_tokens=False)
                    
                    # add the tokens to the current batch
                    current_batch.append(tokens)
                    
                    # check if the current batch is full
                    if len(current_batch) == batch_size:
                        # construct the file path for the batch
                        file_path = os.path.join(output_directory, f'{file_prefix}_batch_{batch_idx + 1:04d}.npz')

                        # process the batch
                        process_batch(batch_idx, current_batch, file_path, max_sequence_length, tokenizer.pad_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id, print_summaries)
                        
                        # reset the current batch
                        current_batch = []
                        batch_idx += 1
                        total_batches += 1
        except (OSError, IOError) as e:
            print(f"Error reading file {input_file}: {e}")
    
    # process any remaining samples in the current batch
    if current_batch:
        file_path = os.path.join(output_directory, f'{file_prefix}_batch_{batch_idx + 1:04d}.npz')
        process_batch(batch_idx, current_batch, file_path, max_sequence_length, tokenizer.pad_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id, print_summaries)
        total_batches += 1

    print(f"Total batches processed: {total_batches}")