import os
import time
import numpy as np
from batch_generation.pretrain_target_batch_generator import create_target_batch
from batch_generation.text_utils import get_line_text
from batch_generation.batch_generation_summary import generate_batch_generation_summary
from batch_generation.batch_analysis_summary import generate_batch_analysis_summary_table

def save_batch(batch_data, file_path, max_sequence_length, pad_token_id, initial_pad_token_id, eos_token_id):
    processed_batch = []

    # loop through the batches
    for i, seq in enumerate(batch_data):
        # flatten
        flat_seq = [item if isinstance(item, int) else item[0] for item in seq]
        
        # check if we need to truncate
        if len(flat_seq) >= max_sequence_length:
            # If the sequence is too long, truncate and ensure EOS is the last token
            flat_seq = flat_seq[:max_sequence_length - 1] + [eos_token_id]
        else:
            # If the sequence is shorter than max length, add EOS if it's not there and then pad
            if flat_seq[-1] != eos_token_id:
                flat_seq.append(eos_token_id)
            
            # pad
            padding_needed = max_sequence_length - len(flat_seq)

            # flatten
            flat_seq += [pad_token_id] * padding_needed
        
        processed_batch.append(flat_seq)
    
    # create the input tensor
    input_tensor = np.array(processed_batch, dtype=np.int32)

    # create the target tensor
    target_tensor, lengths = create_target_batch(input_tensor, pad_token_id, max_sequence_length)
    
    # save the batch
    np.savez(file_path, input_tensor=input_tensor, target_tensor=target_tensor)

    # return the tensor
    return input_tensor

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

def tokenize_and_batch(input_files, tokenizer, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries):
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
