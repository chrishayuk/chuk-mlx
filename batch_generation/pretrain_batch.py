import os
import time
import numpy as np
from batch_generation.batch_base import BatchBase
from batch_generation.text_utils import get_line_text
from batch_generation.batch_generation_summary import generate_batch_generation_summary
from batch_generation.batch_analysis_summary import generate_batch_analysis_summary_table

class PretrainBatchGenerator(BatchBase):
    def __init__(self, tokenizer, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries):
        # Call the parent initializer with the tokenizer object directly
        super().__init__(tokenizer, output_directory, file_prefix, max_sequence_length, batch_size, print_summaries)

    def tokenize_line(self, line):
        # Tokenize the line similar to how it was done in the pretrain script
        text = get_line_text(line)

        # Encode to tokens using the passed tokenizer object
        tokens = self.tokenizer.encode(text, max_length=self.max_sequence_length, truncation=True, add_special_tokens=False)

        # Return tokens
        return tokens

    def save_batch(self, batch_data, file_path):
        # Empty process list
        processed_batch = []

        # Loop through the batches
        for seq in batch_data:
            # Flatten and truncate/pad sequences as needed
            flat_seq = [item if isinstance(item, int) else item[0] for item in seq]

            # Check if we're larger than the sequence
            if len(flat_seq) >= self.max_sequence_length:
                # Flatten and truncate last token
                flat_seq = flat_seq[:self.max_sequence_length - 1] + [self.tokenizer.eos_token_id]
            else:
                if flat_seq[-1] != self.tokenizer.eos_token_id:
                    flat_seq.append(self.tokenizer.eos_token_id)
                padding_needed = self.max_sequence_length - len(flat_seq)
                flat_seq += [self.tokenizer.pad_token_id] * padding_needed
            
            # Add the processed batch
            processed_batch.append(flat_seq)
        
        # Load the input tensor
        input_tensor = np.array(processed_batch, dtype=np.int32)

        # Create the target batch
        target_tensor, lengths = self.create_target_batch(input_tensor, self.tokenizer.pad_token_id, self.max_sequence_length)
        
        # Save the batch as a .npz file
        np.savez(file_path, input_tensor=input_tensor, target_tensor=target_tensor)

        # Return the input tensor
        return input_tensor

    def process_batch(self, batch_idx, batch_data, file_path):
        # Start the batch timer
        batch_start_time = time.time()

        # Save the batch
        batch_data = self.save_batch(batch_data, file_path)

        # Capture batch end time
        batch_end_time = time.time()

        # Generate summaries
        summary_table = generate_batch_analysis_summary_table(batch_data, file_path, self.tokenizer.pad_token_id)
        generation_stats = generate_batch_generation_summary(batch_idx, batch_data, batch_start_time, batch_end_time, self.tokenizer.pad_token_id)
        
        if self.print_summaries:
            print(f"Batch {batch_idx + 1} Summary:")
            print(generation_stats)
            print(summary_table)

    def tokenize_and_batch(self, input_files):
        # Create the output directory if it doesn't exist
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        
        # initalize
        batch_idx = 0
        current_batch = []
        
        # loop through the input files
        for input_file in input_files:
            try:
                # open the file
                with open(input_file, 'r') as file:
                    # loop through each line
                    for line in file:
                        # tokenize the line
                        tokens = self.tokenize_line(line)

                        # check if we have tokens
                        if tokens:
                            # add the tokens to the batch
                            current_batch.append(tokens)

                            # check if we're over our batch size
                            if len(current_batch) == self.batch_size:
                                # get the path
                                file_path = os.path.join(self.output_directory, f'{self.file_prefix}_batch_{batch_idx + 1:04d}.npz')
                                
                                # process the batch
                                self.process_batch(batch_idx, current_batch, file_path)

                                # clear the current batch
                                current_batch = []

                                # move to next batch
                                batch_idx += 1
            except (OSError, IOError) as e:
                print(f"Error reading file {input_file}: {e}")
        
        # check if we're the current batch
        if current_batch:
            # set the path
            file_path = os.path.join(self.output_directory, f'{self.file_prefix}_batch_{batch_idx + 1:04d}.npz')

            # process
            self.process_batch(batch_idx, current_batch, file_path)

    def create_target_batch(self, input_batch, pad_token_id, max_seq_length):
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
