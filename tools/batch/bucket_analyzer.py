import argparse
import os
import numpy as np
import pandas as pd

# Add the parent directory of the tools directory to sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# imports
from core.utils.tokenizer_loader import load_tokenizer

def analyze_bucket_directory(directory, tokenizer_name, chunk_size=200):
    # Load the tokenizer
    tokenizer = load_tokenizer(tokenizer_name)
    pad_token_id = tokenizer.pad_token_id
    
    # Initialize a list to hold summary data for each batch file
    summaries = []
    
    # Iterate over all .npz files in the directory
    for batch_file in sorted(os.listdir(directory)):
        if batch_file.endswith(".npz"):
            batch_path = os.path.join(directory, batch_file)
            
            # Load the batch data from the .npz file
            batch_data = np.load(batch_path)
            
            # Extract input and target tensors if they exist
            input_tensor = batch_data.get('input_tensor')
            target_tensor = batch_data.get('target_tensor')
            
            # Prepare data for summary
            for tensor_type, tensor in [('input', input_tensor), ('target', target_tensor)]:
                if tensor is not None:
                    num_sequences = tensor.shape[0]
                    max_sequence_length = tensor.shape[1]
                    
                    # Calculate real and padding tokens
                    total_real_tokens = np.sum((tensor != pad_token_id) & (tensor != 0))
                    total_padding_tokens = np.sum((tensor == pad_token_id) | (tensor == 0))
                    avg_real_tokens_per_row = total_real_tokens / num_sequences
                    avg_padding_tokens_per_row = total_padding_tokens / num_sequences
                    
                    # Append the summary data to the list
                    summaries.append({
                        'Filename': batch_file,
                        'Tensor Type': tensor_type,
                        'Batch Size': num_sequences,
                        'Length': max_sequence_length,
                        'Avg Real': f"{avg_real_tokens_per_row:.2f}",
                        'Avg Padding': f"{avg_padding_tokens_per_row:.2f}",
                        'Total Real': f"{total_real_tokens:,}",
                        'Total Padding': f"{total_padding_tokens:,}"
                    })
    
    # Convert the summaries list to a DataFrame for easier manipulation
    summary_df = pd.DataFrame(summaries)
    
    # Adjust display settings
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    
    # Calculate total number of rows
    total_rows = len(summary_df)
    
    if total_rows == 0:
        print("No batch files found in the specified directory.")
        return
    
    # Iterate over DataFrame in chunks
    for start_row in range(0, total_rows, chunk_size):
        end_row = min(start_row + chunk_size, total_rows)
        chunk_df = summary_df.iloc[start_row:end_row]
        
        # Print chunk
        print(chunk_df.to_string(index=False, justify='center'))
        
        # Check if there are more rows to display
        if end_row < total_rows:
            input("\nPress Enter to continue...\n")
    
    print("\nAnalysis complete.")

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Analyze all batch files in a directory and print a summary.')
    
    # Set the arguments
    parser.add_argument('--directory', type=str, required=True, help='Path to the directory containing batch files')
    parser.add_argument('--tokenizer', type=str, required=True, help='Name of the tokenizer to use')
    parser.add_argument('--chunk_size', type=int, default=200, help='Number of rows to display per page')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Analyze the batch files in the directory
    analyze_bucket_directory(args.directory, args.tokenizer, args.chunk_size)

if __name__ == '__main__':
    # Call main
    main()
