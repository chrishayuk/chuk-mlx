import argparse
import numpy as np
import os
import sys

# Add the parent directory of the tools directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# imports
from core.utils.tokenizer_loader import load_tokenizer
from core.batch.batch_analysis_summary import generate_batch_analysis_summary_table

def analyze_batch_file(batch_file, tokenizer_name, tensor_type=None):
    # Load the tokenizer
    tokenizer = load_tokenizer(tokenizer_name)
    
    # Load the batch data from the .npz file
    batch_data = np.load(batch_file)
    
    # Analyze the specified tensor type or both
    tensor_types = [tensor_type] if tensor_type else ['input', 'target', 'attention_mask']
    
    # loop through the tensor types
    for tensor_type in tensor_types:
        # set the tensor key
        tensor_key = f"{tensor_type}_tensor"
        if tensor_key in batch_data:
            # get the tensor
            tensor = batch_data[tensor_key]

            # generate a summary table
            summary_table = generate_batch_analysis_summary_table(tensor, batch_file, tokenizer.pad_token_id)

            # print the summary table
            print(f"\nSummary for {tensor_type} tensor:")
            print(summary_table)
        else:
            print(f"No {tensor_type} tensor found in the .npz file")

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Analyze a batch file and print a summary.')
    
    # Set the arguments
    parser.add_argument('--batch_file', type=str, required=True, help='Path to the batch file')
    parser.add_argument('--tokenizer', type=str, required=True, help='Name of the tokenizer to use')
    parser.add_argument('--tensor_type', type=str, choices=['input', 'target'], help='Type of tensor to analyze. Default: analyze both tensors separately')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Analyze the batch file
    analyze_batch_file(args.batch_file, args.tokenizer, args.tensor_type)

if __name__ == '__main__':
    # Call main
    main()
