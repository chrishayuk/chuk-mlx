import argparse
import numpy as np
from utils.tokenizer_loader import load_tokenizer
from batches.batch_analysis_summary import generate_batch_analysis_summary_table

def analyze_batch_file(batch_file, tokenizer_name):
    # Load the tokenizer
    tokenizer = load_tokenizer(tokenizer_name)
    
    # Load the batch data from the .npz file
    batch_data = np.load(batch_file)
    
    # Determine the correct tensor key
    if 'input_tensor' in batch_data:
        tensor = batch_data['input_tensor']
    elif 'target_tensor' in batch_data:
        tensor = batch_data['target_tensor']
    else:
        raise KeyError("Neither 'input_tensor' nor 'target_tensor' found in the .npz file")
    
    # Generate summary table
    summary_table = generate_batch_analysis_summary_table(tensor, batch_file, tokenizer.pad_token_id)
    
    # Print the summary table
    print(summary_table)

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Analyze a batch file and print a summary.')
    
    # Set the arguments
    parser.add_argument('--batch_file', type=str, required=True, help='Path to the batch file')
    parser.add_argument('--tokenizer', type=str, required=True, help='Name of the tokenizer to use')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Analyze the batch file
    analyze_batch_file(args.batch_file, args.tokenizer)

if __name__ == '__main__':
    # Call main
    main()
