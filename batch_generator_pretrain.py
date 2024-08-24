import argparse
import os
import shutil
from core.utils.tokenizer_loader import load_tokenizer
from core.batch.pretrain_batch import PretrainBatchGenerator

def clear_output_directory(output_directory):
    """Clear the output directory."""
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

def main():
    # Set argument parser
    parser = argparse.ArgumentParser(description='Tokenize JSONL scripts into batches.')

    # Set parameters
    parser.add_argument('--input_files', type=str, nargs='+', required=True, help='Input JSONL files')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to the tokenizer')
    parser.add_argument('--output_directory', type=str, default='./output', help='Output directory for tokenized batches')
    parser.add_argument('--file_prefix', type=str, default='tokenized', help='Prefix for output batch files')
    parser.add_argument('--max_sequence_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of sequences per batch')
    parser.add_argument('--print_summaries', action='store_true', help='Print summaries for each batch')

    # Parse arguments
    args = parser.parse_args()
    
    # Clear the output directory
    clear_output_directory(args.output_directory)

    # Load the tokenizer object
    tokenizer = load_tokenizer(args.tokenizer)

    # Initialize PretrainBatchGenerator with the provided arguments
    batch_generator = PretrainBatchGenerator(
        tokenizer=tokenizer,
        output_directory=args.output_directory,
        file_prefix=args.file_prefix,
        max_sequence_length=args.max_sequence_length,
        batch_size=args.batch_size,
        print_summaries=args.print_summaries
    )

    # Tokenize and batch
    batch_generator.tokenize_and_batch(args.input_files)

if __name__ == '__main__':
    main()
