import argparse
import os
import shutil

import numpy as np
from core.utils.tokenizer_loader import load_tokenizer
from core.batch.pretrain_batch import PretrainBatchGenerator

def clear_output_directory(output_directory):
    """Clear the output directory."""
    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

def parse_dtype(dtype_str):
    """Convert a string representation of a dtype to a numpy dtype."""
    try:
        return getattr(np, dtype_str)
    except AttributeError:
        raise ValueError(f"Invalid dtype: {dtype_str}. Please choose a valid numpy dtype.")

def main():
    # Set argument parser
    parser = argparse.ArgumentParser(description='Tokenize JSONL scripts into batches.')

    # Set parameters
    parser.add_argument('--input_files', type=str, nargs='+', required=True, help='Input JSONL files')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to the tokenizer')
    parser.add_argument('--output_directory', type=str, help='Output directory for tokenized batches (overrides default if provided)')
    parser.add_argument('--file_prefix', type=str, default='tokenized', help='Prefix for output batch files')
    parser.add_argument('--max_sequence_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of sequences per batch')
    parser.add_argument('--print_summaries', action='store_true', help='Print summaries for each batch')
    parser.add_argument('--regenerate_batches', action='store_true', help='Regenerate the batches before training.')
    parser.add_argument('--dtype', type=str, default='int32', help='Data type for the output arrays (e.g., int32, float32)')

    # Parse arguments
    args = parser.parse_args()

    # Convert dtype argument to a numpy dtype
    dtype = parse_dtype(args.dtype)

    # Construct the batch output directory using the model name
    if args.output_directory:
        batch_output_dir = args.output_directory  # Use the user-provided directory
    else:
        batch_output_dir = f'./output/batches/{args.tokenizer}'

    # Clear the output directory if regenerate_batches is True or directory doesn't exist
    if args.regenerate_batches or not os.path.exists(batch_output_dir) or len(os.listdir(batch_output_dir)) == 0:
        print("Clearing the output directory...")
        clear_output_directory(batch_output_dir)
    else:
        print("Batch files found. Skipping batch generation...")

    # Load the tokenizer object
    tokenizer = load_tokenizer(args.tokenizer)

    # Initialize PretrainBatchGenerator with the provided arguments
    batch_generator = PretrainBatchGenerator(
        tokenizer=tokenizer,
        output_directory=batch_output_dir,
        file_prefix=args.file_prefix,
        max_sequence_length=args.max_sequence_length,
        batch_size=args.batch_size,
        print_summaries=args.print_summaries,
        dtype=dtype  # Use the dtype from CLI
    )

    # Check if batches exist, if not or if regenerate_batches is True, generate them
    if args.regenerate_batches or not os.path.exists(batch_output_dir) or len(os.listdir(batch_output_dir)) == 0:
        print("Generating batches...")
        batch_generator.tokenize_and_batch(args.input_files)
    else:
        print("Batch files found. Skipping batch generation...")

if __name__ == '__main__':
    main()
