import argparse
import os
import shutil
from core.batch.llama_finetune_batch import LLaMAFineTuneBatch
from core.utils.tokenizer_loader import load_tokenizer

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
    parser.add_argument('--tokenizer', type=str, required=True, help='Name or path of the tokenizer')
    parser.add_argument('--output_directory', type=str, help='Output directory for tokenized batches (overrides default if provided)')
    parser.add_argument('--file_prefix', type=str, default='tokenized', help='Prefix for output batch files')
    parser.add_argument('--max_sequence_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of sequences per batch')
    
    # Parse arguments
    args = parser.parse_args()

    # Determine the batch output directory
    if args.output_directory:
        batch_output_dir = args.output_directory  # Use the user-provided directory
    else:
        batch_output_dir = f'./output/batches/{args.tokenizer}'  # Default directory structure

    # Clear the output directory if necessary
    clear_output_directory(batch_output_dir)

    # Load the tokenizer object
    tokenizer = load_tokenizer(args.tokenizer)
    
    # Setup the Batcher
    batcher = LLaMAFineTuneBatch(
        tokenizer=tokenizer,
        output_directory=batch_output_dir,
        file_prefix=args.file_prefix,
        max_sequence_length=args.max_sequence_length,
        batch_size=args.batch_size,
        print_summaries=False
    )

    # Tokenize and batch
    batcher.tokenize_and_batch(args.input_files)

if __name__ == '__main__':
    main()
