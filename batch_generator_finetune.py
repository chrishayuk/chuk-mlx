import argparse
from batches.llama_finetune_batch import LLaMAFineTuneBatch

def main():
    # set argument parser
    parser = argparse.ArgumentParser(description='Tokenize JSONL scripts into batches.')

    # set parameters
    parser.add_argument('--input_files', type=str, nargs='+', required=True, help='Input JSONL files')
    parser.add_argument('--tokenizer', type=str, required=True, help='Name or path of the tokenizer')
    parser.add_argument('--output_directory', type=str, default='./output', help='Output directory for tokenized batches')
    parser.add_argument('--file_prefix', type=str, default='tokenized', help='Prefix for output batch files')
    parser.add_argument('--max_sequence_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of sequences per batch')
    
    # parse arguments
    args = parser.parse_args()
    
    # tokenize and batch
    batcher = LLaMAFineTuneBatch(args.tokenizer, args.output_directory, args.file_prefix, args.max_sequence_length, args.batch_size, False)
    batcher.tokenize_and_batch(args.input_files)

if __name__ == '__main__':
    main()