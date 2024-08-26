import json
import argparse
from core.utils.tokenizer_loader import load_tokenizer

def preprocess_text_to_jsonl(input_file, output_file, tokenizer_name, max_sequence_length, overlap):
    """
    Preprocesses a large text file into tokenized sequences with overlap and saves them in JSONL format.

    Args:
        input_file (str): Path to the input text file.
        output_file (str): Path to the output JSONL file.
        tokenizer_name (str): The name of the tokenizer to use.
        max_sequence_length (int): Maximum number of tokens per sequence.
        overlap (int): Number of tokens to overlap between sequences.
    """
    # Load the tokenizer
    tokenizer = load_tokenizer(tokenizer_name)
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        # empty current tokens
        current_tokens = []

        # loop through each line
        for line in f_in:
            # tokenize
            tokens = tokenizer.encode(line.strip(), add_special_tokens=False)

            # add to the current tokens
            current_tokens.extend(tokens)

            # loop through until we hit the max sequence length
            while len(current_tokens) >= max_sequence_length:
                # Extract the sequence up to the max_sequence_length
                sequence_tokens = current_tokens[:max_sequence_length]
                
                # Decode the tokens back to text
                sequence_text = tokenizer.decode(sequence_tokens, clean_up_tokenization_spaces=True)
                
                # Create the JSON object with additional attributes
                json_obj = {
                    "text": sequence_text.strip(),
                    "tokens": len(sequence_tokens),
                    "words": len(sequence_text.split()),
                    "characters": len(sequence_text)
                }
                
                # Write the JSON object to the JSONL file
                f_out.write(json.dumps(json_obj) + "\n")
                
                # Apply overlap: move the current_tokens window forward by (max_sequence_length - overlap)
                current_tokens = current_tokens[max_sequence_length - overlap:]

        # Handle any remaining tokens in the last sequence
        if current_tokens:
            sequence_text = tokenizer.decode(current_tokens, clean_up_tokenization_spaces=True)
            json_obj = {
                "text": sequence_text.strip(),
                "tokens": len(current_tokens),
                "words": len(sequence_text.split()),
                "characters": len(sequence_text)
            }
            f_out.write(json.dumps(json_obj) + "\n")

def main():
    # setup the parser
    parser = argparse.ArgumentParser(description="Preprocess a large text file into tokenized sequences with overlap and save as JSONL format.")
    
    # arguments
    parser.add_argument('--input_file', type=str, default="sample_data/datasets/tiny_shakespeare.txt", help="Path to the input text file.")
    parser.add_argument('--output_file', type=str, default="output/datasets/tiny_shakespeare.jsonl", help="Path to the output JSONL file.")
    parser.add_argument('--tokenizer_name', type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="Name of the tokenizer to use.")
    parser.add_argument('--max_sequence_length', type=int, default=4096, help="Maximum number of tokens per sequence.")
    parser.add_argument('--overlap', type=int, default=64, help="Number of tokens to overlap between sequences.")  # Overlap setting in CLI
    
    # parse
    args = parser.parse_args()
    
    # pre-process to jsonl
    preprocess_text_to_jsonl(
        input_file=args.input_file,
        output_file=args.output_file,
        tokenizer_name=args.tokenizer_name,
        max_sequence_length=args.max_sequence_length,
        overlap=args.overlap
    )

if __name__ == "__main__":
    main()
