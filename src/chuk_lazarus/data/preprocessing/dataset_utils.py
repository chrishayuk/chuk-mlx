def tokenize_dataset(input_files, tokenize_line):
    """
    Tokenizes a dataset given a list of input files.

    Args:
        input_files (list of str): List of file paths to tokenize.
        tokenize_line (function): A function that tokenizes a single line of text.

    Returns:
        list: A list of tokenized lines, where each line is a tuple (input_tokens, target_tokens, attention_mask).
    """
    # Empty dataset to store tokenized data
    tokenized_dataset = []

    # Loop through each file in the dataset
    for input_file in input_files:
        # Open the file
        with open(input_file) as file:
            # Loop through each line in the file
            for line in file:
                # Tokenize the line
                tokenized_line = tokenize_line(line)

                # Ensure that we've got input tokens, target tokens, and attention mask
                if len(tokenized_line) == 3:
                    # Add the row to the dataset
                    tokenized_dataset.append(tokenized_line)

    # Return the tokenized dataset
    return tokenized_dataset
