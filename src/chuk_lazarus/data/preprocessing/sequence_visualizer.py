def visualize_sequences(sequences, tokenizer, max_seq_length, max_columns=8):
    # Ensure max_columns does not exceed max_seq_length
    max_columns = min(max_columns, max_seq_length)

    # Fixed width for each column
    fixed_width = 10  

    # Prepare column headers
    columns = [f"T{i}" for i in range(max_columns - 1)] + ["LT", f"T{max_seq_length - 1}"]
    header = '|'.join(col.center(fixed_width) for col in columns)
    separator = '-' * len(header)

    # Print the headers
    print(header)
    print(separator)

    # Process each sequence in the batch
    for seq in sequences:
        # Decode tokens and gather token IDs
        tokens = [tokenizer.decode([token_id]) for token_id in seq[:max_columns - 1]]
        token_ids = [str(token_id) for token_id in seq[:max_columns - 1]]

        # Find the last non-pad token's index correctly
        last_non_pad_index = max((index for index, token_id in enumerate(seq) if token_id != tokenizer.pad_token_id), default=0)
        last_token = tokenizer.decode([seq[last_non_pad_index]])
        last_token_id = str(seq[last_non_pad_index])
        lt_display = f"T{last_non_pad_index}:({last_token_id})"

        # Handle the token at the maximum sequence length - 1
        max_seq_token_index = max_seq_length - 1
        max_seq_token = tokenizer.decode([seq[max_seq_token_index]])
        max_seq_token_id = str(seq[max_seq_token_index])
        t_max_display = f"({max_seq_token_id})"

        # Truncate tokens if necessary and prepare display lines
        token_line = ''
        id_line = ''
        for i, (token, token_id) in enumerate(zip(tokens, token_ids)):
            truncated_token = (token[:fixed_width-2]+".." if len(token) > fixed_width-2 else token).center(fixed_width)
            formatted_id = f"({token_id})".center(fixed_width)
            token_line += truncated_token + '|'
            id_line += formatted_id + '|'

        # Add the last non-pad token and the final token displays
        token_line += last_token.center(fixed_width) + '|' + max_seq_token.center(fixed_width) + '|'
        id_line += lt_display.center(fixed_width) + '|' + t_max_display.center(fixed_width) + '|'

        # Print the tokens and their IDs, ensuring exact column alignment
        print(token_line[:-1])  # Trim the trailing '|'
        print(id_line[:-1])  # Trim the trailing '|'
        print(separator)
