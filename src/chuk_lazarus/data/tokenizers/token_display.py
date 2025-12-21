from tabulate import tabulate


class TokenDisplayUtility:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def safe_str_conversion(self, obj):
        """Safely convert any object to a string representation."""
        if obj is None:
            return "None"
        elif isinstance(obj, bool):
            return str(obj)
        elif isinstance(obj, (int, float)):
            return str(obj)
        elif isinstance(obj, str):
            return obj
        else:
            return repr(obj)

    def truncate_string(self, string, max_length=30):
        """Truncate a string to a maximum length, adding '...' if truncated."""
        safe_string = self.safe_str_conversion(string)
        return (
            safe_string if len(safe_string) <= max_length else safe_string[: max_length - 3] + "..."
        )

    def display_tokens_from_prompt(self, prompt, add_special_tokens=True):
        # encode the prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)

        # display the tokens
        self.display_tokens(input_ids)

    def display_tokens_from_ids(self, input_ids):
        # display the tokens
        self.display_tokens(input_ids)

    def display_tokens(self, input_ids):
        # create a table of prompts
        table_data = [
            [i, token_id, self.truncate_string(self.tokenizer.decode([token_id]).strip())]
            for i, token_id in enumerate(input_ids)
        ]

        # print the table
        print(
            tabulate(
                table_data,
                headers=["Index", "Token ID", "Decoded Token"],
                tablefmt="grid",
                maxcolwidths=[None, None, 30],
            )
        )

    def display_full_vocabulary(self, chunk_size=1000, pause_between_chunks=False):
        # get the full vocabulary
        vocab = self.tokenizer.get_vocab()

        # get the token IDs and sort them
        token_ids = sorted(vocab.values())

        # display the vocabulary in chunks
        for i in range(0, len(token_ids), chunk_size):
            chunk = token_ids[i : i + chunk_size]

            # create a table of the vocabulary chunk
            table_data = [
                [j + i, token_id, self.truncate_string(self.tokenizer.decode([token_id]).strip())]
                for j, token_id in enumerate(chunk)
            ]

            # print the chunk table
            print(f"\nVocabulary Chunk {i // chunk_size + 1}")
            try:
                print(
                    tabulate(
                        table_data,
                        headers=["Index", "Token ID", "Decoded Token"],
                        tablefmt="grid",
                        maxcolwidths=[None, None, 30],
                    )
                )
            except AttributeError:
                print("Error displaying chunk with tabulate. Using fallback formatting:")
                print(self.manual_format_table(table_data, ["Index", "Token ID", "Decoded Token"]))

            # pause between chunks if enabled
            if pause_between_chunks and i + chunk_size < len(token_ids):
                input("Press Enter to continue to the next chunk...")

    def manual_format_table(self, table_data, headers):
        """Manually format the table data as a string."""
        # Determine column widths
        col_widths = [
            max(len(str(row[i])) for row in table_data + [headers]) for i in range(len(headers))
        ]

        # Format the headers
        header_str = " | ".join(f"{header:<{width}}" for header, width in zip(headers, col_widths))
        separator = "-+-".join("-" * width for width in col_widths)

        # Format the rows
        rows = [
            " | ".join(
                f"{self.safe_str_conversion(cell):<{width}}" for cell, width in zip(row, col_widths)
            )
            for row in table_data
        ]

        # Combine all parts
        return "\n".join([header_str, separator] + rows)
