import numpy as np

def generate_header():
    # prints the header
    return "\nBatch Analysis Summary:\n" + "=" * 50 + "\n"

def generate_footer():
    # returns the footer
    return "=" * 50 + "\n"

def generate_row(label, value):
    # generates the row
    return f"{label:<35} {value:>15}\n"

def generate_separator():
    # sticks in a - as a seperator
    return "-" * 50 + "\n"

def generate_batch_analysis_summary_table(batch_data, batch_file, pad_token_id):
    # Get the number of rows and sequence length
    num_rows, max_sequence_length = batch_data.shape

    # Count the number of real tokens and padding tokens in each row
    real_tokens_per_row = np.sum((batch_data != pad_token_id) & (batch_data != 0), axis=1)
    padding_tokens_per_row = max_sequence_length - real_tokens_per_row

    # Calculate the total number of real tokens and padding tokens in the batch
    total_real_tokens = np.sum(real_tokens_per_row)
    total_padding_tokens = np.sum(padding_tokens_per_row)

    # Calculate the average number of real tokens and padding tokens per row
    avg_real_tokens_per_row = total_real_tokens / num_rows
    avg_padding_tokens_per_row = total_padding_tokens / num_rows

    # Calculate the memory usage for real tokens and padding tokens (assuming 4 bytes per token)
    memory_usage_real_tokens = total_real_tokens * 4
    memory_usage_padding_tokens = total_padding_tokens * 4

    # Generate the summary table using the utility functions
    summary_table = generate_header()
    summary_table += generate_row("Batch File", batch_file)
    summary_table += generate_separator()
    summary_table += generate_row("Number of Rows", f"{num_rows:,}")
    summary_table += generate_row("Number of Tokens", f"{total_real_tokens + total_padding_tokens:,}")
    summary_table += generate_row("Max Sequence Length", f"{max_sequence_length:,}")
    summary_table += generate_separator()
    summary_table += generate_row("Average Real Tokens per Row", f"{avg_real_tokens_per_row:,.2f}")
    summary_table += generate_row("Average Padding Tokens per Row", f"{avg_padding_tokens_per_row:,.2f}")
    summary_table += generate_separator()
    summary_table += generate_row("Total Real Tokens in Batch", f"{total_real_tokens:,}")
    summary_table += generate_row("Total Padding Tokens in Batch", f"{total_padding_tokens:,}")
    summary_table += generate_separator()
    summary_table += generate_row("Memory Usage for Real Tokens", f"{memory_usage_real_tokens:,} bytes")
    summary_table += generate_row("Memory Usage for Padding Tokens", f"{memory_usage_padding_tokens:,} bytes")
    summary_table += generate_footer()

    # return the table
    return summary_table