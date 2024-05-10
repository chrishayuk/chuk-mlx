import numpy as np
from datetime import datetime

def generate_analysis_summary_header():
    return "\nBatch Analysis Summary:\n" + "=" * 50 + "\n"

def generate_generation_summary_header():
    return "\nBatch Generation Summary:\n" + "=" * 50 + "\n"

def generate_summary_footer():
    return "=" * 50 + "\n"

def generate_summary_row(label, value):
    return f"{label:<35} {value:>15}\n"

def generate_summary_separator():
    return "-" * 50 + "\n"

def generate_batch_analysis_summary_table(batch_data, batch_file, pad_token_id):
    # Get the number of rows and sequence length
    num_rows, max_sequence_length = batch_data.shape

    # Count the number of real tokens and padding tokens in each row
    real_tokens_per_row = np.sum(batch_data != pad_token_id, axis=1)
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
    summary_table = generate_analysis_summary_header()
    summary_table += generate_summary_row("Batch File", batch_file)
    summary_table += generate_summary_separator()
    summary_table += generate_summary_row("Number of Rows", f"{num_rows:,}")
    summary_table += generate_summary_row("Number of Tokens", f"{total_real_tokens + total_padding_tokens:,}")
    summary_table += generate_summary_row("Max Sequence Length", f"{max_sequence_length:,}")
    summary_table += generate_summary_separator()
    summary_table += generate_summary_row("Average Real Tokens per Row", f"{avg_real_tokens_per_row:,.2f}")
    summary_table += generate_summary_row("Average Padding Tokens per Row", f"{avg_padding_tokens_per_row:,.2f}")
    summary_table += generate_summary_separator()
    summary_table += generate_summary_row("Total Real Tokens in Batch", f"{total_real_tokens:,}")
    summary_table += generate_summary_row("Total Padding Tokens in Batch", f"{total_padding_tokens:,}")
    summary_table += generate_summary_separator()
    summary_table += generate_summary_row("Memory Usage for Real Tokens", f"{memory_usage_real_tokens:,} bytes")
    summary_table += generate_summary_row("Memory Usage for Padding Tokens", f"{memory_usage_padding_tokens:,} bytes")
    summary_table += generate_summary_footer()

    return summary_table

def generate_batch_generation_summary(batch_idx, batch_data, batch_start_time, batch_end_time, pad_token_id):
    # calculate the batch generation time and tokens per second
    batch_generation_time = batch_end_time - batch_start_time
    batch_tokens_per_second = np.sum(batch_data != pad_token_id) / batch_generation_time
    
    # format the start and end times with millisecond precision
    start_time_str = datetime.fromtimestamp(batch_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    end_time_str = datetime.fromtimestamp(batch_end_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    
    # calculate the table
    stats = generate_generation_summary_header()
    stats += generate_summary_row("Batch Index", f"{batch_idx:,}")
    stats += generate_summary_row("Batch Start Time", start_time_str)
    stats += generate_summary_row("Batch End Time", end_time_str)
    stats += generate_summary_row("Batch Generation Time", f"{batch_generation_time:.4f} seconds")
    stats += generate_summary_row("Batch Tokens per Second", f"{batch_tokens_per_second:.2f}")
    stats += generate_summary_footer()
    
    # return the table
    return stats