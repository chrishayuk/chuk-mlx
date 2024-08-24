import numpy as np
from datetime import datetime

def generate_header():
    return "\nBatch Generation Summary:\n" + "=" * 50 + "\n"

def generate_footer():
    return "=" * 50 + "\n"

def generate_row(label, value):
    return f"{label:<35} {value:>15}\n"

def generate_summary_separator():
    return "-" * 50 + "\n"

def generate_batch_generation_summary(batch_idx, batch_data, batch_start_time, batch_end_time, pad_token_id):
    # calculate the batch generation time and tokens per second
    batch_generation_time = batch_end_time - batch_start_time
    batch_tokens_per_second = np.sum(batch_data != pad_token_id) / batch_generation_time
    
    # format the start and end times with millisecond precision
    start_time_str = datetime.fromtimestamp(batch_start_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    end_time_str = datetime.fromtimestamp(batch_end_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    
    # calculate the table
    stats = generate_header()
    stats += generate_row("Batch Index", f"{batch_idx:,}")
    stats += generate_row("Batch Start Time", start_time_str)
    stats += generate_row("Batch End Time", end_time_str)
    stats += generate_row("Batch Generation Time", f"{batch_generation_time:.4f} seconds")
    stats += generate_row("Batch Tokens per Second", f"{batch_tokens_per_second:.2f}")
    stats += generate_footer()
    
    # return the table
    return stats