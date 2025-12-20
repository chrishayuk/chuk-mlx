from datasets import load_dataset

# Load the dataset
split = 'train_sft'
dataset = load_dataset('HuggingFaceTB/instruct-data-basics-smollm-H4', split=split)

# Convert to a pandas DataFrame and display the first few rows
import pandas as pd
df = pd.DataFrame(dataset)


# Set pandas to display all rows
pd.set_option('display.max_rows', None)

# Order the DataFrame by the 'instruction' column
df_sorted = df.sort_values(by='instruction')

# Display the DataFrame
print(df_sorted)
