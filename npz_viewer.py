import argparse
import os
import zipfile
import numpy as np

def check_batch_file(filepath):
    try:
        with zipfile.ZipFile(filepath, 'r') as file:
            print("Batch file is valid.")
            return True
    except zipfile.BadZipFile:
        print("Batch file is corrupted or invalid.")
        return False

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Analyze a .npz batch file.')

    # Define the arguments
    parser.add_argument('--batch_file', type=str, help='Path to the .npz batch file.', required=True)
    parser.add_argument('--tokenizer', type=str, required=False, default=None, help='Name or path of the tokenizer')
    parser.add_argument('--rows', type=int, default=None, help='Number of rows to display (default: all rows)')

    # Parse the arguments
    args = parser.parse_args()

    # Check if the batch file exists
    if not os.path.isfile(args.batch_file):
        print(f"Error: Batch file '{args.batch_file}' does not exist.")
        return

    # Validate the batch file
    if not check_batch_file(args.batch_file):
        return

    # Load the npz file
    data = np.load(args.batch_file)

    # Print the contents of the npz file
    print(f"Contents of '{args.batch_file}':")
    for key in data:
        print(f"\nKey: {key}")
        tensor = data[key]
        if args.rows is not None:
            print(tensor[:args.rows])
        else:
            print(tensor)

if __name__ == '__main__':
    main()
