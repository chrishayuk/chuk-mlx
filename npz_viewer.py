import argparse
import os
import zipfile
import numpy as np
import mlx.core as mx

def check_batch_file(filepath):
    try:
        with zipfile.ZipFile(filepath, 'r') as file:
            print("Batch file is a valid zip file.")
            return True
    except zipfile.BadZipFile:
        print("Batch file is not a valid zip file.")
        return False

def load_with_numpy(filepath):
    try:
        np_data = np.load(filepath)
        print("Successfully loaded with NumPy.")
        return np_data
    except Exception as e:
        print(f"Error loading with NumPy: {str(e)}")
        return None

def load_with_mlx(filepath):
    try:
        mx_data = mx.load(filepath)
        print("Successfully loaded with MLX.")
        return mx_data
    except Exception as e:
        print(f"Error loading with MLX: {str(e)}")
        return None

def print_array_info(array, name, rows):
    print(f"\nKey: {name}")
    print(f"Shape: {array.shape}")
    print(f"Dtype: {array.dtype}")
    if rows is not None:
        print(array[:rows])
    else:
        print(array)

def main():
    parser = argparse.ArgumentParser(description='Analyze a .npz batch file.')
    parser.add_argument('--batch_file', type=str, help='Path to the .npz batch file.', required=True)
    parser.add_argument('--rows', type=int, default=None, help='Number of rows to display (default: all rows)')
    args = parser.parse_args()

    if not os.path.isfile(args.batch_file):
        print(f"Error: Batch file '{args.batch_file}' does not exist.")
        return

    if not check_batch_file(args.batch_file):
        return
    
    print("\nNumPy and MLX versions:")
    print(f"NumPy version: {np.__version__}")
    print(f"MLX version: {mx.__version__}")
    print(f"\nFile size: {os.path.getsize(args.batch_file)} bytes")

    print("\nAttempting to load with NumPy:")
    np_data = load_with_numpy(args.batch_file)
    
    print("\nAttempting to load with MLX:")
    mx_data = load_with_mlx(args.batch_file)
    
    if np_data is not None:
        print(f"\nContents of '{args.batch_file}' (NumPy):")
        for key, array in np_data.items():
            print_array_info(array, key, args.rows)
    
    if mx_data is not None:
        print(f"\nContents of '{args.batch_file}' (MLX):")
        for key, array in mx_data.items():
            print_array_info(array, key, args.rows)

if __name__ == '__main__':
    main()