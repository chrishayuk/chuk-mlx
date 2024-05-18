import os
import mlx.core as mx
from .batch_dataset_base import BatchDatasetBase

class DirectoryBatchDataset(BatchDatasetBase):
    def __init__(self, batch_output_dir, batchfile_prefix):
        # call the constructor
        super().__init__()
        
        # set the output directory, and prefix
        self.batch_output_dir = batch_output_dir
        self.batchfile_prefix = batchfile_prefix
        
        # load all the batch files
        self._load_batch_files()
        
        # Set the length of the dataset
        self.length = len(self.batch_files)
        
        # Initialize the current index
        self.current_index = 0

        # Calculate the lengths during initialization
        self.lengths = self._get_lengths()
    
    def _load_batch_files(self):
        # Get a list of all batch files in the directory
        for filename in os.listdir(self.batch_output_dir):
            # check it's a valid filename
            if filename.startswith(self.batchfile_prefix) and filename.endswith(".npy"):
                # ignore target files, so we only process input files
                if "target" not in filename:
                    # set the target filename as the input filename with _target
                    target_filename = filename.replace(".npy", "_target.npy")
                    # add the input and the target to the batch
                    self.batch_files.append((filename, target_filename))
        
        # Sort the batch files
        self.batch_files.sort()
    
    def __iter__(self):
        # Reset the current index
        self.current_index = 0
        return self
    
    def __next__(self):
        # If all batch files have been processed, stop iteration
        if self.current_index >= len(self.batch_files):
            raise StopIteration
        
        # Get the current batch file and target file
        batch_file, target_file = self.batch_files[self.current_index]
        
        # Load the tensors
        input_tensor, target_tensor = self._load_tensors(batch_file, target_file)
        
        # Increment the current index
        self.current_index += 1
        
        # Return the tensors
        return input_tensor, target_tensor, self.lengths
    
    def _load_tensors(self, batch_file, target_file):
        batch_path = os.path.join(self.batch_output_dir, batch_file)
        target_path = os.path.join(self.batch_output_dir, target_file)
        
        # Load the batch and target tensors
        input_tensor = mx.load(batch_path)
        target_tensor = mx.load(target_path)
        
        # return the tensors
        return input_tensor, target_tensor
    
    def _get_lengths(self):
        # Load the first batch file to get the batch size
        batch_file, _ = self.batch_files[0]
        batch_path = os.path.join(self.batch_output_dir, batch_file)
        input_tensor = mx.load(batch_path)
        
        # Calculate the lengths based on the first batch
        return mx.sum(mx.greater(input_tensor, 0), axis=1)