import os
import mlx.core as mx
from batches.dataset.batch_dataset_base import BatchDatasetBase

class PreTrainBatchDatasetBase(BatchDatasetBase):
    def __init__(self, batch_output_dir, batchfile_prefix):
        # Call the base constructor
        super().__init__()

        # Set the output directories etc.
        self.batch_output_dir = batch_output_dir
        self.batchfile_prefix = batchfile_prefix
        self.lengths_cache = {}

        # Call the method directly instead of using super() in _load_batch_files
        self._load_batch_files()
        self.length = len(self.batch_files)
    
    def __getitem__(self, index):
        # Get the batch file
        batch_file = self.batch_files[index]

        # Load the input and target tensors
        return self._load_and_cache_tensors(batch_file, index)
    
    def _load_and_cache_tensors(self, batch_file, index):
        # Load the input batch file
        batch_path = os.path.join(self.batch_output_dir, batch_file)
        batch_data = mx.load(batch_path)

        # Ensure that the 'target_tensor' key exists in the target batch
        if 'target_tensor' not in batch_data:
            raise KeyError(f"'target_tensor' not found in the file {target_batch_file}")

        # Get the input and target tensors
        input_tensor = batch_data['input_tensor']
        target_tensor = batch_data['target_tensor']

        # Cache lengths if not already cached
        if 'lengths' not in batch_data:
            # If lengths are missing in input batch, regenerate them from target tensors
            lengths = mx.array([len(seq) for seq in target_tensor])
        else:
            lengths = batch_data['lengths']

        # Cache lengths
        self.lengths_cache[index] = lengths

        # Return the tensors and lengths
        return input_tensor, target_tensor, lengths
