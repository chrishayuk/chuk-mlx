import os
import mlx.core as mx

class BatchDatasetBase:
    def __init__(self, batch_output_dir, batchfile_prefix):
        # Initialize
        self.batch_output_dir = batch_output_dir
        self.batchfile_prefix = batchfile_prefix
        self.batch_files = []
        self.length = 0
        self.current_index = 0
        self.cache = {}
        
        # Load batch file names
        self._load_batch_files()
        self.length = len(self.batch_files)

    def __len__(self):
        # Returns the length of the batch
        return self.length

    def __getitem__(self, index):
        # Get the batch file
        if index in self.cache:
            return self.cache[index]
        else:
            batch_file = self.batch_files[index]
            return self._load_and_cache_tensors(batch_file, index)

    def _load_batch_files(self):
        # Loop through the directory and load batch files
        for filename in os.listdir(self.batch_output_dir):
            if filename.startswith(self.batchfile_prefix) and filename.endswith(".npz"):
                self.batch_files.append(filename)
        self.batch_files.sort()

    def _load_and_cache_tensors(self, batch_file, index):
        # Load the input batch file
        batch_path = os.path.join(self.batch_output_dir, batch_file)
        batch_data = mx.load(batch_path)

        # Ensure that the 'target_tensor' key exists in the target batch
        if 'target_tensor' not in batch_data:
            raise KeyError(f"'target_tensor' not found in the file {batch_file}")

        # Get the input and target tensors
        input_tensor = batch_data['input_tensor']
        target_tensor = batch_data['target_tensor']
        lengths = batch_data.get('lengths', [len(seq) for seq in target_tensor])

        # Cache the current and next batch
        self.cache[index] = (input_tensor, target_tensor, lengths)
        if index + 1 < self.length and index + 1 not in self.cache:
            next_batch_file = self.batch_files[index + 1]
            self._load_and_cache_tensors(next_batch_file, index + 1)
        
        # Remove the previous batch from cache if it exists
        if index - 1 in self.cache:
            del self.cache[index - 1]

        return input_tensor, target_tensor, lengths

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        # Check if we have exceeded the length
        if self.current_index >= self.length:
            raise StopIteration

        # Set the batch data as the data in the current index
        batch_data = self[self.current_index]

        # Increment
        self.current_index += 1

        # Return the data
        return batch_data
