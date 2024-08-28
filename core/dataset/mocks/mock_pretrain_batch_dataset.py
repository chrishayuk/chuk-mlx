import os
import numpy as np
from core.dataset.train_batch_dataset import TrainBatchDataset

class MockPreTrainBatchDataset(TrainBatchDataset):
    def __init__(self, batch_output_dir, batchfile_prefix, num_batches, batch_size, seq_length):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.batch_output_dir = batch_output_dir
        self.batchfile_prefix = batchfile_prefix
        
        # Generate mock batches before calling super init to load them
        self._generate_mock_batches()
        
        # Call the base class constructor to handle the rest
        super().__init__(batch_output_dir, batchfile_prefix)

    def _generate_mock_batches(self):
        # Ensure the output directory exists
        if not os.path.exists(self.batch_output_dir):
            os.makedirs(self.batch_output_dir)

        # Loop through the batches and generate mock data
        for i in range(self.num_batches):
            # Generate random tensors
            input_tensor = np.random.randint(0, 100, (self.batch_size, self.seq_length))
            target_tensor = np.random.randint(0, 100, (self.batch_size, self.seq_length))

            # Generate attention masks based on the presence of actual tokens (non-zero)
            attention_mask_tensor = np.where(input_tensor > 0, 1, 0)

            # Calculate lengths as a 1D array
            lengths = np.full((self.batch_size,), self.seq_length)  # Ensure lengths is a 1D array

            # Get the batch file name
            batch_file = f"{self.batchfile_prefix}_{i}.npz"
            batch_path = os.path.join(self.batch_output_dir, batch_file)

            # Save tensors including attention masks
            np.savez(batch_path, input_tensor=input_tensor, target_tensor=target_tensor, attention_mask_tensor=attention_mask_tensor, lengths=lengths)


    def _load_batch_files(self):
        # Call parent method to load batch files after mock generation
        super()._load_batch_files()
