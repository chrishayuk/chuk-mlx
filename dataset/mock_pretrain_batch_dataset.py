import os
import numpy as np
from dataset.batch_dataset_base import PreTrainBatchDatasetBase

class MockPreTrainBatchDataset(PreTrainBatchDatasetBase):
    def __init__(self, batch_output_dir, batchfile_prefix, num_batches, batch_size, seq_length):
        # Initialize base class with directory and prefix
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
        # check for the output directory, and create if doesn't exist
        if not os.path.exists(self.batch_output_dir):
            os.makedirs(self.batch_output_dir)

        # loop through the batches
        for i in range(self.num_batches):
            # generate random tensors
            input_tensor = np.random.randint(0, 100, (self.batch_size, self.seq_length))
            target_tensor = np.random.randint(0, 100, (self.batch_size, self.seq_length))

            # calculate lengths
            lengths = np.full((self.batch_size, self.seq_length), self.seq_length)

            # get the batch file names
            batch_file = f"{self.batchfile_prefix}_{i}.npz"
            batch_path = os.path.join(self.batch_output_dir, batch_file)

            # save them
            np.savez(batch_path, input_tensor=input_tensor, target_tensor=target_tensor, lengths=lengths)

    def _load_batch_files(self):
        # Call parent method to load batch files after mock generation
        super()._load_batch_files()
