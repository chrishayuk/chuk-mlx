import os
import numpy as np
import mlx.core as mx

class MockPreTrainBatchDataset:
    def __init__(self, batch_output_dir, batchfile_prefix, num_batches, batch_size, seq_length):
        self.batch_output_dir = batch_output_dir
        self.batchfile_prefix = batchfile_prefix
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.batch_files = []
        self.lengths = []

        # Generate mock batches
        self._generate_mock_batches()

    def _generate_mock_batches(self):
        if not os.path.exists(self.batch_output_dir):
            os.makedirs(self.batch_output_dir)
        
        for i in range(self.num_batches):
            input_tensor = np.random.randint(0, 100, (self.batch_size, self.seq_length))
            target_tensor = np.random.randint(0, 100, (self.batch_size, self.seq_length))
            lengths = np.random.randint(1, self.seq_length + 1, (self.batch_size,))
            batch_file = f"{self.batchfile_prefix}_{i}.npz"
            batch_path = os.path.join(self.batch_output_dir, batch_file)
            np.savez(batch_path, input_tensor=input_tensor, target_tensor=target_tensor, lengths=lengths)
            self.batch_files.append(batch_file)
        
        # Set lengths based on the first batch (simulating the logic)
        concatenated_tensor = np.load(os.path.join(self.batch_output_dir, self.batch_files[0]))
        self.lengths = concatenated_tensor['lengths']

    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, index):
        batch_file = self.batch_files[index]
        concatenated_tensor = self._load_tensor(batch_file)
        return concatenated_tensor['input_tensor'], concatenated_tensor['target_tensor'], concatenated_tensor['lengths']

    def _load_tensor(self, batch_file):
        batch_path = os.path.join(self.batch_output_dir, batch_file)
        concatenated_tensor = np.load(batch_path)
        return concatenated_tensor

    def _get_lengths(self):
        return self.lengths
