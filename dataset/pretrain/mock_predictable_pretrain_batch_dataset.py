import os
import numpy as np
from dataset.pretrain.pretrain_batch_dataset_base import PreTrainBatchDatasetBase

class SimpleMultiplicationDataset(PreTrainBatchDatasetBase):
    def __init__(self, batch_output_dir, batchfile_prefix, num_batches, batch_size, seq_length):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.batch_output_dir = batch_output_dir
        self.batchfile_prefix = batchfile_prefix

        self._generate_predictable_batches()

        super().__init__(batch_output_dir, batchfile_prefix)

    def _generate_predictable_batches(self):
        if not os.path.exists(self.batch_output_dir):
            os.makedirs(self.batch_output_dir)

        for i in range(self.num_batches):
            input_tensor = np.random.randint(0, 10, (self.batch_size, self.seq_length))
            target_tensor = input_tensor * 2
            lengths = np.full((self.batch_size,), self.seq_length)

            batch_file = f"{self.batchfile_prefix}_{i}.npz"
            batch_path = os.path.join(self.batch_output_dir, batch_file)

            np.savez(batch_path, input_tensor=input_tensor, target_tensor=target_tensor, lengths=lengths)
