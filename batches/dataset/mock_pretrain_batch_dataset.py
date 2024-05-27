import os
import numpy as np

class MockPreTrainBatchDataset:
    def __init__(self, batch_output_dir, batchfile_prefix, num_batches, batch_size, seq_length):
        # initialize
        self.batch_output_dir = batch_output_dir
        self.batchfile_prefix = batchfile_prefix
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.batch_files = []

        # Generate mock batches
        self._generate_mock_batches()

    def _generate_mock_batches(self):
        # create the output directory if it doesn't exist
        if not os.path.exists(self.batch_output_dir):
            os.makedirs(self.batch_output_dir)

        # loop through the batches
        for i in range(self.num_batches):
            # input tensor
            input_tensor = np.random.randint(0, 100, (self.batch_size, self.seq_length))
            target_tensor = np.random.randint(0, 100, (self.batch_size, self.seq_length))
            lengths = np.random.randint(1, self.seq_length + 1, (self.batch_size,))

            # batch file and path
            batch_file = f"{self.batchfile_prefix}_{i}.npz"
            batch_path = os.path.join(self.batch_output_dir, batch_file)

            # save the batch
            np.savez(batch_path, input_tensor=input_tensor, target_tensor=target_tensor, lengths=lengths)
            self.batch_files.append(batch_file)

    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, index):
        batch_file = self.batch_files[index]
        batch_data = self._load_tensor(batch_file)

        # get the lengths for the current batch
        lengths = batch_data['lengths']

        return batch_data['input_tensor'], batch_data['target_tensor'], lengths

    def _load_tensor(self, batch_file):
        # get the batch file
        batch_path = os.path.join(self.batch_output_dir, batch_file)

        # load the tensor
        batch_data = np.load(batch_path)

        # return the tensor
        return batch_data

    def _get_lengths(self, index):
        # load the tensor
        batch_data = self._load_tensor(self.batch_files[index])

        # return lengths
        return batch_data['lengths']

# # Usage example
# batch_dataset = MockPreTrainBatchDataset(batch_output_dir='batches', batchfile_prefix='batch', num_batches=10, batch_size=32, seq_length=50)
# input_tensor, target_tensor, lengths = batch_dataset[0]
# print("Input Tensor:")
# print(input_tensor)

# print("\nTarget Tensor:")
# print(target_tensor)

# print("\nLengths:")
# print(lengths)
