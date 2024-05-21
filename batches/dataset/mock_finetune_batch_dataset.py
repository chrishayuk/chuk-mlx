import os
import mlx.core as mx

class MockFineTuneBatchDataset:
    def __init__(self, batch_output_dir, batchfile_prefix, num_batches, batch_size, seq_length):
        # intiailize
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

        for i in range(self.num_batches):
            # generate a random number for the input and target
            input_tensor = mx.random.randint(0, 100, (self.batch_size, self.seq_length))
            target_tensor = mx.random.randint(0, 100, (self.batch_size, self.seq_length))

            # lengths looks odd
            lengths = mx.random.randint(1, self.seq_length + 1, (self.batch_size,))

            # concatenate the tensor
            concatenated_tensor = mx.concatenate((input_tensor, target_tensor), axis=1)

            # set the batch file and path
            batch_file = f"{self.batchfile_prefix}_{i}.npz"
            batch_path = os.path.join(self.batch_output_dir, batch_file)

            # save it
            mx.savez(batch_path, concatenated_tensor=concatenated_tensor, lengths=lengths)

            # add
            self.batch_files.append(batch_file)

    def __len__(self):
        # return the size of the dataset
        return len(self.batch_files)

    def __getitem__(self, index):
        # load the tensor
        concatenated_tensor = self._load_tensor(self.batch_files[index])

        # get the lengths
        lengths = self._get_lengths(index)

        # return the tensor and lengths
        return concatenated_tensor['concatenated_tensor'], lengths

    def _load_tensor(self, batch_file):
         # get the batch file
        batch_path = os.path.join(self.batch_output_dir, batch_file)

        # load the tensor
        concatenated_tensor = mx.load(batch_path)

        # return the tensor
        return concatenated_tensor

    def _get_lengths(self, index):
        # load the tensor
        concatenated_tensor = self._load_tensor(self.batch_files[index])
        
        # return the lengths
        return concatenated_tensor['lengths']