import os
import mlx.core as mx

class MockFineTuneBatchDataset:
    def __init__(self, batch_output_dir, batchfile_prefix, num_batches, batch_size, seq_length, sep_token_id=99):
        # Initialize
        self.batch_output_dir = batch_output_dir
        self.batchfile_prefix = batchfile_prefix
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.batch_files = []
        self.sep_token_id = sep_token_id

        # Generate mock batches
        self._generate_mock_batches()

    def _generate_mock_batches(self):
        # Create the output directory if it doesn't exist
        if not os.path.exists(self.batch_output_dir):
            os.makedirs(self.batch_output_dir)

        for i in range(self.num_batches):
            # Generate random input and target tensors starting from 3 onwards
            input_tensor = mx.random.randint(3, 100, (self.batch_size, self.seq_length // 2 - 1))
            target_tensor = mx.random.randint(3, 100, (self.batch_size, self.seq_length // 2))

            # Append the separator token at the end of the input tensor
            sep_column = mx.full((self.batch_size, 1), self.sep_token_id)
            input_tensor = mx.concatenate((input_tensor, sep_column), axis=1)

            # Create lengths (full length of the input tensor including the separator)
            lengths = mx.full((self.batch_size,), self.seq_length // 2)

            # Concatenate the input and target tensors
            concatenated_tensor = mx.concatenate((input_tensor, target_tensor), axis=1)

            # Set the batch file and path
            batch_file = f"{self.batchfile_prefix}_{i}.npz"
            batch_path = os.path.join(self.batch_output_dir, batch_file)

            # Save it
            mx.savez(batch_path, concatenated_tensor=concatenated_tensor, lengths=lengths)

            # Add to the batch files list
            self.batch_files.append(batch_file)

    def __len__(self):
        # Return the size of the dataset
        return len(self.batch_files)

    def __getitem__(self, index):
        # Load the tensor
        concatenated_tensor = self._load_tensor(self.batch_files[index])

        # Get the lengths
        lengths = self._get_lengths(index)

        # Debugging: Check if the separator token is present at the end of the input tensor
        for i, seq in enumerate(concatenated_tensor['concatenated_tensor'][:, :self.seq_length // 2]):
            if seq[-1] != self.sep_token_id:
                print(f"Warning: No separator found at the end of sequence {i} of batch {index}")

        # Return the tensor and lengths
        return concatenated_tensor['concatenated_tensor'], lengths

    def _load_tensor(self, batch_file):
        # Get the batch file path
        batch_path = os.path.join(self.batch_output_dir, batch_file)

        # Load the tensor
        concatenated_tensor = mx.load(batch_path)

        # Return the tensor
        return concatenated_tensor

    def _get_lengths(self, index):
        # Load the tensor
        concatenated_tensor = self._load_tensor(self.batch_files[index])
        
        # Return the lengths
        return concatenated_tensor['lengths']
