import os
import mlx.core as mx
from .batch_dataset_base import BatchDatasetBase

class PreTrainBatchDataset(BatchDatasetBase):
    def __init__(self, batch_output_dir, batchfile_prefix):
        # call base construtor
        super().__init__()

        # set the output directory
        self.batch_output_dir = batch_output_dir
        self.batchfile_prefix = batchfile_prefix

        # load the batch files
        self._load_batch_files()
        
        # set the lengths
        self.length = len(self.batch_files)
        self.lengths = self._get_lengths()

    def _load_batch_files(self):
        # loop through the files in the directory
        for filename in os.listdir(self.batch_output_dir):
            # if an input file
            if filename.startswith(self.batchfile_prefix) and filename.endswith(".npy") and "target" not in filename:
                # add the filename
                self.batch_files.append(filename)
        self.batch_files.sort()

    def __getitem__(self, index):
        # set the batch file as the current index
        batch_file = self.batch_files[index]

        # load the in input and target tensors
        input_tensor, target_tensor = self._load_tensors(batch_file)

        # return the tensors and the lengths
        return input_tensor, target_tensor, self.lengths

    def _load_tensors(self, batch_file):
        # set the input and target filenames
        input_path = os.path.join(self.batch_output_dir, batch_file)
        target_path = input_path.replace(".npy", "_target.npy")

        # load the input and target tensors
        input_tensor = mx.load(input_path)
        target_tensor = mx.load(target_path)

        # return input and target tensors
        return input_tensor, target_tensor

    def _get_lengths(self):
        # set the batch file
        batch_file = self.batch_files[0]
        input_path = os.path.join(self.batch_output_dir, batch_file)

        # load the input tensor
        input_tensor = mx.load(input_path)

        # calculate the lengths
        lengths = mx.sum(mx.greater(input_tensor, 0), axis=1)
        
        # return the lengths
        return lengths