import os
import mlx.core as mx
from .batch_dataset_base import BatchDatasetBase

class FineTuneBatchDataset(BatchDatasetBase):
    def __init__(self, batch_output_dir, batchfile_prefix):
        # call the constructor
        super().__init__()

        # set the output directory and prefiz
        self.batch_output_dir = batch_output_dir
        self.batchfile_prefix = batchfile_prefix
        
        # load the batch files
        self._load_batch_files()

        # set the length and lengths
        self.length = len(self.batch_files)
        self.lengths = self._get_lengths()

    def _load_batch_files(self):
        # loop through the directory
        for filename in os.listdir(self.batch_output_dir):
            # check for a file
            if filename.startswith(self.batchfile_prefix) and filename.endswith(".npy"):
                self.batch_files.append(filename)

        # sort the files in order
        self.batch_files.sort()

    def __getitem__(self, index):
        # get the batch file
        batch_file = self.batch_files[index]

        # load the tensor
        concatenated_tensor = self._load_tensor(batch_file)

        # return the tensor
        return concatenated_tensor, self.lengths

    def _load_tensor(self, batch_file):
        # set the batch file
        batch_path = os.path.join(self.batch_output_dir, batch_file)

        # load the tensor
        concatenated_tensor = mx.load(batch_path)

        # return the tensor
        return concatenated_tensor

    def _get_lengths(self):
        # set the batch file
        batch_file = self.batch_files[0]
        batch_path = os.path.join(self.batch_output_dir, batch_file)

        # load the tensor
        concatenated_tensor = mx.load(batch_path)

        # calculaate the lengths
        lengths = mx.sum(mx.greater(concatenated_tensor, 0), axis=1)

        # return the lengths
        return lengths