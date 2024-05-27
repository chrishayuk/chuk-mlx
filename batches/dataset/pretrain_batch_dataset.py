import os
import numpy as np
from .batch_dataset_base import BatchDatasetBase

class PreTrainBatchDataset(BatchDatasetBase):
    def __init__(self, batch_output_dir, batchfile_prefix):
        # call base constructor
        super().__init__()

        # set the output directory
        self.batch_output_dir = batch_output_dir
        self.batchfile_prefix = batchfile_prefix

        # initialize the list of batch files
        self.batch_files = []

        # load the batch files
        self._load_batch_files()
        
        # set the length of the dataset
        self.length = len(self.batch_files)

    def _load_batch_files(self):
        # loop through the files in the directory
        for filename in os.listdir(self.batch_output_dir):
            # if an input file
            if filename.startswith(self.batchfile_prefix) and filename.endswith(".npz") and "target" not in filename:
                # add the filename
                self.batch_files.append(filename)
        self.batch_files.sort()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # set the batch file as the current index
        batch_file = self.batch_files[index]
        
        # load the input and target tensors
        input_tensor, target_tensor = self._load_tensors(batch_file)
        
        # get the lengths for the current batch
        lengths = self._get_lengths(index)
        
        # return the tensors and the lengths
        return input_tensor, target_tensor, lengths

    def _load_tensors(self, batch_file):
        # set the batch file path
        batch_path = os.path.join(self.batch_output_dir, batch_file)

        # load the batch data
        batch_data = np.load(batch_path)

        # extract input and target tensors
        input_tensor = batch_data['input_tensor']
        target_tensor = batch_data['target_tensor']

        # return input and target tensors
        return input_tensor, target_tensor

    def _get_lengths(self, index):
        # set the batch file
        batch_file = self.batch_files[index]
        batch_path = os.path.join(self.batch_output_dir, batch_file)
        
        # load the batch data
        batch_data = np.load(batch_path)
        
        # extract lengths
        lengths = batch_data['lengths']
        
        # return the lengths
        return lengths
