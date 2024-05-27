import os
class BatchDatasetBase:
    def __init__(self):
        # initialize
        self.batch_output_dir = None
        self.batchfile_prefix = None
        self.batch_files = []
        self.length = 0
        self.current_index = 0
        self.lengths = None

    def __len__(self):
        # returns the length of the batch
        return self.length

    def __getitem__(self, index):
        # not implemented, must be implemented by the subclass
        raise NotImplementedError

    def _load_batch_files(self):
        # loop through the dir
        for filename in os.listdir(self.batch_output_dir):
            # check for a batch file
            if filename.startswith(self.batchfile_prefix) and filename.endswith(".npz"):
                # add it
                self.batch_files.append(filename)

        # sort
        self.batch_files.sort()

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        # check we haven't exceeded the length
        if self.current_index >= self.length:
            raise StopIteration
        
        # set the batch data as the data in the current index
        batch_data = self[self.current_index]
        
        # increment
        self.current_index += 1

        # return the data
        return batch_data