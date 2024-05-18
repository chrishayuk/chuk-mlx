class BatchDatasetBase:
    def __init__(self):
        # empty list
        self.batch_files = []
        self.length = 0

    def _load_batch_files(self):
        # not implemented
        raise NotImplementedError("Subclasses must implement _load_batch_files method")
    
    def __len__(self):
        return self.length
    
    def __iter__(self):
        # return
        return self

    def __next__(self):
        # if not a batch file, stop
        if not self.batch_files:
            raise StopIteration
        
        # pop the batch_file and target_file
        batch_file, target_file = self.batch_files.pop(0)

        # load the tensors
        input_tensor, target_tensor = self._load_tensors(batch_file, target_file)

        # return the tensors
        return input_tensor, target_tensor

    def _load_tensors(self, batch_file, target_file):
        # not implemented
        raise NotImplementedError("Subclasses must implement _load_tensors method")