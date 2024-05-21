class BatchDatasetBase:
    def __init__(self):
        self.batch_output_dir = None
        self.batchfile_prefix = None
        self.batch_files = []
        self.length = 0
        self.current_index = 0
        self.lengths = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raise NotImplementedError

    def _load_batch_files(self):
        raise NotImplementedError

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= self.length:
            raise StopIteration
        batch_data = self[self.current_index]
        self.current_index += 1
        return batch_data