from dataset.batch_dataset_base import BatchDatasetBase

class FineTuneBatchDataset(BatchDatasetBase):
    def __init__(self, batch_output_dir, batchfile_prefix):
        # call constructor
        super().__init__(batch_output_dir, batchfile_prefix)