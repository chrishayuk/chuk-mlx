import mlx.core as mx
from core.dataset.batch_dataset_base import BatchDatasetBase

class TrainBatchDataset(BatchDatasetBase):
    def __init__(self, batch_output_dir, batchfile_prefix, pre_cache_size=5):
        # call constructor
        super().__init__(batch_output_dir, batchfile_prefix, pre_cache_size)