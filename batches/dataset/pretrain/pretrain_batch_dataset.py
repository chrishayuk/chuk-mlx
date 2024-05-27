import os
import mlx.core as mx
from batches.dataset.pretrain.pretrain_batch_dataset_base import PreTrainBatchDatasetBase

class PreTrainBatchDataset(PreTrainBatchDatasetBase):
    def __init__(self, batch_output_dir, batchfile_prefix):
        # call constructor
        super().__init__(batch_output_dir, batchfile_prefix)