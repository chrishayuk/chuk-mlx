import os
import mlx.core as mx
from dataset.pretrain.pretrain_batch_dataset_base import PreTrainBatchDatasetBase

class FineTuneBatchDataset(PreTrainBatchDatasetBase):
    def __init__(self, batch_output_dir, batchfile_prefix):
        # call constructor
        super().__init__(batch_output_dir, batchfile_prefix)