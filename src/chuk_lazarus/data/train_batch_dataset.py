from chuk_lazarus.data.batch_dataset_base import BatchDatasetBase
from chuk_lazarus.utils.model_adapter import ModelAdapter


class TrainBatchDataset(BatchDatasetBase):
    def __init__(
        self,
        batch_output_dir,
        batchfile_prefix,
        pre_cache_size=5,
        model_adapter=ModelAdapter(framework="mlx"),
    ):
        # call constructor
        super().__init__(
            batch_output_dir, batchfile_prefix, pre_cache_size, model_adapter=model_adapter
        )
