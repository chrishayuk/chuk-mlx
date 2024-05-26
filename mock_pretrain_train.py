from batches.dataset.mock_pretrain_batch_dataset import MockPreTrainBatchDataset
from chuk_loss_function.mock_loss_function import mock_value_and_grad, mockloss
from chuk_models.mock_model import MockModel
from chuk_optimizers.mock_optimizer import MockOptimizer
from training.trainer import Trainer

# Instantiate mock components
model = MockModel()
optimizer = MockOptimizer()
loss_function = mock_value_and_grad(mockloss)

# Settings
output_dir = './output/mock_pretrain'
checkpoint_output_dir = f'{output_dir}/checkpoints'
batchfile_prefix = 'batch'
num_batches = 100
batch_size = 32
seq_length = 50

# Create mock dataset
mock_dataset = MockPreTrainBatchDataset(
    batch_output_dir=output_dir,
    batchfile_prefix=batchfile_prefix,
    num_batches=num_batches,
    batch_size=batch_size,
    seq_length=seq_length
)

# Instantiate the trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_function=loss_function,
    progress_interval=10,
    checkpoint_dir=checkpoint_output_dir,
    checkpoint_freq=50
)

# Run training
trainer.train(
    num_epochs=5,
    batch_dataset=mock_dataset,
    num_iterations=200
)