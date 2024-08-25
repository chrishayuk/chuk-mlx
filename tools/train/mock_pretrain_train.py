import os
import sys

# Add the parent directory of the tools directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# imports
from core.dataset.mock_pretrain_batch_dataset import MockPreTrainBatchDataset
from core.models.architectures.mock.mock_loss_function import mock_value_and_grad, mockloss
from core.models.architectures.mock.mock_model import MockModel
from core.models.architectures.mock.mock_optimizer import MockOptimizer
from core.utils.tokenizer_loader import load_tokenizer
from training.trainer import Trainer

# Instantiate mock components
model = MockModel()
optimizer = MockOptimizer()
loss_function = mock_value_and_grad(mockloss)

# Settings
batch_output_dir = './output/batches/mock_pretrain'
checkpoint_output_dir = f'./output/checkpoints/mock_pretrain'
batchfile_prefix = 'batch'
num_batches = 100
batch_size = 2
seq_length = 50

# TODO: we do need a mock tokenizer
# Load a mock tokenizer (adjust this part according to your tokenizer implementation)
tokenizer = load_tokenizer('mistralai/Mistral-7B-Instruct-v0.2')

# Create mock dataset
mock_dataset = MockPreTrainBatchDataset(
    batch_output_dir=batch_output_dir,
    batchfile_prefix=batchfile_prefix,
    num_batches=num_batches,
    batch_size=batch_size,
    seq_length=seq_length
)

# Instantiate the trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    optimizer=optimizer,
    loss_function=loss_function,
    progress_interval=10,
    checkpoint_dir=checkpoint_output_dir,
    checkpoint_freq_epochs=50,
    warmup_steps=0
)

# Run training
trainer.train(
    num_epochs=5,
    batch_dataset=mock_dataset,
    num_iterations=200
)