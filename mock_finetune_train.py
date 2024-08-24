from core.dataset.mock_finetune_batch_dataset import MockFineTuneBatchDataset
from core.models.architectures.mock.mock_loss_function import mock_value_and_grad, mockloss
from core.models.architectures.mock.mock_model import MockModel
from core.models.architectures.mock.mock_optimizer import MockOptimizer
from training.trainer import Trainer

# Import the tokenizer, assuming it's required for the new Trainer structure
from core.utils.tokenizer_loader import load_tokenizer

# Instantiate mock components
model = MockModel()
optimizer = MockOptimizer()
loss_function = mock_value_and_grad(mockloss)

# Settings
output_dir = './output/mock_finetune'
checkpoint_output_dir = f'{output_dir}/checkpoints'
batchfile_prefix = 'batch'
num_batches = 100
batch_size = 32
seq_length = 50

# TODO: we do need a mock tokenizer
# Load a mock tokenizer (adjust this part according to your tokenizer implementation)
tokenizer = load_tokenizer('mistralai/Mistral-7B-Instruct-v0.2')

# Create mock dataset
mock_dataset = MockFineTuneBatchDataset(
    batch_output_dir=output_dir,
    batchfile_prefix=batchfile_prefix,
    num_batches=num_batches,
    batch_size=batch_size,
    seq_length=seq_length,
    sep_token_id=tokenizer.sep_token_id
)

# Instantiate the trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    optimizer=optimizer,
    loss_function=loss_function,
    progress_interval=10,
    checkpoint_dir=checkpoint_output_dir,
    checkpoint_freq=50,
    warmup_steps=0  # Adjust warmup steps if needed
)

# Run training
trainer.train(
    num_epochs=5,
    batch_dataset=mock_dataset,
    num_iterations=200
)
