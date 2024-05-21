import mlx.core as mx
import mlx.nn as nn
import models
import models.llama
import models.llama.llama_model
import models.llama.model
from models.load_weights import load_model_weights
from models.model_config import ModelConfig
from utils.huggingface_utils import load_from_hub
from trainer import Trainer
from models.loss_function import loss
import mlx.optimizers as optim
from batches.dataset.finetune_batch_dataset import FineTuneBatchDataset

def chukloss(model, inputs, targets, lengths):
    # Check if inputs and targets are concatenated
    if inputs.shape[1] == targets.shape[1] * 2:
        split_index = inputs.shape[1] // 2
        input_tensor = inputs[:, :split_index]
        target_tensor = inputs[:, split_index:]
    else:
        input_tensor = inputs
        target_tensor = targets

    # Run model on inputs
    logits, _ = model(input_tensor)
    logits = logits.astype(mx.float32)

    # Create a mask for the padding tokens
    length_mask = mx.arange(input_tensor.shape[1])[None, :] < lengths[:, None]

    # Calculate the cross-entropy loss
    ce = nn.losses.cross_entropy(logits, target_tensor)

    # Apply the mask to exclude padding tokens
    ce = ce * length_mask

    # Calculate the number of valid tokens
    ntoks = length_mask.sum().item()

    # Normalize the loss by the number of valid tokens
    ce = ce.sum() / ntoks
    return ce, ntoks


# set the model name
model_name = "ibm-granite/granite-3b-code-instruct"

# load the model from huggingface
print(f"Loading Model: {model_name}")
model_path = load_from_hub(model_name)

# load config
model_config = ModelConfig.load(model_path)

# load the model weights
weights = load_model_weights(model_path)

# create the model instance
model = models.llama.model.Model(model_config)

# Model Loaded
print(f"Model Loaded: {model_name}")

# loading weights
print("loading weights")
model.load_weights(list(weights.items()))
mx.eval(model.parameters())
print("weights loaded")

# Generate a batch
input_files = ['./sample_data/calvin_scale_llama/train.jsonl']
output_dir = './output/calvin'
batch_output_dir = f'{output_dir}/batches'
batchfile_prefix = 'calvin'
max_sequence_length = 512
batch_size = 512
total_iterations = 20

# checkpointing, we want to checkpoint every 5 batches
checkpoint_freq=500
checkpoint_output_dir = f'{output_dir}/checkpoints'

# Training loop
num_epochs = 1

# Define the optimizer with learning rate scheduling, same settings as llama-2
initial_lr = 2e-5
lr_schedule = optim.cosine_decay(initial_lr, num_epochs * 1000)
optimizer = optim.AdamW(
    learning_rate=lr_schedule,
    betas=[0.9,0.95],
    eps=1e-5,
    weight_decay=0.1
)

# Create value and grad function for loss
loss_function = nn.value_and_grad(model, chukloss)

# Load the batch data
output_dir = './output/calvin'
batch_output_dir = f'{output_dir}/batches'
batchfile_prefix = "calvin"
batch_dataset = FineTuneBatchDataset(batch_output_dir, batchfile_prefix)

# Create an instance of the Trainer
trainer = Trainer(model, optimizer, loss_function, 1, checkpoint_output_dir, checkpoint_freq)

# Train the model
trainer.train(num_epochs, batch_dataset, total_iterations)