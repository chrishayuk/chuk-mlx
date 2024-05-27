from utils.tokenizer_loader import load_tokenizer
from training.trainer import Trainer
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from dataset.pretrain.pretrain_batch_dataset import PreTrainBatchDataset
from chuk_loss_function.lazyfox_loss_function import chukloss
from chuk_models.simple_language_model import SimpleLanguageModel
from models.model_config import ModelConfig

# Settings
input_files = ['./sample_data/lazyfox_train.jsonl']
output_dir = './output/lazyfox'
batch_output_dir = f'{output_dir}/batches'
batchfile_prefix = 'lazyfox'

# Load tokenizer and define vocabulary size
tokenizer_name = 'lazyfox_tokenizer'
tokenizer = load_tokenizer(tokenizer_name)
pad_token_id = tokenizer.pad_token_id
vocab_size = len(tokenizer.vocab)

# Validate bos_token_id and eos_token_id are integers
bos_token_id = int(tokenizer.bos_token_id[0]) if isinstance(tokenizer.bos_token_id, list) else int(tokenizer.bos_token_id)
eos_token_id = int(tokenizer.eos_token_id[0]) if isinstance(tokenizer.eos_token_id, list) else int(tokenizer.eos_token_id)

# Print to verify values
print(f'bos_token_id: {bos_token_id}, eos_token_id: {eos_token_id}')

# Set the config for simple language model
config_settings = {
    "vocab_size": vocab_size,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "hidden_act": "silu",
    "bos_token_id": bos_token_id,  # Begin-of-sequence token ID
    "eos_token_id": eos_token_id,  # End-of-sequence token ID
    "max_position_embeddings": vocab_size  # Maximum sequence length
}

# Load the model config
model_config = ModelConfig.from_dict(config_settings)

# Load the simple language model
model = SimpleLanguageModel(model_config)

# Define the optimizer
learning_rate = 0.01
optimizer = optim.Adam(learning_rate=learning_rate)

# Create value and grad function for loss
loss_function = nn.value_and_grad(model, chukloss)

# Load the batch data
batch_dataset = PreTrainBatchDataset(batch_output_dir, batchfile_prefix)

# Create an instance of the Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    optimizer=optimizer,
    loss_function=loss_function,
    progress_interval=10,
    checkpoint_dir=output_dir,
    checkpoint_freq=50,
    warmup_steps=0  # Adjust warmup steps if needed
)

# Set the number of epochs
num_epochs = 50

# Train the model
print("Starting Training\n")
trainer.train(num_epochs, batch_dataset)
print("\n\nCompleted Training\n")

#################
# Test the model
#################

# tokenize
input_sequence = 'the quick brown'
input_indices = tokenizer.encode(input_sequence, add_special_tokens=False)
input_tensor = mx.array([input_indices])

# forward pass
output = model(input_tensor)

# get prediction
predicted_index = mx.argmax(output[:, -1, :], axis=-1).item()
predicted_word = tokenizer.decode([predicted_index])

# show prediction
print(f"Input sequence: {input_sequence}")
print(f"Predicted next word: {predicted_word}")
