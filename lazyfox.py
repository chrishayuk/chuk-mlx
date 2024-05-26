import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from batches.dataset.pretrain_batch_dataset import PreTrainBatchDataset
from chuk_loss_function.lazyfox_loss_function import chukloss
from chuk_models.simple_language_model import SimpleLanguageModel
from models.model_config import ModelConfig
from utils.tokenizer_loader import load_tokenizer
from training.trainer import Trainer

# settings
input_files = ['./sample_data/lazyfox_train.jsonl']
tokenizer_name = 'lazyfox_tokenizer'
output_dir = './output/lazyfox'
batch_output_dir = f'{output_dir}/batches'
batchfile_prefix = 'lazyfox'
max_sequence_length = 16
batch_size = 1024

# set the config for simple language model
config_settings = {}
config_settings["vocab_size"]=16
config_settings["hidden_size"]=32
config_settings["intermediate_size"]=64
config_settings["num_hidden_layers"]=1
config_settings["hidden_act"] = "silu"
model_config = ModelConfig.from_dict(config_settings)

# Load the simple language model
model = SimpleLanguageModel(model_config)

# Load tokenizer and define vocabulary size
tokenizer = load_tokenizer(tokenizer_name)
pad_token_id = tokenizer.pad_token_id
vocab_size = len(tokenizer.vocab)

# Define the optimizer
learning_rate = 0.01
optimizer = optim.Adam(learning_rate=learning_rate)

# Create value and grad function for loss
loss_function = nn.value_and_grad(model, chukloss)

# Load the batch data
batch_output_dir = "./output/lazyfox/batches"
batchfile_prefix = "lazyfox"
batch_dataset = PreTrainBatchDataset(batch_output_dir, batchfile_prefix)

# Create an instance of the Trainer
trainer = Trainer(model, tokenizer, optimizer, loss_function)

# set the number of epochs
num_epochs = 50

# train the model
print("Starting Training\n")
trainer.train(num_epochs, batch_dataset)
print("\n\nCompleted Training\n")

input_sequence = 'the quick brown'
input_indices = tokenizer.encode(input_sequence, add_special_tokens=False)
input_tensor = mx.array([input_indices])

output = model(input_tensor)
predicted_index = mx.argmax(output[:, -1, :], axis=-1).item()
predicted_word = tokenizer.decode([predicted_index])

print(f"Input sequence: {input_sequence}")
print(f"Predicted next word: {predicted_word}")