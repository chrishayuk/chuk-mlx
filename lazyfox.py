import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from batches.dataset.pretrain_batch_dataset import PreTrainBatchDataset
from models.loss_function import loss
from chuk_models.simple_language_model import SimpleLanguageModel
from models.model_config import ModelConfig
from utils.tokenizer_loader import load_tokenizer
from trainer import Trainer

def chukloss(model, inputs, targets, lengths):
    # Run model on inputs
    logits = model(inputs)  # Assuming model returns only logits
    logits = logits.astype(mx.float32)

    # Create a mask for the padding tokens
    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    # Calculate the cross-entropy loss
    ce = nn.losses.cross_entropy(logits, targets)
    
    # Apply the mask to exclude padding tokens
    ce = ce * length_mask

    # Calculate the number of valid tokens
    ntoks = length_mask.sum().item()

    # Normalize the loss by the number of valid tokens
    ce = ce.sum() / ntoks
    return ce, ntoks

# settings
input_files = ['./sample_data/lazyfox_train.jsonl']
tokenizer_name = 'lazyfox_tokenizer'
output_dir = './output/lazyfox'
batch_output_dir = f'{output_dir}/batches'
batchfile_prefix = 'lazyfox'
max_sequence_length = 16
batch_size = 1024

# # Load the input batch
# print("loading batch")
# input_tensor = mx.load(f"{batch_output_dir}/{batchfile_prefix}_0001.npy")
# target_tensor = mx.load(f"{batch_output_dir}/{batchfile_prefix}_0001_target.npy")
# print("batch loaded")

# set the config for simple language model
config_settings = {}
config_settings["vocab_size"]=16
config_settings["hidden_size"]=32
config_settings["intermediate_size"]=64
config_settings["num_hidden_layers"]=1
config_settings["hidden_act"] = "silu"
model_config = ModelConfig.from_dict(config_settings)

# # Print the Configuration
# model_config.print_config()

# # Print the Layers
# model_config.print_layers()

# Load the simple language model
model = SimpleLanguageModel(model_config)

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
trainer = Trainer(model, optimizer, loss_function)

# set the number of epochs
num_epochs = 50

# train the model
print("Starting Training\n")
trainer.train(num_epochs, batch_dataset)
print("\n\nCompleted Training\n")

# Prediction
# Load tokenizer and define vocabulary size
tokenizer = load_tokenizer(tokenizer_name)
pad_token_id = tokenizer.pad_token_id
vocab_size = len(tokenizer.vocab)
input_sequence = 'the quick brown'
input_indices = tokenizer.encode(input_sequence, add_special_tokens=False)
input_tensor = mx.array([input_indices])

output = model(input_tensor)
predicted_index = mx.argmax(output[:, -1, :], axis=-1).item()
predicted_word = tokenizer.decode([predicted_index])

print(f"Input sequence: {input_sequence}")
print(f"Predicted next word: {predicted_word}")
