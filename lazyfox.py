import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from models.loss_function import loss
from models.mlx.simple_language_model import SimpleLanguageModel
from utils.tokenizer_loader import load_tokenizer
from utils.sequence_utility import SequenceUtility
from trainer import Trainer
from dataset import Dataset
from batches.generate_batch import tokenize_and_batch

# settings
input_files = ['./sample_data/lazyfox_train.jsonl']
tokenizer_name = 'lazyfox_tokenizer'
output_dir = './output'
max_sequence_length = 16
batchfile_prefix = 'lazyfox'
batch_size = 1024

# Load tokenizer and define vocabulary size
tokenizer = load_tokenizer(tokenizer_name)
pad_token_id = tokenizer.pad_token_id
vocab_size = len(tokenizer.vocab)

# Generate a batch
print("generating batches")
tokenize_and_batch(input_files,tokenizer_name, output_dir, batchfile_prefix, max_sequence_length, batch_size, False)
print("batches generated")

# # Create target indices by shifting input indices by one to the right and padding
# target_indices = []
# for seq in input_indices:
#     # Shift and pad: Remove the first element, append pad_token_id
#     if isinstance(pad_token_id, list):
#         target_seq = seq[1:] + pad_token_id
#     else:
#         target_seq = seq[1:] + [pad_token_id]
    
#     # Ensure the target sequence has the same length as the input sequence
#     target_seq = target_seq[:max_seq_length]
    
#     target_indices.append(target_seq)

# Load the input batch
print("loading batch")
input_tensor = mx.load("./output/lazyfox_batch_0001.npy")
target_tensor = mx.load("./output/lazyfox_batch_0001_target.npy")
print("batch loaded")

#target_tensor = mx.array(target_indices)

# # Visualize input sequences
# print("Input:")
# print("")
# seq_util.visualize_sequences(input_indices, tokenizer)

# Visualize target sequences
# print("")
# print("Target:")
# print("")
# seq_util.visualize_sequences(target_indices, tokenizer)

# Reshape the target_tensor to match the expected shape of the logits
#target_tensor = target_tensor.reshape((-1,))

# # Calculate sequence lengths
# lengths = mx.array([len(seq) for seq in input_sequences])

# Calculate sequence lengths
lengths = mx.sum(input_tensor, axis=1)

# Define model parameters
embedding_dim = 32
hidden_size = 32
intermediate_size = 64

# Create an instance of the SimpleLanguageModel
model = SimpleLanguageModel(vocab_size, embedding_dim, hidden_size, intermediate_size)

# Define the optimizer
learning_rate = 0.01
optimizer = optim.Adam(learning_rate=learning_rate)

# Create value and grad function for loss
loss_function = nn.value_and_grad(model, loss)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    # Forward and backward pass
    (lvalue, ntoks), grad = loss_function(model, input_tensor, target_tensor, lengths)

    # Model update
    optimizer.update(model, grad)
    mx.eval(model.parameters(), optimizer.state, lvalue)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {lvalue.item():.4f}, Tokens: {ntoks.item()}")


# # Train the model
# # Create a dataset
# dataset = Dataset(input_sequences, tokenizer)
# trainer = Trainer(model, optimizer, loss_function, lengths)
# trainer.train(dataset, 32)

# Prediction
input_sequence = 'the quick brown'
input_indices = tokenizer.encode(input_sequence, add_special_tokens=False)
input_tensor = mx.array([input_indices])

output = model(input_tensor)
predicted_index = mx.argmax(output[:, -1, :], axis=-1).item()
predicted_word = tokenizer.decode([predicted_index])

print(f"Input sequence: {input_sequence}")
print(f"Predicted next word: {predicted_word}")
