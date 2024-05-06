import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from models.loss_function import loss
from models.mlx.simple_language_model import SimpleLanguageModel
from utils.tokenizer_loader import load_tokenizer
from utils.sequence_utility import SequenceUtility

# Load tokenizer and define vocabulary size
tokenizer = load_tokenizer('lazyfox_tokenizer')
vocab_size = len(tokenizer.vocab)
pad_token_id = [0]#tokenizer.pad_token_id

# Example input sequences
input_sequences = [
    'the quick brown fox jumps over the lazy dog',
        'the quick brown fox jumps over the lazy',
        'the quick brown fox jumps over the',
        'the quick brown fox jumps over',
        'the quick brown fox jumps',
        'the quick brown fox',
        'the quick brown',
        'the quick',
        'the',
        'quick brown fox jumps over the lazy dog',
        'quick brown fox jumps over the lazy',
        'quick brown fox jumps over the',
        'quick brown fox jumps',
        'quick brown fox',
        'quick brown',
        'brown fox jumps over the lazy dog',
        'brown fox jumps over the lazy',
        'brown fox jumps over the',
        'brown fox jumps',
        'brown fox',
        'fox jumps over the lazy dog',
        'fox jumps over the lazy',
        'fox jumps over the',
        'fox jumps',
        'jumps over the lazy dog',
        'jumps over the lazy',
        'jumps over',
        'over the lazy dog',
        'over the lazy',
        'over the',
        'the lazy dog',
        'the lazy',
        'lazy dog'
]

# Convert input sequences to index sequences
input_indices = [tokenizer.encode(seq, add_special_tokens=False) for seq in input_sequences]

# Determine the maximum length of input sequences for padding
max_seq_length = max(len(seq) for seq in input_indices)

# Instantiate SequenceUtility
seq_util = SequenceUtility(max_seq_length=max_seq_length, padding_value=pad_token_id)

# Pad input indices to make all sequences of the same length
input_indices = seq_util.batch_sequences(input_indices)

# Create target indices by shifting input indices by one to the right and padding
target_indices = []
for seq in input_indices:
    # Shift and pad: Remove the first element, append pad_token_id
    target_seq = seq[1:] + pad_token_id  # Ensure no additional brackets are around pad_token_id
    target_indices.append(target_seq)

# Pad target indices to make all sequences of the same lengths
target_indices_padded = []
for seq in target_indices:
    seq_padded = seq + pad_token_id * (max_seq_length - len(seq))
    target_indices_padded.append(seq_padded)
target_indices = target_indices_padded

# print(input_indices)
# print(target_indices)
# # Visualize input sequences
# print("Input:")
# print("")
# seq_util.visualize_sequences(input_indices, tokenizer)

# # Visualize target sequences
# print("")
# print("Target:")
# print("")
# seq_util.visualize_sequences(target_indices, tokenizer)

# Convert lists to tensors
input_tensor = mx.array(input_indices)
target_tensor = mx.array(target_indices)

# Calculate sequence lengths
lengths = mx.array([len(seq) for seq in input_sequences])

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
loss_value_and_grad = nn.value_and_grad(model, loss)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    # Forward and backward pass
    (lvalue, ntoks), grad = loss_value_and_grad(model, input_tensor, target_tensor, lengths)

    # Model update
    optimizer.update(model, grad)
    mx.eval(model.parameters(), optimizer.state, lvalue)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {lvalue.item():.4f}, Tokens: {ntoks.item()}")

# Prediction
input_sequence = 'fox jumps'
input_indices = tokenizer.encode(input_sequence, add_special_tokens=False)
input_tensor = mx.array([input_indices])

output = model(input_tensor)
predicted_index = mx.argmax(output[:, -1, :], axis=-1).item()
predicted_word = tokenizer.decode([predicted_index])

print(f"Input sequence: {input_sequence}")
print(f"Predicted next word: {predicted_word}")
