import numpy as np

class MockEmbedding:
    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.weights = np.random.random((vocab_size, hidden_size)).astype(np.float32)

    def __call__(self, x):
        # Ensure x is integer type for indexing
        x = np.array(x, dtype=np.int32)
        # Mock embedding lookup
        return self.weights[x]

class MockMLP:
    def __init__(self, hidden_size, intermediate_size):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.weights1 = np.random.random((hidden_size, intermediate_size)).astype(np.float32)
        self.weights2 = np.random.random((intermediate_size, hidden_size)).astype(np.float32)

    def __call__(self, x):
        # Mock forward pass through MLP
        intermediate_output = np.dot(x, self.weights1)
        output = np.dot(intermediate_output, self.weights2)
        return output

class MockModel:
    def __init__(self, vocab_size=10000, hidden_size=256, intermediate_size=512):
        self.embedding = MockEmbedding(vocab_size, hidden_size)
        self.mlp = MockMLP(hidden_size, intermediate_size)
        self.parameters = [self.embedding.weights, self.mlp.weights1, self.mlp.weights2]
        self.trainable_parameters = self.parameters  # Mock trainable parameters should be a list of numpy arrays

    def __call__(self, inputs):
        # Ensure inputs are integer type for embedding lookup
        inputs = np.array(inputs, dtype=np.int32)
        # Simulate a forward pass
        embedded_vectors = self.embedding(inputs)
        output = self.mlp(embedded_vectors)
        return output

    def save_weights(self, file_path):
        # Simulate saving model weights
        np.savez(file_path, embedding=self.embedding.weights, mlp1=self.mlp.weights1, mlp2=self.mlp.weights2)
