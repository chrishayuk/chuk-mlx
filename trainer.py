import mlx.core as mx
import mlx.nn as nn
import random
from models.loss_function import loss

import math

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            random.shuffle(self.indices)

    def __iter__(self):
        batch = []
        for index in self.indices:
            batch.append(self.dataset[index])
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
    
class Trainer:
    def __init__(self, model, optimizer, loss_function, max_sequence_length):
        # set the model
        self.model = model

        # set the optimizer
        self.optimizer = optimizer

        # set the loss function
        self.loss_function = loss_function

        # set the max sequence length
        self.max_sequence_length = max_sequence_length

        # epochs
        self.num_epochs = 7

    def train(self, dataset, batch_size):
        # Create a data loader
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Create value and grad function for loss
        loss_value_and_grad = nn.value_and_grad(self.model, loss)

        # loop through each epoch
        for epoch in range(self.num_epochs):
            # Initialize total loss and tokens for the epoch
            total_loss = 0
            total_tokens = 0

            # Iterate over batches
            for batch in data_loader:
                # Convert batch to tensor
                input_batch = mx.array(batch)

                # Create target batch by shifting input batch by one
                target_batch = mx.array([seq[1:] + [self.tokenizer.pad_token_id] for seq in batch])

                # Calculate sequence lengths
                lengths_batch = mx.array([len(seq) for seq in batch])

                # Forward and backward pass
                (lvalue, ntoks), grad = loss_value_and_grad(self.model, input_batch, target_batch, lengths_batch)

                # Model update
                self.optimizer.update(self.model, grad)
                mx.set_parameters(self.model, self.optimizer.state)

                # Accumulate loss and tokens
                total_loss += lvalue.item()
                total_tokens += ntoks.item()

            # Calculate average loss for the epoch
            avg_loss = total_loss / len(data_loader)

            # print out the result
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}, Tokens: {total_tokens}")
