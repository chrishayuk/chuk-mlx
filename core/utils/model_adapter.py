import torch
import mlx.core as mx
import mlx.nn as nn

class ModelAdapter:
    def __init__(self, framework='mlx', model=None):
        self.framework = framework
        self.model = model

    def to_tensor(self, data):
        if self.framework == 'torch':
            return torch.tensor(data)
        elif self.framework == 'mlx':
            return mx.array(data)

    def forward(self, input_tensor):
        if self.framework == 'torch':
            with torch.no_grad():
                return self.model(input_tensor).numpy()
        elif self.framework == 'mlx':
            return self.model(input_tensor)

    def argmax(self, output, axis=-1):
        if self.framework == 'torch':
            return torch.argmax(torch.tensor(output), dim=axis).tolist()
        elif self.framework == 'mlx':
            return mx.argmax(mx.array(output), axis=axis).tolist()

    def create_value_and_grad_fn(self, loss_function):
        if self.framework == 'torch':
            def value_and_grad(inputs, targets):
                outputs = self.model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                return loss.item(), [param.grad for param in self.model.parameters()]
            return value_and_grad
        elif self.framework == 'mlx':
            return nn.value_and_grad(self.model, loss_function)

    def load_tensor_from_file(self, batch_path):
        """Load batch data based on the framework"""
        if self.framework == 'torch':
            # Assuming a torch-compatible format for batch loading
            # Replace this with the actual torch data loading if needed
            return torch.load(batch_path)
        elif self.framework == 'mlx':
            # Load the batch data using mlx
            return mx.load(batch_path)
