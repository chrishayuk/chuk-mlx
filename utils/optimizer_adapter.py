# utils/optimizer_adapter.py

import mlx.optimizers as mlx_optim
import torch.optim as torch_optim

class OptimizerAdapter:
    def __init__(self, framework='mlx'):
        self.framework = framework
        self.optimizer = None

    def create_optimizer(self, model_parameters, optimizer_name='Adam', **kwargs):
        if self.framework == 'mlx':
            if optimizer_name == 'Adam':
                self.optimizer = mlx_optim.Adam(model_parameters, **kwargs)
            elif optimizer_name == 'SGD':
                self.optimizer = mlx_optim.SGD(model_parameters, **kwargs)
            # Add other optimizers as needed
        elif self.framework == 'torch':
            if optimizer_name == 'Adam':
                self.optimizer = torch_optim.Adam(model_parameters, **kwargs)
            elif optimizer_name == 'SGD':
                self.optimizer = torch_optim.SGD(model_parameters, **kwargs)

        return self.optimizer

    def step(self):
        if self.optimizer is not None:
            self.optimizer.step()

    def zero_grad(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()
