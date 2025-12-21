# utils/optimizer_adapter.py

import mlx.optimizers as mlx_optim

# Optional torch import for cross-framework support
try:
    import torch.optim as torch_optim

    HAS_TORCH = True
except ImportError:
    torch_optim = None
    HAS_TORCH = False


class OptimizerAdapter:
    def __init__(self, framework="mlx"):
        if framework == "torch" and not HAS_TORCH:
            raise ImportError(
                "torch is required for torch framework. Install with: pip install torch"
            )
        self.framework = framework
        self.optimizer = None

    def create_optimizer(self, model_parameters, optimizer_name="Adam", **kwargs):
        if self.framework == "mlx":
            if optimizer_name == "Adam":
                self.optimizer = mlx_optim.Adam(model_parameters, **kwargs)
            elif optimizer_name == "SGD":
                self.optimizer = mlx_optim.SGD(model_parameters, **kwargs)
        elif self.framework == "torch":
            if optimizer_name == "Adam":
                self.optimizer = torch_optim.Adam(model_parameters, **kwargs)
            elif optimizer_name == "SGD":
                self.optimizer = torch_optim.SGD(model_parameters, **kwargs)

        return self.optimizer

    def step(self):
        if self.optimizer is not None:
            self.optimizer.step()

    def zero_grad(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()
