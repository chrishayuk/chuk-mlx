from models.model_config import ModelConfig
from .swiglu_mlp import MLP as SwiGluMLP
from .gelu_glu_mlp import MLP as GeluGluMLP

def create_mlp(config: ModelConfig) -> SwiGluMLP | GeluGluMLP:
    """
    Factory function to create the appropriate MLP based on the model configuration.

    Args:
        config (ModelConfig): The model configuration object.

    Returns:
        SwiGluMLP | GeluGluMLP: An instance of the appropriate MLP.

    Raises:
        ValueError: If an unsupported activation function is specified.
    """
    # llama models use a SWIGlu activation function instead of RELU
    # Llama-Paper
    # We replace the ReLU non-linearity by the SwiGLU activation function
    # introduced by Shazeer (2020) to improve the performance.
    # We use a dimension 2/3 4d instead of 4d as in PaLM.
    if config.hidden_act == "silu":
        # use swiglu mlp
        return SwiGluMLP(config.hidden_size, config.intermediate_size, config.mlp_bias)
    elif config.hidden_act == "gelu":
        # use geluglu mlp
        return GeluGluMLP(config.hidden_size, config.intermediate_size, config.mlp_bias)
    else:
        # unsupported
        raise ValueError(f"Unsupported activation function: {config.hidden_act}")