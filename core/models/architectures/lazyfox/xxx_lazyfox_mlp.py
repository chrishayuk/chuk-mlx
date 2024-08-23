from graphviz import Digraph
from core.models.architectures.model import Model
from core.models.model_config import ModelConfig

def visualize_model(model):
    dot = Digraph()

    # Add the input node
    dot.node('Input', 'Input Layer')

    # Add the embedding layer
    dot.node('Embedding', f'Embedding\n(vocab_size -> hidden_size)')
    dot.edge('Input', 'Embedding')

    # Add the MLP layers
    mlp_layers = [
        ('gate_proj', model.mlp.gate_proj),
        ('up_proj', model.mlp.up_proj),
        ('down_proj', model.mlp.down_proj)
    ]

    previous_node = 'Embedding'

    for name, layer in mlp_layers:
        # Add the MLP layer node
        layer_label = f"{name}: {type(layer).__name__}"
        dot.node(name, layer_label)
        dot.edge(previous_node, name)
        previous_node = name

    # Add the output node
    dot.node('Output', 'Output Layer')
    dot.edge(previous_node, 'Output')

    return dot

# Example usage:
config = ModelConfig(vocab_size=10000, hidden_size=128, intermediate_size=256)
model = Model(config)
graph = visualize_model(model)
graph.render("model_architecture", format="png")