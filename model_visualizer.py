from graphviz import Digraph
from core.models.model_loader import load_model

def visualize_model(model):
    dot = Digraph()

    # Add the input node
    dot.node('Input', 'Input Layer')

    # Add the embedding layer node separately
    dot.node('Embedding', 'Embedding\n(vocab_size -> hidden_size)')
    dot.edge('Input', 'Embedding')

    # Create a subgraph to represent the MLP as a separate layer
    with dot.subgraph(name='cluster_MLP') as mlp:
        mlp.attr(label='MLP Layer', color='blue')

        # Define the MLP layers
        mlp.node('gate_proj', 'gate_proj: Linear')
        mlp.node('up_proj', 'up_proj: Linear')
        mlp.node('down_proj', 'down_proj: Linear')

        # Connect the MLP layers inside the subgraph
        mlp.edge('gate_proj', 'up_proj')
        mlp.edge('up_proj', 'down_proj')

    # Connect the embedding layer to the first MLP layer outside of the subgraph
    dot.edge('Embedding', 'gate_proj')

    # Connect the last MLP layer to the output layer
    dot.node('Output', 'Output Layer')
    dot.edge('down_proj', 'Output')

    return dot

# Example usage:
model_name = "ibm-granite/granite-3b-code-instruct" #"lazyfox"
model = load_model(model_name, load_weights=False)
graph = visualize_model(model)
graph.render("model_architecture", format="png")
