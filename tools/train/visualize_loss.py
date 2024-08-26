import typer
import matplotlib.pyplot as plt
import numpy as np

def compute_loss(inputs: list[float], targets: list[float]) -> float:
    inputs = np.array(inputs)
    targets = np.array(targets)
    loss = np.mean((inputs - targets) ** 2)  # Simple MSE loss for demonstration
    return loss

def main(
    inputs: str = typer.Option(..., help="Comma-separated list of input values."),
    targets: str = typer.Option(..., help="Comma-separated list of target values.")
):
    # Convert the comma-separated strings to lists of floats
    inputs_list = [float(x) for x in inputs.split(",")]
    targets_list = [float(x) for x in targets.split(",")]

    # Compute the loss
    loss = compute_loss(inputs_list, targets_list)
    print(f"Computed Loss: {loss}")
    
    # Visualization (simple line plot for example)
    plt.plot(inputs_list, label="Inputs", marker='o')
    plt.plot(targets_list, label="Targets", marker='x')
    plt.legend()
    plt.title(f"Loss Visualization (Loss: {loss})")
    plt.show()

if __name__ == "__main__":
    typer.run(main)
