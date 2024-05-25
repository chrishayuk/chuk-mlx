# mlx_model_weights_loader.py
import glob
from pathlib import Path
import mlx.core as mx

def load_model_weights(model_path):
    # get the path to the weights file
    weight_files = glob.glob(str(model_path / "*.safetensors"))

    # check the for weight files
    if len(weight_files) == 0:
        raise FileNotFoundError("No safetensors found in {}".format(model_path))

    # load the weights
    return load_weights(weight_files)

def load_checkpoint_weights(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)

    # Check if it's a file
    if checkpoint_path.is_file():
        # Open the file and load the weights using mx
        with open(checkpoint_path, 'rb') as f:
            return mx.load(f)
    else:
        raise ValueError(f"Checkpoint path {checkpoint_path} is not a file.")

def load_weights(weight_files):
    # load the weights
    weights = {}

    # loop through the weights files
    for wf in weight_files:
        # load the weights
        weights.update(mx.load(wf).items())

    # return the weights
    return weights

