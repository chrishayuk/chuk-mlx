# mlx_model_weights_loader.py
import glob
import mlx.core as mx

def load_weight_files(model_path):
    # get the path to the weights file
    weight_files = glob.glob(str(model_path / "*.safetensors"))

    # check the for weight files
    if len(weight_files) == 0:
        raise FileNotFoundError("No safetensors found in {}".format(model_path))

    # load the weights
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf).items())

    # return the weights
    return weights
