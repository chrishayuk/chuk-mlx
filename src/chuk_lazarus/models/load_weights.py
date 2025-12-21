# mlx_model_weights_loader.py
import logging
import os
from pathlib import Path

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


def load_model_weights(model_path):
    model_path = Path(model_path)
    safetensors_files = list(model_path.glob("*.safetensors"))
    bin_files = list(model_path.glob("*.bin"))

    if safetensors_files:
        return load_safetensors(safetensors_files)
    elif bin_files:
        if len(bin_files) == 1:
            return load_bin_weights(bin_files[0])
        else:
            return load_sharded_bin_weights(bin_files)
    else:
        logger.warning(f"No weight files found in {model_path}")
        logger.info(f"Available files: {os.listdir(model_path)}")
        return {}  # Return an empty dict instead of raising an error


def load_sharded_bin_weights(bin_files):
    weights = {}
    for file in sorted(bin_files):
        weights.update(load_bin_weights(file))
    return weights


def load_checkpoint_weights(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.is_file():
        if checkpoint_path.suffix == ".safetensors":
            return load_safetensors([checkpoint_path])
        elif checkpoint_path.suffix == ".bin":
            return load_bin_weights(checkpoint_path)
        else:
            with open(checkpoint_path, "rb") as f:
                return mx.load(f)
    else:
        raise ValueError(f"Checkpoint path {checkpoint_path} is not a file.")


def load_safetensors(weight_files):
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(str(wf)).items())
    return weights


def load_bin_weights(bin_file):
    # This function needs to be implemented to load .bin files without using PyTorch
    # Here's a basic implementation that assumes the .bin file is a simple format
    # You may need to adjust this based on the actual format of your .bin files
    with open(bin_file, "rb") as f:
        data = f.read()

    # Assume the first 8 bytes contain the number of tensors
    num_tensors = int.from_bytes(data[:8], byteorder="little")
    offset = 8

    weights = {}
    for _ in range(num_tensors):
        # Read the length of the key
        key_length = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4

        # Read the key
        key = data[offset : offset + key_length].decode("utf-8")
        offset += key_length

        # Read the shape of the tensor
        shape_length = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        shape = tuple(
            int.from_bytes(data[offset + i * 4 : offset + (i + 1) * 4], byteorder="little")
            for i in range(shape_length)
        )
        offset += shape_length * 4

        # Read the data type
        dtype_length = int.from_bytes(data[offset : offset + 4], byteorder="little")
        offset += 4
        dtype = data[offset : offset + dtype_length].decode("utf-8")
        offset += dtype_length

        # Read the tensor data
        tensor_size = np.prod(shape) * np.dtype(dtype).itemsize
        tensor_data = np.frombuffer(data[offset : offset + tensor_size], dtype=dtype).reshape(shape)
        offset += tensor_size

        # Convert to MLX array
        weights[key] = mx.array(tensor_data)

    return weights
