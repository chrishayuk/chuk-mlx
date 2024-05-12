import glob
import mlx.core as mx
import mlx.nn as nn
import models
import models.llama
import models.llama.llama_model
from models.load_weights import load_model_weights
import models.model
from models.model_config import ModelConfig
from utils.huggingface_utils import load_from_hub

# set the model name
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# load the model from huggingface
print(f"Loading Model: {model_name}")
model_path = load_from_hub(model_name)

# load config
model_config = ModelConfig.load(model_path)

# load the model weights
weights = load_model_weights(model_path)

# create the model instance
model = models.model.Model(model_config)

# Model Loaded
print(f"Model Loaded: {model_name}")


# loading weights
print("loading weights")
model.load_weights(list(weights.items()))
mx.eval(model.parameters())
print("weights loaded")

# Create value and grad function for loss
#loss_value_and_grad = nn.value_and_grad(model, loss)