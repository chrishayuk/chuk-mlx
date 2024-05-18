import mlx.core as mx
import models
import models.llama
import models.llama.llama_model
import models.llama.model
from utils.huggingface_utils import load_from_hub
from models.model_config import ModelConfig

# set the model name
model_name = "ibm-granite/granite-3b-code-instruct"

# load the model from huggingface
print(f"Loading Model: {model_name}")
model_path = load_from_hub(model_name)

# load config
model_config = ModelConfig.load(model_path)

# create the model instance
model = models.llama.model.Model(model_config)

# Model Loaded
print(f"Model Loaded: {model_name}")

# Settingss
output_dir = './output'
batchfile_prefix = 'sample'
max_sequence_length = 512
batch_size = 512

# Load the input batch
print("loading batch")
input_tensor = mx.load("./output/sample_batch_0001.npy")
target_tensor = mx.load("./output/sample_batch_0001_target.npy")
print("batch loaded")
