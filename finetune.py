import mlx.core as mx
import models
import models.llama
import models.llama.llama_model
import models.llama.model
from models.load_weights import load_model_weights
from models.model_config import ModelConfig
from utils.huggingface_utils import load_from_hub
from batches.llama_finetune_batch import LLaMAFineTuneBatch

# set the model name
#model_name = "mistralai/Mistral-7B-Instruct-v0.2"
#model_name = "ibm/granite-7b-base"
#model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = "ibm-granite/granite-3b-code-instruct"

# load the model from huggingface
print(f"Loading Model: {model_name}")
model_path = load_from_hub(model_name)

# load config
model_config = ModelConfig.load(model_path)

# load the model weights
weights = load_model_weights(model_path)

# create the model instance
model = models.llama.model.Model(model_config)

# Model Loaded
print(f"Model Loaded: {model_name}")

# loading weights
print("loading weights")
model.load_weights(list(weights.items()))
mx.eval(model.parameters())
print("weights loaded")

# Generate a batch
input_files = ['./sample_data/calvin_scale_llama/train.jsonl']
output_dir = './output'
batchfile_prefix = 'calvin'
max_sequence_length = 512
batch_size = 512

# Generate batches for LLaMA fine-tuning
print("generating LLaMA fine-tuning batches")
llama_finetune_batching = LLaMAFineTuneBatch(model_name, output_dir, batchfile_prefix, max_sequence_length, batch_size, False)
llama_finetune_batching.tokenize_and_batch(input_files)
print("LLaMA fine-tuning batches generated")

# Load the input batch
print("loading batch")
input_tensor = mx.load("./output/calvin_batch_0001.npy")
target_tensor = mx.load("./output/calvin_batch_0001_target.npy")
print("batch loaded")

# # Create value and grad function for loss
# loss_function = nn.value_and_grad(model, loss)

# # Define the optimizer
# learning_rate = 0.01
# optimizer = optim.Adam(learning_rate=learning_rate)

# # Training loop
# num_epochs = 50
# losses = []
# for epoch in range(num_epochs):
#     # Forward and backward pass
#     (lvalue, toks), grad = loss_function(model, *batch)

#     # Model update
#     optimizer.update(model, grad)
#     mx.eval(model.parameters(), optimizer.state, lvalue)

#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {lvalue.item():.4f}, Tokens: {ntoks.item()}")


# # Generate response for the prompt
# print("Prompting")
# prompt = "Weite a fibonacci function in python?"
# response = generate_response(model, prompt, tokenizer)
# #print(response)
