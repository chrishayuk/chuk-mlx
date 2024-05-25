import mlx.core as mx
import models.llama.model
from utils.huggingface_utils import load_from_hub
from utils.tokenizer_loader import load_tokenizer
from models.load_weights import load_model_weights
from models.model_config import ModelConfig

def load_model(model_name):
    # get the model path
    model_path = load_from_hub(model_name)
    
    # load the model config
    model_config = ModelConfig.load(model_path)

    # use the llama model (will change in future)
    model = models.llama.model.Model(model_config)

    # load the weights
    weights = load_model_weights(model_path)
    model.load_weights(list(weights.items()))

    # evaluate the parameters
    mx.eval(model.parameters())

    # return the model and tokenizer
    return model

def load_model_and_tokenizer(model_name):
    # get the model path
    model_path = load_from_hub(model_name)
    
    # load the model config
    model_config = ModelConfig.load(model_path)

    # use the llama model (will change in future)
    model = models.llama.model.Model(model_config)

    # load the weights
    weights = load_model_weights(model_path)
    model.load_weights(list(weights.items()))

    # evaluate the parameters
    mx.eval(model.parameters())

    # load the tokenizer
    tokenizer = load_tokenizer(model_name)

    # return the model and tokenizer
    return model, tokenizer