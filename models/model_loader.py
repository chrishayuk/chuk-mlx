import logging
import mlx.core as mx
from models.architectures.llama.model import Model as LlamaModel
from models.architectures.mistral.model import Model as MistralModel
from models.architectures.gemma.model import Model as GemmaModel
from utils.huggingface_utils import load_from_hub
from utils.tokenizer_loader import load_tokenizer
from models.load_weights import load_checkpoint_weights, load_model_weights
from models.model_config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_model_from_path(model_path):
    # load the model config
    model_config = ModelConfig.load(model_path)

    # Load the correct model based on architecture
    if model_config.architectures[0] == "LlamaForCausalLM":
        return LlamaModel(model_config)
    elif model_config.architectures[0] == "MistralForCausalLM":
        return MistralModel(model_config)
    elif model_config.architectures[0] == "GemmaForCausalLM":
        return GemmaModel(model_config)
    else:
        return LlamaModel(model_config)

def load_model(model_name):
    # get the model path
    model_path = load_from_hub(model_name)    

    # get the model
    model = get_model_from_path(model_path)

    # load the weights
    weights = load_model_weights(model_path)

    # sanitize stuff we don't need
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # load the weights into the model
    model.load_weights(list(weights.items()))

    # evaluate the parameters
    mx.eval(model.parameters())

    # return the model and tokenizer
    return model

def load_model_and_tokenizer(model_name):
    # load the model
    model = load_model(model_name)

    # load the tokenizer
    tokenizer = load_tokenizer(model_name)

    # return the model and tokenizer
    return model, tokenizer

def load_model_tokenizer_and_checkpoint(model_name, checkpoint_path=None, tokenizer_name=None):
    try:
        # load the model from the hub
        model_path = load_from_hub(model_name)
        
         # get the model
        model = get_model_from_path(model_path)

        # Initialize and load model weights
        if checkpoint_path:
            logger.info(f"Loading weights from checkpoint: {checkpoint_path}")
            checkpoint_weights = load_checkpoint_weights(checkpoint_path)
            model.load_weights(list(checkpoint_weights.items()))
        else:
            logger.info(f"Loading initial weights from model path: {model_path}")

            # load the weights
            weights = load_model_weights(model_path)

            # sanitize stuff we don't need
            if hasattr(model, "sanitize"):
                weights = model.sanitize(weights)

            # load the weights into the model
            model.load_weights(list(weights.items()))
        
        # Set model to evaluation mode
        mx.eval(model.parameters())
        
        # Load the tokenizer
        tokenizer_path = tokenizer_name if tokenizer_name else model_name
        tokenizer = load_tokenizer(tokenizer_path)
        
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        raise
