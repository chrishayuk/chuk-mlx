import os
import logging
from pathlib import Path
import mlx.core as mx
from utils.huggingface_utils import load_from_hub
from utils.tokenizer_loader import load_tokenizer
from models.load_weights import load_checkpoint_weights, load_model_weights
from models.model_config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_LOCAL_CONFIG_PATH = "./model_config"

def get_model_from_path(model_path):
    """ Load the model configuration and return the appropriate model instance. """
    model_config = ModelConfig.load(model_path / 'config.json')  # Assume config.json is the config file
    
    # get the architecture
    architecture = model_config.architectures[0]
    
    print(architecture)
    # a giant if..elif statement switchign architecture
    if architecture == "LlamaForCausalLM":
        from models.architectures.llama.llama_model import LlamaForCausalLM
        model_class = LlamaForCausalLM
    elif architecture == "MistralForCausalLM":
        from models.architectures.mistral.mistral_model import MistralForCausalLM
        model_class = MistralForCausalLM
    elif architecture == "GemmaForCausalLM":
        from models.architectures.gemma.gemma_model import GemmaForCausalLM
        model_class = GemmaForCausalLM
    elif architecture == "Starcoder2ForCausalLM":
        print('hello')
        from models.architectures.starcoder2.starcoder2_for_causal_lm import Starcoder2ForCausalLM
        model_class = Starcoder2ForCausalLM
    elif architecture == "SimpleLanguageModel":
        from models.architectures.lazyfox.simple_language_model import SimpleLanguageModel
        model_class = SimpleLanguageModel
    else:
        from models.architectures.llama.llama_model import LlamaForCausalLM
        model_class = LlamaForCausalLM

    return model_class(model_config)

def load_model(model_name, local_config_path=DEFAULT_LOCAL_CONFIG_PATH, load_weights=True):
    """ Load the model from a local path if available, otherwise from the Hugging Face hub and initialize weights."""

    # get the local path of the model
    local_model_path = Path(local_config_path) / model_name

    # check if we have the model locally
    if local_model_path.exists():
        model_path = local_model_path
    else:
        # nope, so get from hub
        model_path = load_from_hub(model_name)

    # get the model locally     
    model = get_model_from_path(model_path)

    # check if we need to load weights
    if load_weights:
        # load weights
        weights = load_model_weights(model_path)
        
        # if we have a sanitizer, sanitize
        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)
        
        # load the weights
        model.load_weights(list(weights.items()))

    # eval  
    mx.eval(model.parameters())
    
    return model

def load_model_and_tokenizer(model_name, local_config_path=DEFAULT_LOCAL_CONFIG_PATH, load_weights=True):
    """ Load the model and tokenizer, preferring a local config path if provided and exists. """
    model = load_model(model_name, local_config_path, load_weights)
    tokenizer = load_tokenizer(model_name)
    
    return model, tokenizer

def load_model_tokenizer_and_checkpoint(model_name, checkpoint_path=None, tokenizer_name=None, local_config_path=DEFAULT_LOCAL_CONFIG_PATH):
    """ Load the model, tokenizer, and optionally, weights from a checkpoint, preferring a local config path if provided and exists. """
    try:
        # Get the local model path
        local_model_path = Path(local_config_path) / model_name

        # If we have it locally, load
        if local_model_path.exists():
            # Set the path as local
            model_path = local_model_path
        else:
            # Load from hub
            model_path = load_from_hub(model_name)
        
        # Get the model
        model = get_model_from_path(model_path)
        
        inv_freq = None
        # Check if we have a checkpoint path
        if checkpoint_path:
            # Load weights from checkpoint, into the model
            logger.info(f"Loading weights from checkpoint: {checkpoint_path}")
            checkpoint_weights = load_checkpoint_weights(checkpoint_path)
            for k, v in checkpoint_weights.items():
                if 'rotary_emb.inv_freq' in k:
                    inv_freq = v
                    break
            model.load_weights(list(checkpoint_weights.items()))
        else:
            # Use the model weights
            logger.info(f"Loading initial weights from model path: {model_path}")
            weights = load_model_weights(model_path)
            
            # Check if weights were loaded successfully
            if weights:
                for k, v in weights.items():
                    if 'rotary_emb.inv_freq' in k:
                        inv_freq = v
                        break
                # Sanitize
                if hasattr(model, "sanitize"):
                    weights = model.sanitize(weights)
                
                # Load weights
                model.load_weights(list(weights.items()))
            else:
                logger.warning("No weights found. Initializing model with random weights.")
        
        # Set the inverse frequencies if they were found
        if inv_freq is not None:
            if hasattr(model, "set_inv_freq"):
                model.set_inv_freq(inv_freq)
                logger.info("Inverse frequencies for rotary embeddings loaded from weights.")
            else:
                logger.warning("Model doesn't have set_inv_freq method. Unable to set inverse frequencies.")
        else:
            logger.info("No 'rotary_emb.inv_freq' found in weights. Using default initialization.")
        
        # Eval
        mx.eval(model.parameters())
        
        # Tokenize
        tokenizer_path = tokenizer_name if tokenizer_name else model_name
        tokenizer = load_tokenizer(tokenizer_path)

        # Loaded
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
    
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found: {fnf_error}")
        raise
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        raise