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
    """
    Load the model configuration and return the appropriate model instance.
    """
    model_config = ModelConfig.load(model_path / 'config.json')  # Assume config.json is the config file
    architecture = model_config.architectures[0]
    
    if architecture == "LlamaForCausalLM":
        from models.architectures.llama.llama_for_causal_lm import LlamaForCausalLM
        model_class = LlamaForCausalLM
    elif architecture == "MistralForCausalLM":
        from models.architectures.mistral.mistral_for_causal_lm import MistralForCausalLM
        model_class = MistralForCausalLM
    elif architecture == "GemmaForCausalLM":
        from models.architectures.gemma.gemma_for_causal_lm import GemmaForCausalLM
        model_class = GemmaForCausalLM
    elif architecture == "SimpleLanguageModel":
        from models.architectures.lazyfox.simple_language_model import SimpleLanguageModel
        model_class = SimpleLanguageModel
    else:
        from models.architectures.llama.llama_for_causal_lm import LlamaForCausalLM
        model_class = LlamaForCausalLM

    return model_class(model_config)

def load_model(model_name, local_config_path=DEFAULT_LOCAL_CONFIG_PATH, load_weights=True):
    """
    Load the model from a local path if available, otherwise from the Hugging Face hub and initialize weights.
    """
    model_dir = Path(local_config_path) / model_name
    if model_dir.exists():
        model_path = model_dir
    else:
        model_path = load_from_hub(model_name)
        
    model = get_model_from_path(model_path)

    if load_weights:
        weights = load_model_weights(model_path)
        
        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)
        
        model.load_weights(list(weights.items()))
        
    mx.eval(model.parameters())
    
    return model

def load_model_and_tokenizer(model_name, local_config_path=DEFAULT_LOCAL_CONFIG_PATH, load_weights=True):
    """
    Load the model and tokenizer, preferring a local config path if provided and exists.
    """
    model = load_model(model_name, local_config_path, load_weights)
    tokenizer = load_tokenizer(model_name)
    
    return model, tokenizer

def load_model_tokenizer_and_checkpoint(model_name, checkpoint_path=None, tokenizer_name=None, local_config_path=DEFAULT_LOCAL_CONFIG_PATH):
    """
    Load the model, tokenizer, and optionally, weights from a checkpoint, preferring a local config path if provided and exists.
    """
    try:
        model_dir = Path(local_config_path) / model_name
        if model_dir.exists():
            model_path = model_dir
        else:
            model_path = load_from_hub(model_name)
        
        model = get_model_from_path(model_path)
        
        if checkpoint_path:
            logger.info(f"Loading weights from checkpoint: {checkpoint_path}")
            checkpoint_weights = load_checkpoint_weights(checkpoint_path)
            model.load_weights(list(checkpoint_weights.items()))
        else:
            logger.info(f"Loading initial weights from model path: {model_path}")
            weights = load_model_weights(model_path)
            
            if hasattr(model, "sanitize"):
                weights = model.sanitize(weights)
            
            model.load_weights(list(weights.items()))
        
        mx.eval(model.parameters())
        
        tokenizer_path = tokenizer_name if tokenizer_name else model_name
        tokenizer = load_tokenizer(tokenizer_path)
        
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
    
    except FileNotFoundError as fnf_error:
        logger.error(f"File not found: {fnf_error}")
        raise
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        raise
