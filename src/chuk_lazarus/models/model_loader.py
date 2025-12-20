import importlib
import os
import logging
from pathlib import Path
import mlx.core as mx
from chuk_lazarus.utils.huggingface import load_from_hub
from chuk_lazarus.utils.tokenizer_loader import load_tokenizer
from chuk_lazarus.models.load_weights import load_checkpoint_weights, load_model_weights
from chuk_lazarus.models.model_config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_LOCAL_CONFIG_PATH = "./model_config"

def get_model_from_path(model_path):
    """ Load the model configuration, tokenizer, and return the appropriate model instance. """
    model_config = ModelConfig.load(model_path / 'config.json')  # Load the model config
    # Convert `model_path` to a string to ensure compatibility with `load_tokenizer`
    tokenizer = load_tokenizer(str(model_path))  # Load the tokenizer from the model path as a string


    # Get the architecture
    architecture = model_config.architectures[0]

    # Conditional import based on the architecture
    if architecture == "LlamaForCausalLM":
        from chuk_lazarus.models.architectures.llama.llama_model import LlamaForCausalLM
        model_class = LlamaForCausalLM
    elif architecture == "GraniteForCausalLM":
        from chuk_lazarus.models.architectures.granite.granite_model import GraniteForCausalLM
        model_class = GraniteForCausalLM
    elif architecture == "MistralForCausalLM":
        from chuk_lazarus.models.architectures.mistral.mistral_model import MistralForCausalLM
        model_class = MistralForCausalLM
    elif architecture == "GemmaForCausalLM":
        from chuk_lazarus.models.architectures.gemma.gemma_model import GemmaForCausalLM
        model_class = GemmaForCausalLM
    elif architecture == "Starcoder2ForCausalLM":
        from chuk_lazarus.models.architectures.starcoder2.starcoder2_for_causal_lm import Starcoder2ForCausalLM
        model_class = Starcoder2ForCausalLM
    elif architecture == "Lazyfox":
        from chuk_lazarus.models.architectures.lazyfox.lazyfox_model import LazyFoxModel
        model_class = LazyFoxModel
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # Pass both model_config and tokenizer to the model
    return model_class(model_config, tokenizer)


def load_model(model_name, load_weights=True):
    """
    Load a model from a specified name. If the model is in the local 'models' folder, load that.
    Otherwise, load from the Hugging Face model repository.
    """
    model = None
    model_module = None
    
    try:
        # Attempt to import the model module from the 'models' directory
        model_module = importlib.import_module(f"core.models.architectures.{model_name}.{model_name}_model")
        logger.debug(f"Model module imported: core.models.architectures.{model_name}.{model_name}_model")

        if hasattr(model_module, 'CustomModel'):
            # Load the model config from the correct path
            config_path = Path(os.path.join(os.path.dirname(model_module.__file__), "config.json"))
            model_config = ModelConfig.load(config_path)
            model = model_module.CustomModel(config=model_config)
            logger.debug(f"Loaded local model: {model_name} from {config_path}")
        else:
            logger.debug(f"No CustomModel class found in the module 'core.models.architectures.{model_name}.{model_name}_model'.")
    except ImportError as e:
        logger.info(f"Local model module not found: {e}")

    if model is None:
        try:
            # Load the model from the hub if local loading fails
            logger.debug(f"Attempting to load model from Hugging Face hub: {model_name}")
            resolved_path = load_from_hub(model_name)
            config_path = Path(resolved_path) / "config.json"
            model_config = ModelConfig.load(config_path)
            model = get_model_from_path(resolved_path)
        except Exception as e:
            logger.error(f"Error loading model from Hugging Face hub: {e}")
            raise

    if model is None:
        raise FileNotFoundError(f"Model {model_name} could not be loaded from local path or Hugging Face hub.")

    # Load weights if necessary
    if load_weights:
        logger.debug(f"Loading model weights for: {model_name}")
        if model_module:
            weights = load_model_weights(Path(os.path.dirname(model_module.__file__)))
        else:
            weights = load_model_weights(Path(resolved_path))
        
        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)
        
        model.load_weights(list(weights.items()))

    # Set model to evaluation mode
    mx.eval(model.parameters())
    
    return model


def load_model_tokenizer_and_checkpoint(model_name, checkpoint_path=None, tokenizer_name=None, local_config_path=DEFAULT_LOCAL_CONFIG_PATH):
    """ Load the model, tokenizer, and optionally, weights from a checkpoint, preferring a local config path if provided and exists. """
    try:
        local_model_path = Path(local_config_path) / model_name

        if local_model_path.exists():
            model_path = local_model_path
        else:
            model_path = load_from_hub(model_name)

        # Get the model and tokenizer
        model = get_model_from_path(model_path)
        
        inv_freq = None
        if checkpoint_path:
            logger.debug(f"Loading weights from checkpoint: {checkpoint_path}")
            checkpoint_weights = load_checkpoint_weights(checkpoint_path)
            for k, v in checkpoint_weights.items():
                if 'rotary_emb.inv_freq' in k:
                    inv_freq = v
                    break
            model.load_weights(list(checkpoint_weights.items()))
        else:
            logger.debug(f"Loading initial weights from model path: {model_path}")
            weights = load_model_weights(model_path)
            
            if weights:
                for k, v in weights.items():
                    if 'rotary_emb.inv_freq' in k:
                        inv_freq = v
                        break
                if hasattr(model, "sanitize"):
                    weights = model.sanitize(weights)
                model.load_weights(list(weights.items()))
            else:
                logger.warning("No weights found. Initializing model with random weights.")
        
        if inv_freq is not None:
            if hasattr(model, "set_inv_freq"):
                model.set_inv_freq(inv_freq)
                logger.debug("Inverse frequencies for rotary embeddings loaded from weights.")
            else:
                logger.debug("Model doesn't have set_inv_freq method. Unable to set inverse frequencies.")
        else:
            logger.debug("No 'rotary_emb.inv_freq' found in weights. Using default initialization.")
        
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
