import importlib
import transformers
from utils.huggingface_utils import load_from_hub

import importlib
import transformers
from utils.huggingface_utils import load_from_hub
import logging

logger = logging.getLogger(__name__)

def load_tokenizer(tokenizer_name):
    """
    Load a tokenizer from a specified name. If the tokenizer is in the local 'chuk_tokenizers' folder, load that.
    Otherwise, load from the Hugging Face model repository.
    """
    tokenizer = None
    local_tokenizer_name = tokenizer_name.strip("_tokenizer")
    
    try:
        # Attempt to import the tokenizer module from the 'chuk_tokenizers' directory
        tokenizer_module = importlib.import_module(f"chuk_tokenizers.{local_tokenizer_name}_tokenizer")
        
        # If the module exists and has a class named CustomTokenizer, create an instance
        if hasattr(tokenizer_module, 'CustomTokenizer'):
            logger.info(f"Loading tokenizer locally from 'chuk_tokenizers.{local_tokenizer_name}_tokenizer'")
            tokenizer = tokenizer_module.CustomTokenizer()
        else:
            logger.info(f"No CustomTokenizer class found in the module 'chuk_tokenizers.{local_tokenizer_name}_tokenizer'.")
    except ImportError as e:
        logger.info(f"Local tokenizer module not found: {e}")

    if tokenizer is None:
        try:
            # Load the model from the hub if local loading fails
            logger.info(f"Attempting to load tokenizer from Hugging Face hub: {tokenizer_name}")
            resolved_path = load_from_hub(tokenizer_name)
            tokenizer = transformers.AutoTokenizer.from_pretrained(resolved_path)
        except Exception as e:
            logger.error(f"Error loading tokenizer from Hugging Face hub: {e}")
            raise

    # Check if the tokenizer is LLAMA-based and set the pad_token_id accordingly
    if hasattr(tokenizer, 'eos_token_id'):
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.sep_token_id = tokenizer.eos_token_id
    
    # check pad_token_id is none
    if tokenizer.pad_token_id is None:
        # default to zero
        tokenizer.pad_token_id = 0

    logger.info("Tokenizer loaded successfully")
    return tokenizer
