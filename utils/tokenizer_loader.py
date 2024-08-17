import importlib
import os
import transformers
from utils.huggingface_utils import load_from_hub
import logging

# set the logger
logger = logging.getLogger(__name__)

def load_tokenizer(tokenizer_name):
    """
    Load a tokenizer from a specified name. If a vocabulary file is found in the 'model_configuration' folder,
    load the local tokenizer. Otherwise, load from the Hugging Face model repository.
    """
    tokenizer = None
    model_config_dir = f"model_configuration/{tokenizer_name}"
    vocab_file = os.path.join(model_config_dir, "tokenizer.json")
    
    if os.path.exists(vocab_file):
        try:
            # Load the local tokenizer using the vocab file
            logger.debug(f"Loading tokenizer locally from '{vocab_file}'")
            from chuk_tokenizers.lazyfox_tokenizer import CustomTokenizer
            tokenizer = CustomTokenizer(vocab_file=vocab_file)
        except Exception as e:
            logger.error(f"Error loading local tokenizer: {e}")
            raise
    else:
        try:
            # Load the tokenizer from the Hugging Face hub if local loading fails
            logger.debug(f"Attempting to load tokenizer from Hugging Face hub: {tokenizer_name}")
            resolved_path = load_from_hub(tokenizer_name)
            tokenizer = transformers.AutoTokenizer.from_pretrained(resolved_path)
        except Exception as e:
            logger.error(f"Error loading tokenizer from Hugging Face hub: {e}")
            raise

    # Check if the tokenizer is LLAMA-based or has eos_token_id, and set the pad_token_id accordingly
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.sep_token_id = tokenizer.eos_token_id
    
    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        logger.debug("pad_token_id is None. Defaulting to 0.")
        tokenizer.pad_token_id = 0

    # success
    logger.debug("Tokenizer loaded successfully")
    return tokenizer