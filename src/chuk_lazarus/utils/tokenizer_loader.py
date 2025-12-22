import logging
import os

import transformers

from chuk_lazarus.data.tokenizers.custom_tokenizer import CustomTokenizer
from chuk_lazarus.data.tokenizers.tiktoken_wrapper import TiktokenWrapper, is_tiktoken_model
from chuk_lazarus.utils.huggingface import load_from_hub

# Set the logger
logger = logging.getLogger(__name__)


def load_tokenizer(tokenizer_name):
    """
    Load a tokenizer from a specified name.

    Supports:
    - Local tokenizer files (tokenizer.json)
    - HuggingFace models (e.g., "gpt2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    - OpenAI/tiktoken models (e.g., "gpt-4", "gpt-3.5-turbo", "o200k_base")

    For tiktoken support, install with: pip install 'chuk-lazarus[openai]'
    """
    tokenizer = None
    model_config_dir = f"core/models/architectures/{tokenizer_name}"
    vocab_file = os.path.join(model_config_dir, "tokenizer.json")

    # Check if this is a tiktoken/OpenAI model
    if is_tiktoken_model(tokenizer_name):
        try:
            logger.debug(f"Loading tiktoken tokenizer for: {tokenizer_name}")
            tokenizer = TiktokenWrapper.from_model(tokenizer_name)
            logger.debug(f"Tiktoken tokenizer loaded: vocab_size={tokenizer.vocab_size}")
            return tokenizer
        except ImportError as e:
            logger.error(f"tiktoken not installed: {e}")
            raise
        except ValueError as e:
            logger.error(f"Error loading tiktoken tokenizer: {e}")
            raise

    if os.path.exists(vocab_file):
        try:
            # Load the local tokenizer using the vocab file
            logger.debug(f"Loading tokenizer locally from '{vocab_file}'")
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

    # Restrict setting pad_token_id to eos_token_id to LLAMA architecture
    if "llama" in tokenizer_name.lower() or "llama" in tokenizer.__class__.__name__.lower():
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.sep_token_id = tokenizer.eos_token_id

    # Ensure pad_token_id is set for non-LLAMA architectures
    if tokenizer.pad_token_id is None:
        logger.debug("pad_token_id is None. Defaulting to 0.")
        tokenizer.pad_token_id = 0

    # success
    logger.debug("Tokenizer loaded successfully")
    return tokenizer
