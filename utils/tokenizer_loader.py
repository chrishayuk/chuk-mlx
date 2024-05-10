import importlib
import transformers
from utils.huggingface_utils import load_from_hub

def load_tokenizer(tokenizer_name):
    """
    Load a tokenizer from a specified name. If the tokenizer is in the local 'chuk_tokenizers' folder, load that.
    Otherwise, load from the Hugging Face model repository.
    """
    try:
        # Attempt to import the tokenizer module from the 'chuk_tokenizers' directory
        print(f"chuk_tokenizers.{tokenizer_name}")
        tokenizer_module = importlib.import_module(f"chuk_tokenizers.{tokenizer_name}")
        # If the module exists and has a class named CustomTokenizer, create an instance
        if hasattr(tokenizer_module, 'CustomTokenizer'):
            tokenizer = tokenizer_module.CustomTokenizer()
        else:
            print(f"No CustomTokenizer class found in the module {tokenizer_name}. Falling back to Hugging Face hub.")
            tokenizer = None
    except ImportError as e:
        print(f"Local tokenizer {tokenizer_name} not found, loading from Hugging Face hub. Error: {e}")
        tokenizer = None
    
    if tokenizer is None:
        # Load the model from the hub if local loading fails
        resolved_path = load_from_hub(tokenizer_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(resolved_path)
    
    # Check if the tokenizer is LLAMA-based and set the pad_token_id accordingly
    if hasattr(tokenizer, 'eos_token_id'):
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # check pad_token_id is none
    if tokenizer.pad_token_id is None:
        # default to zero
        tokenizer.pad_token_id = 0

    # return the tokenizer
    return tokenizer