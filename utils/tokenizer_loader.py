import importlib
import transformers
from utils.huggingface_utils import load_from_hub

def load_tokenizer(tokenizer_name):
    """
    Load a tokenizer from a specified name. If the tokenizer is in the local 'chuk_tokenizers' folder,
    load that. Otherwise, load from the Hugging Face model repository.
    """
    try:
        # Attempt to import the tokenizer module from the 'chuk_tokenizers' directory
        print(f"chuk_tokenizers.{tokenizer_name}")
        tokenizer_module = importlib.import_module(f"chuk_tokenizers.{tokenizer_name}")

        # If the module exists and has a class named CustomTokenizer, create an instance
        if hasattr(tokenizer_module, 'CustomTokenizer'):
            return tokenizer_module.CustomTokenizer()
        else:
            print(f"No CustomTokenizer class found in the module {tokenizer_name}. Falling back to Hugging Face hub.")
    except ImportError as e:
        print(f"Local tokenizer {tokenizer_name} not found, loading from Hugging Face hub. Error: {e}")

    # Load the model from the hub if local loading fails
    resolved_path = load_from_hub(tokenizer_name)
    return transformers.AutoTokenizer.from_pretrained(resolved_path)

