import importlib
import logging
import os
from pathlib import Path
from core.models.model_config import ModelConfig
from utils.huggingface_utils import load_from_hub

logger = logging.getLogger(__name__)

def load_model(model_name):
    """
    Load a model from a specified name. If the model is in the local 'models' folder, load that.
    Otherwise, load from the Hugging Face model repository.
    """
    model = None
    
    try:
        # Attempt to import the model module from the 'models' directory
        model_module = importlib.import_module(f"models.architectures.{model_name}.{model_name}_model")
        
        # If the module exists and has a class named CustomModel, create an instance
        if hasattr(model_module, 'CustomModel'):
            # load the model config
            config_path = Path(os.path.join(os.path.dirname(model_module.__file__), f"config.json"))
            model = ModelConfig.load(config_path)
        else:
            logger.info(f"No CustomModel class found in the module 'models.architectures.{model_name}.{model_name}_model'.")
    except ImportError as e:
        logger.info(f"Local model module not found: {e}")

    if model is None:
        try:
            # Load the model from the hub if local loading fails
            logger.info(f"Attempting to load model from Hugging Face hub: {model_name}")
            resolved_path = load_from_hub(model_name)

            # load the file
            config_path = Path(os.path.join(resolved_path, f"config.json"))
            model = ModelConfig.load(config_path)
        except Exception as e:
            logger.error(f"Error loading model from Hugging Face hub: {e}")
            raise

    logger.info("Model loaded successfully")
    return model
