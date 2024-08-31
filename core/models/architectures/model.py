import mlx.core as mx
import mlx.nn as nn
from core.models.model_config import ModelConfig
from core.utils.memory_utils import log_memory_usage

import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ModelMode(Enum):
    TRAIN = "train"
    INFERENCE = "inference"

class Model(nn.Module):
        
    def __init__(self, args: ModelConfig):
        # call the base
        super().__init__()

        # set the config
        self.config = args

        # Default cache usage to False for training
        self.use_cache = False 

        # Initialize the language model head
        if not self.config.tie_word_embeddings:
            # tied 
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        else:
            # no head
            self.lm_head = None

    def __call__(self, inputs: mx.array, cache=None):
        # log memory
        log_memory_usage("model: pre forward pass")
        
        # Use cache only if required
        if self.use_cache:
            # forward pass with cache
            out, cache = self.model(inputs, cache)
        else:
            # forward pass, skipping cache generation
            out, _ = self.model(inputs, None)
        
        # log memory
        log_memory_usage("model: post forward pass")
        
        # Apply the language model head
        if self.lm_head is not None:
            out = self.lm_head(out)
        else:
            out = self.model.embed_tokens.as_linear(out)
        
        # return
        return out, cache if self.use_cache else None

    def sanitize(self, weights):
        """
        Sanitize the model weights if needed.
        Currently, this method just returns the weights as-is.
        """
        return {k: v for k, v in weights.items()}

    def set_mode(self, mode: ModelMode):
        """
        Set the mode of the model: ModelMode.TRAIN or ModelMode.INFERENCE.
        This controls whether the cache is used.
        """
        if mode == ModelMode.TRAIN:
            # cache disabled
            self.use_cache = False

            # log output
            logger.info("Model set to training mode: cache disabled.")
        elif mode == ModelMode.INFERENCE:
            # cache enabled
            self.use_cache = True

            # log output
            logger.info("Model set to inference mode: cache enabled.")
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def reset_cache(self):
        """
        Reset or clear the cache if necessary. This can be used between sequences
        during inference to manage memory.
        """
        # only reset if using cache
        if self.use_cache:
            # check for cache
            if hasattr(self, 'cache') and self.cache is not None:
                logger.info("Resetting the model cache.")

                # loop throughn the cache keys
                for key in self.cache.keys():
                    # Clear individual entries
                    self.cache[key] = None  

                # completely remove the cache object
                self.cache = None  
            else:
                logger.info("No cache to reset.")

