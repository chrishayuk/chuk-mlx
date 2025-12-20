import mlx.core as mx
import mlx.nn as nn
from chuk_lazarus.models.model_config import ModelConfig
from chuk_lazarus.utils.memory import log_memory_usage
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ModelMode(Enum):
    TRAIN = "train"
    INFERENCE = "inference"

class Model(nn.Module):
    def __init__(self, args: ModelConfig):
        # call the parent constructor
        super().__init__()

        # set the config
        self.config = args

        # Default cache usage to False for training
        self.use_cache = False  

        # The model attribute will be set by subclasses
        self.model = None

        # Initialize the language model head
        if not self.config.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        else:
            self.lm_head = None

    def __call__(self, inputs: mx.array, cache=None):
        # check we have a model set
        if self.model is None:
            raise ValueError("The model has not been set. Ensure that a subclass sets the model.")

        # Use cache only if required
        out, cache = self.model(inputs, cache=cache if self.use_cache else None)

        # Apply the language model head
        if self.lm_head is not None:
            out = self.lm_head(out)
        else:
            out = self.model.embed_tokens.as_linear(out)
        
        return out, cache if self.use_cache else None

    def sanitize(self, weights):
        return {k: v for k, v in weights.items()}

    def set_mode(self, mode: ModelMode):
        if mode == ModelMode.TRAIN:
            self.use_cache = False
            logger.info("Model set to training mode: cache disabled.")
        elif mode == ModelMode.INFERENCE:
            self.use_cache = True
            logger.info("Model set to inference mode: cache enabled.")
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def reset_cache(self):
        """
        Reset or clear the cache if necessary. This can be used between sequences
        during inference to manage memory.
        """
        if self.use_cache:
            if self.model is not None:
                self.model.reset_cache()
                logger.info("Model cache has been reset.")
            else:
                logger.info("No model set; cache reset skipped.")
