import os
import pytest
import tempfile
import numpy as np
from batch_generation.llama_finetune_batch import LLaMAFineTuneBatch
from unittest.mock import MagicMock

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.bos_token_id = 1  # Assuming <s> is token ID 1
    
    def mock_encode(text, **kwargs):
        token_map = {
            "<s>": 1,
            "</s>": 2,
            "[INST]": 91,
            "[/INST]": 93,
            "Hi": 105,
            "Bye.": 120,
        }
        tokens = []
        
        # Replace known tokens with spaces to ensure correct tokenization
        for token, token_id in token_map.items():
            text = text.replace(token, f" {token} ")
        
        # Process the text by splitting and mapping to tokens
        for word in text.split():
            if word in token_map:
                tokens.append(token_map[word])
            else:
                tokens.extend([ord(char) for char in word])
        
        # Ensure that the </s> token is only added if it's not already the last token
        if tokens and tokens[-1] == token_map["</s>"]:
            return tokens
        else:
            tokens.append(token_map["</s>"])
            return tokens
    
    tokenizer.encode = mock_encode
    return tokenizer
