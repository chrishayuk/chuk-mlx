import pytest
from core.batch.llama_finetune_batch import LLaMAFineTuneBatch
from unittest.mock import MagicMock

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.bos_token_id = 1  # Assuming <s> is token ID 1

    def mock_encode(text, add_special_tokens=False):
        token_map = {
            "<s>": 1,
            "</s>": 2,
            "[INST]": 91,
            "[/INST]": 93,
            "What": 121,
            "classification": 122,
            "comes": 123,
            "with": 124,
            "-30°": 125,
            "Calvin?": 126,
            "You've": 127,
            "taken": 128,
            "that": 129,
            "too": 130,
            "far": 131,
            "The": 132,
            "temperature": 133,
            "is": 134,
            "-30°C": 135,
            "Stay": 136,
            "warm!": 137,
            "Instruction": 138,
            "here": 139,
            "Extra": 140,
            "content": 141,
            "Another": 142,
            "More": 143,
        }
        tokens = []
        
        for token, token_id in token_map.items():
            text = text.replace(token, f" {token} ")
        
        for word in text.split():
            if word in token_map:
                tokens.append(token_map[word])
            else:
                tokens.extend([ord(char) for char in word])
        
        if add_special_tokens:
            if tokens and tokens[0] != tokenizer.bos_token_id:
                tokens.insert(0, tokenizer.bos_token_id)
            if tokens and tokens[-1] != tokenizer.eos_token_id:
                tokens.append(tokenizer.eos_token_id)
        
        return tokens

    tokenizer.encode = mock_encode
    return tokenizer

@pytest.fixture
def llama_batch(mock_tokenizer):
    return LLaMAFineTuneBatch(mock_tokenizer, '/tmp', 'llama', 128, 32, True)

def test_tokenization_with_special_characters(llama_batch):
    line = '{"text": "<s>[INST] What classification comes with -30° Calvin? [/INST] You\'ve taken that too far </s>"}'
    input_tokens, target_tokens, attention_mask = llama_batch.tokenize_line(line)
    assert input_tokens is not None
    assert target_tokens is not None
    assert attention_mask is not None
    assert len(input_tokens) > 0
    assert len(target_tokens) > 0
    assert input_tokens[0] == llama_batch.tokenizer.bos_token_id
    assert target_tokens[-1] == llama_batch.tokenizer.eos_token_id
    assert len(attention_mask) == len(input_tokens)  # Ensure attention mask matches input length

def test_missing_instruction_tags(llama_batch):
    line = '{"text": "What classification comes with -30° Calvin?"}'
    input_tokens, target_tokens, attention_mask = llama_batch.tokenize_line(line)
    assert input_tokens is not None
    assert target_tokens is not None
    assert attention_mask is not None
    assert len(input_tokens) > 0
    assert target_tokens == [llama_batch.tokenizer.eos_token_id]
    assert len(attention_mask) == len(input_tokens)

def test_edge_case_special_characters(llama_batch):
    line = '{"text": "<s>[INST] The temperature is -30°C [/INST] Stay warm! </s>"}'
    input_tokens, target_tokens, attention_mask = llama_batch.tokenize_line(line)
    assert input_tokens is not None
    assert target_tokens is not None
    assert attention_mask is not None
    assert len(input_tokens) > 0
    assert len(target_tokens) > 0
    assert input_tokens[0] == llama_batch.tokenizer.bos_token_id
    assert target_tokens[-1] == llama_batch.tokenizer.eos_token_id
    assert len(attention_mask) == len(input_tokens)

def test_no_special_tokens(llama_batch):
    line = '{"text": "Hello world"}'
    input_tokens, target_tokens, attention_mask = llama_batch.tokenize_line(line)
    assert input_tokens is not None
    assert target_tokens is not None
    assert attention_mask is not None
    assert len(input_tokens) > 0
    assert len(target_tokens) == 1  # Only EOS token
    assert len(attention_mask) == len(input_tokens)

def test_extra_tags(llama_batch):
    line = '{"text": "<s>[INST] Instruction here [/INST] Extra content [INST] Another instruction [/INST] More content </s>"}'
    input_tokens, target_tokens, attention_mask = llama_batch.tokenize_line(line)
    assert input_tokens is not None
    assert target_tokens is not None
    assert attention_mask is not None
    assert len(input_tokens) > 0
    assert len(target_tokens) > 0
    assert input_tokens[0] == llama_batch.tokenizer.bos_token_id
    assert target_tokens[-1] == llama_batch.tokenizer.eos_token_id
    assert len(attention_mask) == len(input_tokens)

def test_edge_case_empty_text(llama_batch):
    line = '{"text": ""}'
    input_tokens, target_tokens, attention_mask = llama_batch.tokenize_line(line)
    assert input_tokens is None
    assert target_tokens is None
    assert attention_mask is None



