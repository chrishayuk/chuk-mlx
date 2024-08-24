import pytest
from core.batch.text_utils import get_line_text

def test_get_line_text_json():
    line = '{"text": "hello world"}\n'
    assert get_line_text(line) == "hello world"

def test_get_line_text_json_no_text_key():
    line = '{"content": "hello world"}\n'
    assert get_line_text(line) == "hello world"

def test_get_line_text_plain():
    line = "hello world\n"
    assert get_line_text(line) == "hello world"

def test_get_line_text_invalid_json():
    line = '{"wrong": "format"}\n'
    with pytest.raises(ValueError):
        get_line_text(line)

