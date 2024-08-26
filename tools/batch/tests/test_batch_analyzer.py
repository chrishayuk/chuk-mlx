import pytest
import numpy as np
import os
from unittest.mock import patch, MagicMock

from tools.batch.batch_analyzer import analyze_batch_file

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0  # Assuming pad_token_id is 0
    tokenizer.encode = MagicMock(return_value=[1, 2, 3])
    tokenizer.decode = MagicMock(return_value="decoded string")
    yield tokenizer

@pytest.fixture
def mock_generate_summary():
    with patch('tools.batch.batch_analyzer.generate_batch_analysis_summary_table') as mock_summary:
        mock_summary.return_value = "Mocked Summary Table"
        yield mock_summary

def test_analyze_batch_file_with_input_tensor(mock_tokenizer, mock_generate_summary, tmp_path):
    batch_file = tmp_path / "batch_data.npz"
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]])
    np.savez(batch_file, input_tensor=input_tensor)

    with patch('tools.batch.batch_analyzer.load_tokenizer', return_value=mock_tokenizer):
        analyze_batch_file(str(batch_file), "dummy_tokenizer", tensor_type="input")

    # Use np.testing.assert_array_equal for comparing arrays
    called_args = mock_generate_summary.call_args[0]
    np.testing.assert_array_equal(called_args[0], input_tensor)
    assert called_args[1] == str(batch_file)
    assert called_args[2] == 0

def test_analyze_batch_file_with_target_tensor(mock_tokenizer, mock_generate_summary, tmp_path):
    batch_file = tmp_path / "batch_data.npz"
    target_tensor = np.array([[7, 8, 9], [10, 11, 12]])
    np.savez(batch_file, target_tensor=target_tensor)

    with patch('tools.batch.batch_analyzer.load_tokenizer', return_value=mock_tokenizer):
        analyze_batch_file(str(batch_file), "dummy_tokenizer", tensor_type="target")

    # Use np.testing.assert_array_equal for comparing arrays
    called_args = mock_generate_summary.call_args[0]
    np.testing.assert_array_equal(called_args[0], target_tensor)
    assert called_args[1] == str(batch_file)
    assert called_args[2] == 0

def test_analyze_batch_file_with_both_tensors(mock_tokenizer, mock_generate_summary, tmp_path):
    batch_file = tmp_path / "batch_data.npz"
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]])
    target_tensor = np.array([[7, 8, 9], [10, 11, 12]])
    np.savez(batch_file, input_tensor=input_tensor, target_tensor=target_tensor)

    with patch('tools.batch.batch_analyzer.load_tokenizer', return_value=mock_tokenizer):
        analyze_batch_file(str(batch_file), "dummy_tokenizer")

    # Check that both input and target tensors were analyzed
    assert mock_generate_summary.call_count == 2
    called_args_input = mock_generate_summary.call_args_list[0][0]
    called_args_target = mock_generate_summary.call_args_list[1][0]

    np.testing.assert_array_equal(called_args_input[0], input_tensor)
    assert called_args_input[1] == str(batch_file)
    assert called_args_input[2] == 0

    np.testing.assert_array_equal(called_args_target[0], target_tensor)
    assert called_args_target[1] == str(batch_file)
    assert called_args_target[2] == 0

def test_analyze_batch_file_with_attention_mask(mock_tokenizer, mock_generate_summary, tmp_path):
    batch_file = tmp_path / "batch_data.npz"
    attention_mask_tensor = np.array([[1, 1, 1], [1, 0, 0]])
    np.savez(batch_file, attention_mask_tensor=attention_mask_tensor)

    with patch('tools.batch.batch_analyzer.load_tokenizer', return_value=mock_tokenizer):
        analyze_batch_file(str(batch_file), "dummy_tokenizer", tensor_type="attention_mask")

    called_args = mock_generate_summary.call_args[0]
    np.testing.assert_array_equal(called_args[0], attention_mask_tensor)
    assert called_args[1] == str(batch_file)
    assert called_args[2] == 0

def test_analyze_batch_file_with_no_tensors(mock_tokenizer, mock_generate_summary, tmp_path, capsys):
    batch_file = tmp_path / "batch_data.npz"
    np.savez(batch_file)

    with patch('tools.batch.batch_analyzer.load_tokenizer', return_value=mock_tokenizer):
        analyze_batch_file(str(batch_file), "dummy_tokenizer", tensor_type="input")

    # Check that the summary function was not called
    mock_generate_summary.assert_not_called()

    # Capture and check output
    captured = capsys.readouterr()
    assert "No input tensor found in the .npz file" in captured.out

def test_analyze_batch_file_with_incorrect_tensor_type(mock_tokenizer, mock_generate_summary, tmp_path, capsys):
    batch_file = tmp_path / "batch_data.npz"
    np.savez(batch_file, input_tensor=np.array([[1, 2, 3], [4, 5, 6]]))

    with patch('tools.batch.batch_analyzer.load_tokenizer', return_value=mock_tokenizer):
        analyze_batch_file(str(batch_file), "dummy_tokenizer", tensor_type="unknown")

    # Check that the summary function was not called
    mock_generate_summary.assert_not_called()

    # Capture and check output
    captured = capsys.readouterr()
    assert "No unknown tensor found in the .npz file" in captured.out

