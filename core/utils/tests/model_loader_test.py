import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from core.models.model_config import ModelConfig
from core.utils.huggingface_utils import load_from_hub
from core.utils.model_loader import load_model
from huggingface_hub import snapshot_download

# Constants
FAKE_LOCAL_PATH = 'core/models/architecture/lazyfox/lazyfox_model.py'
FAKE_CONFIG_PATH = 'core/models/architecture/lazyfox/config.json'
LOCAL_MODEL_NAME = 'lazyfox'


@pytest.fixture
def mock_model_config():
    mock = MagicMock(spec=ModelConfig)
    mock.some_attribute = "some_value"  # Set up any attributes you expect
    return mock

# Test loading model locally
@patch('importlib.import_module')
@patch('os.path.join', return_value=FAKE_CONFIG_PATH)
@patch('core.models.model_config.ModelConfig.load')
def test_load_local_model(mock_load, mock_join, mock_import_module, mock_model_config):
    # Mock the module to have a __file__ attribute
    mock_import_module.return_value = MagicMock(__file__=FAKE_LOCAL_PATH, CustomModel=MagicMock())
    mock_load.return_value = mock_model_config

    model = load_model(LOCAL_MODEL_NAME)

    mock_import_module.assert_called_once_with(f'core.models.architectures.{LOCAL_MODEL_NAME}.{LOCAL_MODEL_NAME}_model')
    mock_load.assert_called_once_with(Path(FAKE_CONFIG_PATH))
    assert model == mock_model_config