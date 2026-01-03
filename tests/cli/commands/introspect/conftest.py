"""Shared fixtures for introspect CLI tests."""

import sys
from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Check if sklearn is available and working
SKLEARN_AVAILABLE = False
try:
    SKLEARN_AVAILABLE = True
except Exception:
    # Catch any exception - numpy incompatibility raises various errors
    pass

# Marker for tests that require sklearn
requires_sklearn = pytest.mark.skipif(
    not SKLEARN_AVAILABLE,
    reason="sklearn not available or incompatible with numpy version",
)


def _create_introspection_mock():
    """Create a comprehensive mock for the introspection module."""
    mock = MagicMock()

    # Mock all the commonly used classes
    mock.AblationStudy = MagicMock()
    mock.ModelAnalyzer = MagicMock()
    mock.ModelAccessor = MagicMock()
    mock.ModelHooks = MagicMock()
    mock.ActivationSteering = MagicMock()
    mock.ActivationPatcher = MagicMock()
    mock.CommutativityAnalyzer = MagicMock()
    mock.AnalysisConfig = MagicMock()
    mock.LayerStrategy = MagicMock()
    mock.LayerStrategy.SPECIFIC = "specific"
    mock.LayerStrategy.EVENLY_SPACED = "evenly_spaced"
    mock.LayerStrategy.ALL = "all"

    # Mock helper functions
    mock.apply_chat_template = MagicMock(side_effect=lambda t, p: p)
    mock.extract_expected_answer = MagicMock(return_value=None)

    # Mock parse_prompts_from_arg to return proper parsed prompts
    def mock_parse_prompts(arg):
        if arg is None:
            return []
        if arg.startswith("@"):
            # File format - return empty for now
            return []
        return [p.strip() for p in arg.split("|") if p.strip()]

    mock.parse_prompts_from_arg = MagicMock(side_effect=mock_parse_prompts)

    # Mock ParsedArithmeticPrompt
    mock_parsed = MagicMock()
    mock_parsed.prompt = "7*4="
    mock_parsed.operand_a = 7
    mock_parsed.operand_b = 4
    mock_parsed.result = 28

    mock.ParsedArithmeticPrompt = MagicMock()
    mock.ParsedArithmeticPrompt.parse = MagicMock(
        side_effect=lambda p, r: MagicMock(
            prompt=p, operand_a=7, operand_b=4, result=r if r else 28
        )
    )

    # Mock CaptureConfig and PositionSelection
    mock.CaptureConfig = MagicMock()
    mock.PositionSelection = MagicMock()
    mock.PositionSelection.LAST = "last"

    return mock


@pytest.fixture(autouse=True)
def setup_introspection_module():
    """Set up mock introspection module in sys.modules before patches run."""
    mock_intro = _create_introspection_mock()

    # Create submodule mocks
    mock_ablation = MagicMock()
    mock_ablation.AblationStudy = mock_intro.AblationStudy

    mock_hooks = MagicMock()
    mock_hooks.ModelHooks = mock_intro.ModelHooks
    mock_hooks.CaptureConfig = MagicMock()
    mock_hooks.PositionSelection = MagicMock()
    mock_hooks.PositionSelection.LAST = "last"

    mock_external_memory = MagicMock()
    mock_external_memory.ExternalMemoryStore = MagicMock()
    mock_external_memory.ExternalMemory = MagicMock()

    mock_steering = MagicMock()
    mock_steering.SteeringHook = MagicMock()

    # Mock moe submodules to allow moe_expert imports to succeed
    mock_moe = MagicMock()
    mock_moe.ExpertRouter = MagicMock()
    mock_moe.MoEModelInfo = MagicMock()
    mock_moe.MoEArchitecture = MagicMock()

    mock_moe_enums = MagicMock()
    mock_moe_enums.MoEArchitecture = MagicMock()
    mock_moe_enums.MoEArchitecture.GPT_OSS = "gpt_oss"

    mock_moe_models = MagicMock()
    mock_moe_models.MoEModelInfo = MagicMock()
    mock_moe_models.LayerRouterWeights = MagicMock()
    mock_moe_models.RouterWeightCapture = MagicMock()

    mock_moe_router = MagicMock()
    mock_moe_router.ExpertRouter = MagicMock()

    # Mock introspection.enums
    mock_enums = MagicMock()
    mock_enums.OverrideMode = MagicMock()
    mock_enums.OverrideMode.REPLACE = "replace"
    mock_enums.OverrideMode.ADD = "add"

    # Pre-populate sys.modules so patch() calls can resolve the module path
    original_modules = {}
    modules_to_add = {
        "chuk_lazarus.introspection": mock_intro,
        "chuk_lazarus.introspection.ablation": mock_ablation,
        "chuk_lazarus.introspection.hooks": mock_hooks,
        "chuk_lazarus.introspection.external_memory": mock_external_memory,
        "chuk_lazarus.introspection.steering": mock_steering,
        "chuk_lazarus.introspection.enums": mock_enums,
        "chuk_lazarus.introspection.moe": mock_moe,
        "chuk_lazarus.introspection.moe.enums": mock_moe_enums,
        "chuk_lazarus.introspection.moe.models": mock_moe_models,
        "chuk_lazarus.introspection.moe.router": mock_moe_router,
    }

    for mod_name in modules_to_add:
        if mod_name in sys.modules:
            original_modules[mod_name] = sys.modules[mod_name]
        sys.modules[mod_name] = modules_to_add[mod_name]

    yield mock_intro

    # Restore original state
    for mod_name in modules_to_add:
        if mod_name in original_modules:
            sys.modules[mod_name] = original_modules[mod_name]
        elif mod_name in sys.modules:
            del sys.modules[mod_name]


@pytest.fixture
def mock_model():
    """Create a mock model with typical structure."""
    model = MagicMock()

    # Nested structure (like Llama/Gemma)
    model.model.layers = [MagicMock() for _ in range(12)]
    model.model.embed_tokens = MagicMock()
    model.model.norm = MagicMock()
    model.lm_head = MagicMock()

    # Config
    model.config = MagicMock()
    model.config.hidden_size = 768
    model.config.num_hidden_layers = 12
    model.config.vocab_size = 32000

    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "test output"
    tokenizer.chat_template = None
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    return tokenizer


@pytest.fixture
def mock_mlx_lm_load(mock_model, mock_tokenizer):
    """Patch mlx_lm.load to return mock model and tokenizer.

    We patch the entire mlx_lm module to avoid importing it, since
    importing mlx_lm triggers transformers -> sklearn imports which
    fail with numpy 2.x incompatibility.
    """
    # Create a mock mlx_lm module with load function
    mock_mlx_lm = MagicMock()
    mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
    mock_mlx_lm.generate.return_value = "generated text"

    with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
        yield mock_mlx_lm.load


@pytest.fixture
def mock_mlx_lm_generate():
    """Patch mlx_lm.generate."""
    with patch("mlx_lm.generate") as mock_gen:
        mock_gen.return_value = "generated text"
        yield mock_gen


@pytest.fixture
def basic_args():
    """Create basic CLI arguments."""
    return Namespace(
        model="test-model",
        prompt="test prompt",
        prompts="test prompt",
        output=None,
        layer=None,
        layers=None,
        top_k=5,
        temperature=0.0,
        max_tokens=10,
        raw=False,
    )


@pytest.fixture
def mock_hf_loader():
    """Mock HFLoader for model loading."""
    with patch("chuk_lazarus.inference.loader.HFLoader") as mock_loader:
        mock_result = MagicMock()
        mock_result.model_path = MagicMock()
        mock_result.model_path.__truediv__ = lambda self, x: MagicMock()
        mock_loader.download.return_value = mock_result
        mock_loader.load_tokenizer.return_value = MagicMock()
        yield mock_loader


@pytest.fixture
def mock_model_analyzer():
    """Mock ModelAnalyzer for analysis commands."""
    with patch("chuk_lazarus.introspection.ModelAnalyzer") as mock_cls:
        mock_analyzer = MagicMock()
        # Use AsyncMock for async context manager methods
        mock_analyzer.__aenter__ = AsyncMock(return_value=mock_analyzer)
        mock_analyzer.__aexit__ = AsyncMock(return_value=None)

        # Model info - use real values for attributes used in f-strings/formatting
        mock_analyzer.model_info = MagicMock()
        mock_analyzer.model_info.model_id = "test-model"
        mock_analyzer.model_info.model_type = "llama"
        mock_analyzer.model_info.num_layers = 12
        mock_analyzer.model_info.hidden_size = 768
        mock_analyzer.model_info.vocab_size = 32000
        mock_analyzer.model_info.has_tied_embeddings = False

        # Model config - used for printing info
        mock_analyzer.config = MagicMock()
        mock_analyzer.config.model_type = "llama"
        mock_analyzer.config.embedding_scale = None  # Avoid format string issues

        # Analysis result - provide real values for attributes used in JSON serialization
        mock_result = MagicMock()
        mock_result.prompt = "test prompt"
        mock_result.tokens = ["test", "prompt"]
        mock_result.num_layers = 12
        mock_result.captured_layers = [0, 4, 8, 11]
        mock_result.final_prediction = []
        mock_result.layer_predictions = []
        mock_result.token_evolutions = []
        # Provide proper to_dict for JSON serialization
        mock_result.to_dict.return_value = {
            "prompt": "test prompt",
            "tokens": ["test", "prompt"],
            "num_layers": 12,
            "captured_layers": [0, 4, 8, 11],
        }

        # analyze is an async method - use AsyncMock
        mock_analyzer.analyze = AsyncMock(return_value=mock_result)
        mock_cls.from_pretrained.return_value = mock_analyzer

        yield mock_cls


@pytest.fixture
def mock_ablation_study():
    """Mock AblationStudy for ablation commands."""
    # Patch at both locations - where it's defined and where it's imported
    with (
        patch("chuk_lazarus.introspection.AblationStudy") as mock_cls1,
        patch("chuk_lazarus.introspection.ablation.AblationStudy") as mock_cls2,
    ):
        mock_study = MagicMock()
        mock_study.adapter.model = MagicMock()
        mock_study.adapter.tokenizer = MagicMock()
        mock_study.adapter.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_study.adapter.tokenizer.decode.return_value = "output"
        mock_study.adapter.config = MagicMock()
        mock_study.adapter.num_layers = 12

        mock_cls1.from_pretrained.return_value = mock_study
        mock_cls2.from_pretrained.return_value = mock_study

        yield mock_cls1


@pytest.fixture
def mock_activation_steering():
    """Mock ActivationSteering for steering commands."""
    with patch("chuk_lazarus.introspection.ActivationSteering") as mock_cls:
        mock_steerer = MagicMock()
        mock_steerer.num_layers = 12
        mock_steerer.model = MagicMock()
        mock_steerer.model.config.hidden_size = 768
        # Provide proper tokenizer with encode returning a list
        mock_steerer.tokenizer = MagicMock()
        mock_steerer.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_steerer.tokenizer.decode.return_value = "decoded text"
        mock_steerer.generate.return_value = "generated output"

        mock_cls.from_pretrained.return_value = mock_steerer

        yield mock_cls


@pytest.fixture
def mock_numpy():
    """Mock numpy for tests that need it."""
    with patch.dict("sys.modules", {"numpy": MagicMock()}):
        yield


@pytest.fixture
def mock_sklearn():
    """Mock sklearn for tests that need it."""
    mock_pca = MagicMock()
    mock_pca.fit_transform.return_value = MagicMock()
    mock_pca.explained_variance_ratio_ = [0.5, 0.3]

    mock_logreg = MagicMock()
    mock_logreg.fit.return_value = mock_logreg
    mock_logreg.score.return_value = 0.95

    with patch.dict(
        "sys.modules",
        {
            "sklearn": MagicMock(),
            "sklearn.decomposition": MagicMock(PCA=MagicMock(return_value=mock_pca)),
            "sklearn.linear_model": MagicMock(
                LogisticRegression=MagicMock(return_value=mock_logreg),
                LinearRegression=MagicMock(),
                Ridge=MagicMock(),
            ),
            "sklearn.model_selection": MagicMock(),
        },
    ):
        yield
