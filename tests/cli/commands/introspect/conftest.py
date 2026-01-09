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

    # Set up SteeringService with async methods
    mock_steering_service = MagicMock()

    # Mock extract_direction async method
    mock_extract_result = MagicMock()
    mock_extract_result.layer = 6
    mock_extract_result.norm = 1.0
    mock_extract_result.cosine_similarity = 0.5
    mock_extract_result.separation = 1.0
    mock_extract_result.direction = MagicMock()
    mock_steering_service.extract_direction = AsyncMock(return_value=mock_extract_result)

    # Mock compare_coefficients async method
    mock_compare_result = MagicMock()
    mock_compare_result.results = {-1.0: "negative", 0.0: "neutral", 1.0: "positive"}
    mock_steering_service.compare_coefficients = AsyncMock(return_value=mock_compare_result)

    # Mock generate_with_steering async method
    mock_gen_result = MagicMock()
    mock_gen_result.prompt = "test"
    mock_gen_result.output = "generated"
    mock_gen_result.layer = 6
    mock_gen_result.coefficient = 1.0
    mock_steering_service.generate_with_steering = AsyncMock(return_value=[mock_gen_result])

    # Mock sync methods
    mock_steering_service.save_direction = MagicMock()
    mock_steering_service.load_direction = MagicMock(return_value=(MagicMock(), 6, {}))
    mock_steering_service.create_neuron_direction = MagicMock(return_value=MagicMock())

    mock_steering.SteeringService = mock_steering_service

    # Mock ActivationSteering
    mock_activation_steerer = MagicMock()
    mock_activation_steerer.num_layers = 12
    mock_activation_steerer.model.config.hidden_size = 768
    mock_steering.ActivationSteering = MagicMock()
    mock_steering.ActivationSteering.from_pretrained = MagicMock(
        return_value=mock_activation_steerer
    )

    # Mock neuron_service
    mock_neuron_service = MagicMock()

    # NeuronAnalysisService mock
    mock_neuron_analysis_service = MagicMock()
    mock_neuron_analysis_service.load_neurons_from_direction = MagicMock(
        return_value=(
            [100, 200],
            {100: 0.8, 200: -0.5},
            {"positive_label": "pos", "negative_label": "neg"},
        )
    )

    # Mock auto_discover_neurons async method
    mock_discovered_neuron = MagicMock()
    mock_discovered_neuron.idx = 100
    mock_discovered_neuron.separation = 1.5
    mock_discovered_neuron.best_pair = ("easy", "hard")
    mock_discovered_neuron.overall_std = 0.5
    mock_discovered_neuron.mean_range = 2.0
    mock_discovered_neuron.group_means = {"easy": 1.0, "hard": -1.0}
    mock_discovered_neuron.model_dump = MagicMock(
        return_value={
            "idx": 100,
            "separation": 1.5,
            "best_pair": ("easy", "hard"),
            "overall_std": 0.5,
            "mean_range": 2.0,
            "group_means": {"easy": 1.0, "hard": -1.0},
        }
    )

    mock_neuron_analysis_service.auto_discover_neurons = AsyncMock(
        return_value=[mock_discovered_neuron]
    )

    # Mock analyze_neurons async method
    mock_neuron_result = MagicMock()
    mock_neuron_result.neuron_idx = 100
    mock_neuron_result.min_val = -1.0
    mock_neuron_result.max_val = 1.0
    mock_neuron_result.mean_val = 0.5
    mock_neuron_result.std_val = 0.3
    mock_neuron_result.model_dump = MagicMock(
        return_value={
            "neuron_idx": 100,
            "min_val": -1.0,
            "max_val": 1.0,
            "mean_val": 0.5,
            "std_val": 0.3,
        }
    )

    mock_neuron_analysis_service.analyze_neurons = AsyncMock(
        return_value={12: [mock_neuron_result]}
    )

    mock_neuron_service.NeuronAnalysisService = mock_neuron_analysis_service
    mock_neuron_service.DiscoveredNeuron = MagicMock()
    mock_neuron_service.NeuronActivationResult = MagicMock()

    # Mock memory module
    mock_memory = MagicMock()

    # MemoryAnalysisService mock
    mock_memory_result = MagicMock()
    mock_memory_result.to_display = MagicMock(
        return_value=(
            "MEMORY STRUCTURE ANALYSIS\n"
            "Model: test-model\n"
            "Fact type: multiplication\n"
            "Layer: 6 (60%)"
        )
    )
    mock_memory_result.save = MagicMock()
    mock_memory_result.save_plot = MagicMock()

    mock_memory_service = MagicMock()
    mock_memory_service.analyze = AsyncMock(return_value=mock_memory_result)

    mock_memory.MemoryAnalysisService = mock_memory_service
    mock_memory.MemoryAnalysisConfig = MagicMock()
    mock_memory.MemoryAnalysisResult = MagicMock()

    # Mock clustering module
    mock_clustering = MagicMock()

    # ClusteringService mock
    mock_clustering_result = MagicMock()
    mock_clustering_result.to_display = MagicMock(
        return_value=(
            "ACTIVATION CLUSTERING\n"
            "Model: test-model\n"
            "Classes: easy, hard\n"
            "Legend: + = easy, o = hard"
        )
    )

    mock_clustering_service = MagicMock()
    mock_clustering_service.analyze = AsyncMock(return_value=mock_clustering_result)

    mock_clustering.ClusteringService = mock_clustering_service
    mock_clustering.ClusteringConfig = MagicMock()
    mock_clustering.ClusteringResult = MagicMock()

    # Mock generation module
    mock_generation = MagicMock()

    # GenerationService mock
    mock_generation_result = MagicMock()
    mock_generation_result.to_display = MagicMock(
        return_value=("GENERATION ANALYSIS\nModel: test-model\nPrompt: 2+2=\nGenerated: 4")
    )
    mock_generation_result.save = MagicMock()

    mock_generation_service = MagicMock()
    mock_generation_service.generate = AsyncMock(return_value=mock_generation_result)

    # LogitEvolutionService mock
    mock_evolution_result = MagicMock()
    mock_evolution_result.to_display = MagicMock(
        return_value=("LOGIT EVOLUTION\nModel: test-model\nTracked tokens: 4, 5")
    )

    mock_evolution_service = MagicMock()
    mock_evolution_service.analyze = AsyncMock(return_value=mock_evolution_result)

    mock_generation.GenerationService = mock_generation_service
    mock_generation.GenerationConfig = MagicMock()
    mock_generation.LogitEvolutionService = mock_evolution_service
    mock_generation.LogitEvolutionConfig = MagicMock()

    # Mock circuit module
    mock_circuit = MagicMock()

    # CircuitService.capture mock
    mock_capture_result = MagicMock()
    mock_capture_result.to_display = MagicMock(
        return_value=("CIRCUIT CAPTURE\nModel: test-model\nLayer: 6\nCaptured 3 prompts")
    )
    mock_capture_result.save = MagicMock()

    # CircuitService.invoke mock
    mock_invoke_result = MagicMock()
    mock_invoke_result.to_display = MagicMock(
        return_value=("CIRCUIT INVOCATION\nMethod: steer\nResults: [4, 6, 8]")
    )

    # CircuitService.decode mock
    mock_decode_result = MagicMock()
    mock_decode_result.to_display = MagicMock(
        return_value=("DECODE INJECTION\nPrompt: 2+2=\nOutput: 4")
    )

    # CircuitService.view mock
    mock_view_result = MagicMock()
    mock_view_result.to_display = MagicMock(return_value=("CIRCUIT VIEW\nEntries: 64\nLayer: 6"))

    # CircuitService.test mock
    mock_test_result = MagicMock()
    mock_test_result.to_display = MagicMock(
        return_value=("CIRCUIT TEST\nTesting circuit on prompts\nAccuracy: 0.95")
    )

    # CircuitService.compare mock
    mock_compare_result = MagicMock()
    mock_compare_result.to_display = MagicMock(
        return_value=("CIRCUIT COMPARE\nComparing circuits\nCosine similarity: 0.85")
    )

    mock_circuit_service = MagicMock()
    mock_circuit_service.capture = AsyncMock(return_value=mock_capture_result)
    mock_circuit_service.invoke = AsyncMock(return_value=mock_invoke_result)
    mock_circuit_service.decode = AsyncMock(return_value=mock_decode_result)
    mock_circuit_service.view = AsyncMock(return_value=mock_view_result)
    mock_circuit_service.test = AsyncMock(return_value=mock_test_result)
    mock_circuit_service.compare = AsyncMock(return_value=mock_compare_result)

    mock_circuit.CircuitService = mock_circuit_service
    mock_circuit.CircuitCaptureConfig = MagicMock()
    mock_circuit.CircuitInvokeConfig = MagicMock()
    mock_circuit.CircuitDecodeConfig = MagicMock()
    mock_circuit.CircuitViewConfig = MagicMock()
    mock_circuit.CircuitTestConfig = MagicMock()
    mock_circuit.CircuitCompareConfig = MagicMock()

    # Mock analyzer.service module
    mock_analyzer_service = MagicMock()

    # AnalyzerService.analyze mock
    mock_analyze_result = MagicMock()
    mock_analyze_result.to_display = MagicMock(
        return_value=(
            "LOGIT LENS ANALYSIS\nModel: test-model\nPrompt: 2+2=\nFinal prediction: 4 (0.95)"
        )
    )
    mock_analyze_result.save = MagicMock()

    # AnalyzerService.compare_models mock
    mock_compare_result = MagicMock()
    mock_compare_result.to_display = MagicMock(
        return_value=("MODEL COMPARISON\nModel 1: model-a\nModel 2: model-b\nPrediction diff: 0.05")
    )

    # AnalyzerService.demonstrate_hooks mock
    mock_hooks_result = MagicMock()
    mock_hooks_result.to_display = MagicMock(
        return_value=("HOOKS DEMONSTRATION\nModel: test-model\nCaptured States: 8 layers")
    )

    mock_analyzer = MagicMock()
    mock_analyzer.analyze = AsyncMock(return_value=mock_analyze_result)
    mock_analyzer.compare_models = AsyncMock(return_value=mock_compare_result)
    mock_analyzer.demonstrate_hooks = AsyncMock(return_value=mock_hooks_result)
    mock_analyzer.Config = MagicMock()

    mock_analyzer_service.AnalyzerService = mock_analyzer

    # Mock embedding module
    mock_embedding = MagicMock()

    # EmbeddingService mock
    mock_embedding_result = MagicMock()
    mock_embedding_result.to_display = MagicMock(
        return_value=("EMBEDDING ANALYSIS\nModel: test-model\nTask classification: 0.95")
    )

    mock_embedding_service = MagicMock()
    mock_embedding_service.analyze = AsyncMock(return_value=mock_embedding_result)

    mock_embedding.EmbeddingService = mock_embedding_service
    mock_embedding.EmbeddingConfig = MagicMock()

    # Mock early layers service
    mock_early_layers_result = MagicMock()
    mock_early_layers_result.to_display = MagicMock(
        return_value=("EARLY LAYERS ANALYSIS\nModel: test-model\nLayer 0: token=test")
    )

    mock_early_layers_service = MagicMock()
    mock_early_layers_service.analyze = AsyncMock(return_value=mock_early_layers_result)

    mock_embedding.EarlyLayersService = mock_early_layers_service

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

    # Mock probing module with services
    mock_probing = MagicMock()

    # MetacognitiveService mock
    mock_metacog_result = MagicMock()
    mock_metacog_result.to_display.return_value = (
        "METACOGNITIVE ANALYSIS\n"
        "Model: test-model\n"
        "Loading model: test-model\n"
        "Decision layer: 7 (70%)\n"
        "Strategy: DIRECT"
    )
    mock_metacog_service = MagicMock()
    mock_metacog_service.analyze = AsyncMock(return_value=mock_metacog_result)
    mock_probing.MetacognitiveService = mock_metacog_service
    mock_probing.MetacognitiveConfig = MagicMock()

    # UncertaintyService mock
    mock_uncertainty_result = MagicMock()
    mock_uncertainty_result.to_display.return_value = (
        "UNCERTAINTY DETECTION RESULTS\n"
        "Loading model: test-model\n"
        "Detection layer: 5\n"
        "Calibrating probes..."
    )
    mock_uncertainty_service = MagicMock()
    mock_uncertainty_service.analyze = AsyncMock(return_value=mock_uncertainty_result)
    mock_probing.UncertaintyService = mock_uncertainty_service
    mock_probing.UncertaintyConfig = MagicMock()

    # ProbeService mock
    mock_probe_result = MagicMock()
    mock_probe_result.to_display.return_value = "PROBE RESULTS\nAccuracy: 0.95"
    mock_probe_result.save = MagicMock()
    mock_probe_service = MagicMock()
    mock_probe_service.train_and_evaluate = AsyncMock(return_value=mock_probe_result)
    mock_probing.ProbeService = mock_probe_service
    mock_probing.ProbeConfig = MagicMock()

    # Pre-populate sys.modules so patch() calls can resolve the module path
    original_modules = {}
    modules_to_add = {
        "chuk_lazarus.introspection": mock_intro,
        "chuk_lazarus.introspection.ablation": mock_ablation,
        "chuk_lazarus.introspection.hooks": mock_hooks,
        "chuk_lazarus.introspection.external_memory": mock_external_memory,
        "chuk_lazarus.introspection.steering": mock_steering,
        "chuk_lazarus.introspection.steering.neuron_service": mock_neuron_service,
        "chuk_lazarus.introspection.enums": mock_enums,
        "chuk_lazarus.introspection.probing": mock_probing,
        "chuk_lazarus.introspection.memory": mock_memory,
        "chuk_lazarus.introspection.clustering": mock_clustering,
        "chuk_lazarus.introspection.generation": mock_generation,
        "chuk_lazarus.introspection.circuit": mock_circuit,
        "chuk_lazarus.introspection.analyzer.service": mock_analyzer_service,
        "chuk_lazarus.introspection.embedding": mock_embedding,
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
