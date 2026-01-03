"""Shared test fixtures for MoE tests."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chuk_lazarus.introspection.moe import (
    ExpertCategory,
    ExpertRole,
    MoEArchitecture,
)
from chuk_lazarus.introspection.moe.models import (
    CoactivationAnalysis,
    ExpertChatResult,
    ExpertIdentity,
    ExpertPair,
    GenerationStats,
    LayerRouterWeights,
    MoEModelInfo,
    RouterWeightCapture,
)


@pytest.fixture
def mock_moe_model_info() -> MoEModelInfo:
    """Standard MoE model info for testing."""
    return MoEModelInfo(
        moe_layers=(0, 1, 2, 3, 4, 5, 6, 7),
        num_experts=32,
        num_experts_per_tok=4,
        total_layers=8,
        architecture=MoEArchitecture.GPT_OSS,
        has_shared_expert=False,
    )


@pytest.fixture
def mock_generation_stats() -> GenerationStats:
    """Standard generation stats for testing."""
    return GenerationStats(
        expert_idx=6,
        tokens_generated=20,
        layers_modified=8,
        moe_type="gpt_oss_batched",
        prompt_tokens=10,
    )


@pytest.fixture
def mock_chat_result(mock_generation_stats) -> ExpertChatResult:
    """Standard chat result for testing."""
    return ExpertChatResult(
        prompt="127 * 89 = ",
        response="11303",
        expert_idx=6,
        stats=mock_generation_stats,
    )


@pytest.fixture
def mock_router_weights() -> list[LayerRouterWeights]:
    """Mock router weights for testing."""
    return [
        LayerRouterWeights(
            layer_idx=0,
            positions=(
                RouterWeightCapture(
                    layer_idx=0,
                    position_idx=0,
                    token="Hello",
                    expert_indices=(6, 7, 20, 1),
                    weights=(0.4, 0.3, 0.2, 0.1),
                ),
                RouterWeightCapture(
                    layer_idx=0,
                    position_idx=1,
                    token=" world",
                    expert_indices=(7, 6, 15, 3),
                    weights=(0.35, 0.3, 0.2, 0.15),
                ),
            ),
        ),
    ]


@pytest.fixture
def mock_coactivation() -> CoactivationAnalysis:
    """Mock co-activation analysis for testing."""
    return CoactivationAnalysis(
        layer_idx=0,
        total_activations=100,
        top_pairs=(
            ExpertPair(expert_a=6, expert_b=7, coactivation_count=25, coactivation_rate=0.25),
            ExpertPair(expert_a=6, expert_b=20, coactivation_count=15, coactivation_rate=0.15),
        ),
        specialist_pairs=(),
        generalist_experts=(6, 7),
    )


@pytest.fixture
def mock_expert_identity() -> ExpertIdentity:
    """Mock expert identity for testing."""
    return ExpertIdentity(
        expert_idx=6,
        layer_idx=0,
        primary_category=ExpertCategory.MATH,
        secondary_categories=(ExpertCategory.NUMBERS,),
        role=ExpertRole.SPECIALIST,
        confidence=0.85,
        activation_rate=0.12,
        top_tokens=("127", "89", "*", "="),
    )


@pytest.fixture
def mock_mlx_model():
    """Mock MLX model for testing."""
    mock = MagicMock()

    # Create mock layers with MoE structure
    layers = []
    for i in range(8):
        layer = MagicMock()
        layer.mlp = MagicMock()
        layer.mlp.router = MagicMock()
        layer.mlp.router.num_experts = 32
        layer.mlp.router.num_experts_per_tok = 4
        layer.mlp.experts = MagicMock()
        layer.mlp.experts.gate_up_proj = MagicMock()
        layer.mlp.experts.down_proj = MagicMock()
        layers.append(layer)

    mock.model.layers = layers
    return mock


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    mock = MagicMock()
    mock.encode.return_value = [1, 2, 3, 4, 5]
    mock.decode.return_value = "decoded text"
    mock.vocab_size = 32000
    mock.eos_token_id = 2
    mock.chat_template = None
    return mock


@pytest.fixture
def mock_expert_router(
    mock_moe_model_info, mock_chat_result, mock_router_weights, mock_coactivation
):
    """Mock ExpertRouter for CLI testing."""
    with patch("chuk_lazarus.introspection.moe.ExpertRouter") as mock_cls:
        mock_router = AsyncMock()
        mock_router.info = mock_moe_model_info
        mock_router._moe_type = "gpt_oss_batched"
        mock_router.tokenizer = MagicMock()

        mock_router.chat_with_expert = AsyncMock(return_value=mock_chat_result)
        mock_router.compare_experts = AsyncMock()
        mock_router.generate_with_ablation = AsyncMock(
            return_value=("ablated output", mock_chat_result.stats)
        )
        mock_router.generate_with_topk = AsyncMock()
        mock_router.capture_router_weights = AsyncMock(return_value=mock_router_weights)
        mock_router.analyze_coactivation = AsyncMock(return_value=mock_coactivation)
        mock_router._generate_normal_sync = MagicMock(return_value="normal output")

        mock_router.__aenter__ = AsyncMock(return_value=mock_router)
        mock_router.__aexit__ = AsyncMock(return_value=None)

        mock_cls.from_pretrained = AsyncMock(return_value=mock_router)
        yield mock_cls
