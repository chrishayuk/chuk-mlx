"""Tests for ExpertRouter async router manipulation."""

from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from chuk_lazarus.introspection.moe.enums import MoEArchitecture
from chuk_lazarus.introspection.moe.expert_router import ExpertRouter
from chuk_lazarus.introspection.moe.models import (
    CoactivationAnalysis,
    ExpertChatResult,
    ExpertComparisonResult,
    GenerationStats,
    LayerRouterWeights,
    MoEModelInfo,
    TopKVariationResult,
)


class TestExpertRouterInit:
    """Tests for ExpertRouter initialization."""

    def test_init_with_valid_moe_model(self, mock_mlx_model, mock_tokenizer, mock_moe_model_info):
        """Test initialization with valid MoE model."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)
        assert router.info == mock_moe_model_info
        assert router.tokenizer == mock_tokenizer

    def test_init_raises_for_non_moe_model(self, mock_mlx_model, mock_tokenizer):
        """Test initialization raises for non-MoE model."""
        non_moe_info = MoEModelInfo(
            moe_layers=(),  # Empty - no MoE layers
            num_experts=0,
            num_experts_per_tok=0,
            total_layers=8,
        )
        with pytest.raises(ValueError, match="no MoE layers"):
            ExpertRouter(mock_mlx_model, mock_tokenizer, non_moe_info)

    def test_info_property(self, mock_mlx_model, mock_tokenizer, mock_moe_model_info):
        """Test info property returns model info."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)
        assert router.info.num_experts == 32
        assert router.info.num_experts_per_tok == 4

    def test_tokenizer_property(self, mock_mlx_model, mock_tokenizer, mock_moe_model_info):
        """Test tokenizer property."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)
        assert router.tokenizer.vocab_size == 32000


class TestExpertRouterContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_mlx_model, mock_tokenizer, mock_moe_model_info):
        """Test async context manager enter and exit."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)
        async with router as r:
            assert r is router
        # Should not raise on exit


class TestExpertRouterMoETypeDetection:
    """Tests for MoE type detection."""

    def test_detect_gpt_oss_batched(self, mock_tokenizer, mock_moe_model_info):
        """Test detection of GPT-OSS batched style."""
        # Create a model with GPT-OSS batched structure
        mock_model = MagicMock()
        layers = []
        for _ in range(8):
            layer = MagicMock()
            layer.mlp = MagicMock()
            layer.mlp.router = MagicMock()
            # GPT-OSS batched has experts.gate_up_proj
            layer.mlp.experts = MagicMock()
            layer.mlp.experts.gate_up_proj = MagicMock()
            layers.append(layer)
        mock_model.model.layers = layers

        router = ExpertRouter(mock_model, mock_tokenizer, mock_moe_model_info)
        assert router._moe_type == "gpt_oss_batched"

    def test_detect_standard(self, mock_tokenizer, mock_moe_model_info):
        """Test detection of standard MoE style."""
        mock_model = MagicMock()
        layers = []
        for _ in range(8):
            layer = MagicMock()
            layer.mlp = MagicMock()
            layer.mlp.router = MagicMock()
            # Standard MoE has experts list but no gate_up_proj
            layer.mlp.experts = [MagicMock() for _ in range(8)]
            # Remove the attribute to make hasattr return False
            del layer.mlp.experts
            layer.mlp.experts = MagicMock(spec=[])  # Empty spec means no gate_up_proj
            layers.append(layer)
        mock_model.model.layers = layers

        router = ExpertRouter(mock_model, mock_tokenizer, mock_moe_model_info)
        assert router._moe_type == "standard"


class TestExpertRouterGeneration:
    """Tests for generation methods."""

    @pytest.mark.asyncio
    async def test_chat_with_expert_returns_result(
        self, mock_mlx_model, mock_tokenizer, mock_moe_model_info
    ):
        """Test chat_with_expert returns ExpertChatResult."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)

        # Mock the sync method
        mock_stats = GenerationStats(
            expert_idx=6,
            tokens_generated=10,
            layers_modified=8,
            moe_type="gpt_oss_batched",
        )
        router._generate_with_forced_expert_sync = MagicMock(return_value=("11303", mock_stats))

        result = await router.chat_with_expert("127 * 89 = ", expert_idx=6)

        assert isinstance(result, ExpertChatResult)
        assert result.prompt == "127 * 89 = "
        assert result.response == "11303"
        assert result.expert_idx == 6

    @pytest.mark.asyncio
    async def test_compare_experts_returns_comparison(
        self, mock_mlx_model, mock_tokenizer, mock_moe_model_info
    ):
        """Test compare_experts returns ExpertComparisonResult."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)

        # Mock the sync method
        mock_stats = GenerationStats(
            expert_idx=6,
            tokens_generated=10,
            layers_modified=8,
            moe_type="gpt_oss_batched",
        )
        router._generate_with_forced_expert_sync = MagicMock(return_value=("response", mock_stats))

        result = await router.compare_experts("Test", expert_indices=[6, 7, 20])

        assert isinstance(result, ExpertComparisonResult)
        assert result.prompt == "Test"
        assert len(result.expert_results) == 3

    @pytest.mark.asyncio
    async def test_generate_with_ablation_returns_tuple(
        self, mock_mlx_model, mock_tokenizer, mock_moe_model_info
    ):
        """Test generate_with_ablation returns response and stats."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)

        mock_stats = GenerationStats(
            expert_idx=-1,
            tokens_generated=15,
            layers_modified=8,
            moe_type="gpt_oss_batched",
        )
        router._generate_with_ablation_sync = MagicMock(return_value=("ablated output", mock_stats))

        text, stats = await router.generate_with_ablation("Test prompt", expert_indices=[6, 7])

        assert text == "ablated output"
        assert isinstance(stats, GenerationStats)
        assert stats.expert_idx == -1

    @pytest.mark.asyncio
    async def test_generate_with_topk_returns_result(
        self, mock_mlx_model, mock_tokenizer, mock_moe_model_info
    ):
        """Test generate_with_topk returns TopKVariationResult."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)

        router._generate_normal_sync = MagicMock(return_value="normal response")
        router._generate_with_topk_sync = MagicMock(return_value="topk response")

        result = await router.generate_with_topk("Test prompt", k=2)

        assert isinstance(result, TopKVariationResult)
        assert result.k_value == 2
        assert result.default_k == 4
        assert result.normal_response == "normal response"
        assert result.response == "topk response"


class TestExpertRouterAnalysis:
    """Tests for analysis methods."""

    @pytest.mark.asyncio
    async def test_capture_router_weights_returns_list(
        self, mock_mlx_model, mock_tokenizer, mock_moe_model_info
    ):
        """Test capture_router_weights returns list of LayerRouterWeights."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)

        # Mock the sync method
        from chuk_lazarus.introspection.moe.models import RouterWeightCapture

        mock_weights = [
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
                ),
            )
        ]
        router._capture_router_weights_sync = MagicMock(return_value=mock_weights)

        result = await router.capture_router_weights("Hello world")

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], LayerRouterWeights)

    @pytest.mark.asyncio
    async def test_analyze_coactivation_returns_analysis(
        self, mock_mlx_model, mock_tokenizer, mock_moe_model_info
    ):
        """Test analyze_coactivation returns CoactivationAnalysis."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)

        mock_analysis = CoactivationAnalysis(
            layer_idx=0,
            total_activations=100,
            generalist_experts=(6, 7),
        )
        router._analyze_coactivation_sync = MagicMock(return_value=mock_analysis)

        result = await router.analyze_coactivation(["Test 1", "Test 2"])

        assert isinstance(result, CoactivationAnalysis)
        assert result.layer_idx == 0
        assert result.total_activations == 100


class TestExpertRouterSampling:
    """Tests for token sampling."""

    def test_sample_token_greedy(self, mock_mlx_model, mock_tokenizer, mock_moe_model_info):
        """Test greedy sampling (temperature=0)."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)

        # Create logits with clear maximum
        logits = mx.array([[[0.1, 0.2, 0.9, 0.3]]])  # Token 2 has highest logit

        token = router._sample_token(logits, temperature=0.0)
        assert token == 2

    def test_sample_token_with_temperature(
        self, mock_mlx_model, mock_tokenizer, mock_moe_model_info
    ):
        """Test sampling with temperature > 0."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)

        logits = mx.array([[[0.1, 0.2, 0.9, 0.3]]])

        # With temperature, should still return a valid token index
        token = router._sample_token(logits, temperature=1.0)
        assert 0 <= token < 4


class TestExpertRouterForwardPatching:
    """Tests for forward function patching."""

    def test_make_forced_expert_forward_gpt_oss(
        self, mock_mlx_model, mock_tokenizer, mock_moe_model_info
    ):
        """Test GPT-OSS style forced expert forward creation."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)

        # Create a mock MLP with GPT-OSS structure
        mock_mlp = MagicMock()
        mock_mlp.experts = MagicMock()

        # Create properly shaped arrays
        # gate_up_proj: [num_experts, hidden, 2*intermediate]
        # down_proj: [num_experts, hidden, intermediate] (transposed in the code)
        hidden_size = 64
        intermediate_size = 128
        mock_mlp.experts.gate_up_proj = mx.zeros((32, hidden_size, 2 * intermediate_size))
        mock_mlp.experts.down_proj = mx.zeros((32, hidden_size, intermediate_size))

        forward_fn = router._make_forced_expert_forward_gpt_oss(mock_mlp, expert_idx=6)

        # Should be callable
        assert callable(forward_fn)

        # Test with input
        x = mx.zeros((1, 10, hidden_size))
        output = forward_fn(x)
        assert output.shape == x.shape

    def test_make_forced_expert_forward_standard(
        self, mock_mlx_model, mock_tokenizer, mock_moe_model_info
    ):
        """Test standard MoE style forced expert forward creation."""
        router = ExpertRouter(mock_mlx_model, mock_tokenizer, mock_moe_model_info)

        # Create a mock MLP with standard structure
        mock_expert = MagicMock()
        mock_expert.return_value = mx.zeros((1, 10, 64))

        mock_mlp = MagicMock()
        mock_mlp.experts = [mock_expert for _ in range(8)]

        forward_fn = router._make_forced_expert_forward_standard(mock_mlp, expert_idx=0)

        assert callable(forward_fn)


class TestExpertRouterExtractInfo:
    """Tests for MoE info extraction."""

    def test_extract_moe_info_gpt_oss(self):
        """Test extracting info from GPT-OSS style model."""
        mock_model = MagicMock()
        layers = []
        for _ in range(8):
            layer = MagicMock()
            layer.mlp = MagicMock()
            layer.mlp.router = MagicMock()
            layer.mlp.router.num_experts = 32
            layer.mlp.router.num_experts_per_tok = 4
            # No shared_expert
            layer.mlp.shared_expert = None
            del layer.mlp.shared_expert
            layers.append(layer)
        mock_model.model.layers = layers

        info = ExpertRouter._extract_moe_info(mock_model)

        assert info.num_experts == 32
        assert info.num_experts_per_tok == 4
        assert info.total_layers == 8
        assert len(info.moe_layers) == 8
        assert info.architecture == MoEArchitecture.GPT_OSS

    def test_extract_moe_info_mixtral(self):
        """Test extracting info from Mixtral style model."""
        mock_model = MagicMock()
        layers = []
        for _ in range(8):
            layer = MagicMock()
            layer.mlp = MagicMock()
            layer.mlp.router = MagicMock()
            layer.mlp.router.num_experts = 8
            layer.mlp.router.num_experts_per_tok = 2
            del layer.mlp.shared_expert
            layers.append(layer)
        mock_model.model.layers = layers

        info = ExpertRouter._extract_moe_info(mock_model)

        assert info.num_experts == 8
        assert info.num_experts_per_tok == 2
        assert info.architecture == MoEArchitecture.MIXTRAL

    def test_extract_moe_info_with_shared_expert(self):
        """Test extracting info from model with shared expert."""
        mock_model = MagicMock()
        layers = []
        for _ in range(8):
            layer = MagicMock()
            layer.mlp = MagicMock()
            layer.mlp.router = MagicMock()
            layer.mlp.router.num_experts = 16
            layer.mlp.router.num_experts_per_tok = 2
            layer.mlp.shared_expert = MagicMock()  # Has shared expert
            layers.append(layer)
        mock_model.model.layers = layers

        info = ExpertRouter._extract_moe_info(mock_model)

        assert info.has_shared_expert is True
        assert info.architecture == MoEArchitecture.LLAMA4

    def test_extract_moe_info_no_moe_layers(self):
        """Test extracting info from model without MoE."""
        mock_model = MagicMock()
        layers = []
        for _ in range(8):
            layer = MagicMock()
            layer.mlp = MagicMock()
            # No router attribute
            del layer.mlp.router
            layer.mlp.router = None
            layers.append(layer)
        mock_model.model.layers = layers

        # Need to patch hasattr to return False for router
        original_hasattr = hasattr

        def custom_hasattr(obj, name):
            if name == "router" and hasattr(obj, "__class__") and "MagicMock" in str(type(obj)):
                return False
            return original_hasattr(obj, name)

        with patch("builtins.hasattr", custom_hasattr):
            info = ExpertRouter._extract_moe_info(mock_model)

        assert info.moe_layers == ()
        assert info.is_moe is False


class TestExpertRouterFromPretrained:
    """Tests for from_pretrained class method."""

    @pytest.mark.asyncio
    async def test_from_pretrained_calls_load_sync(self):
        """Test that from_pretrained calls _load_model_sync."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_info = MoEModelInfo(
            moe_layers=(0, 1, 2, 3, 4, 5, 6, 7),
            num_experts=32,
            num_experts_per_tok=4,
            total_layers=8,
            architecture=MoEArchitecture.GPT_OSS,
        )

        # Create proper mock model structure
        layers = []
        for _ in range(8):
            layer = MagicMock()
            layer.mlp = MagicMock()
            layer.mlp.router = MagicMock()
            layer.mlp.experts = MagicMock()
            layer.mlp.experts.gate_up_proj = MagicMock()
            layers.append(layer)
        mock_model.model.layers = layers

        with patch.object(
            ExpertRouter,
            "_load_model_sync",
            return_value=(mock_model, mock_tokenizer, mock_info),
        ):
            router = await ExpertRouter.from_pretrained("test/model")

            assert isinstance(router, ExpertRouter)
            assert router.info.num_experts == 32
