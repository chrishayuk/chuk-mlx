"""Tests for overlay_inference.py to improve coverage."""

from unittest.mock import MagicMock, patch

import mlx.core as mx

from chuk_lazarus.introspection.moe.overlay_inference import OverlayMoEModel


class TestOverlayMoEModelInit:
    """Tests for OverlayMoEModel initialization."""

    def test_init(self):
        """Test OverlayMoEModel initialization (lines 37-46)."""
        mock_model = MagicMock()
        mock_overlay = MagicMock()
        mock_tokenizer = MagicMock()

        wrapper = OverlayMoEModel(mock_model, mock_overlay, mock_tokenizer)

        assert wrapper._model is mock_model
        assert wrapper.overlay is mock_overlay
        assert wrapper.tokenizer is mock_tokenizer
        assert wrapper._patched is False

    def test_model_property(self):
        """Test model property returns underlying model (lines 48-51)."""
        mock_model = MagicMock()
        wrapper = OverlayMoEModel(mock_model, MagicMock(), MagicMock())

        assert wrapper.model is mock_model


class TestGetLayer:
    """Tests for _get_layer method."""

    def test_get_layer_model_model_layers(self):
        """Test _get_layer with model.model.layers structure (lines 115-116)."""
        # Create mock model with model.model.layers structure
        mock_layer = MagicMock()
        mock_model = MagicMock()
        mock_model.model.layers = [mock_layer]

        wrapper = OverlayMoEModel(mock_model, MagicMock(), MagicMock())
        result = wrapper._get_layer(0)

        assert result is mock_layer

    def test_get_layer_transformer_h(self):
        """Test _get_layer with transformer.h structure (lines 117-118)."""
        # Create mock model with transformer.h structure
        mock_layer = MagicMock()
        mock_model = MagicMock(spec=[])  # No model attribute
        mock_model.transformer = MagicMock()
        mock_model.transformer.h = [mock_layer]

        wrapper = OverlayMoEModel(mock_model, MagicMock(), MagicMock())
        result = wrapper._get_layer(0)

        assert result is mock_layer

    def test_get_layer_unknown_structure(self):
        """Test _get_layer with unknown model structure (line 119)."""
        # Create mock model with no recognized structure
        mock_model = MagicMock(spec=[])  # No model or transformer attribute

        wrapper = OverlayMoEModel(mock_model, MagicMock(), MagicMock())
        result = wrapper._get_layer(0)

        assert result is None


class TestGetMoEBlock:
    """Tests for _get_moe_block method."""

    def test_get_moe_block_block_sparse_moe(self):
        """Test _get_moe_block with block_sparse_moe attribute (lines 124-128)."""
        mock_moe = MagicMock()
        mock_moe.experts = [MagicMock()]

        mock_layer = MagicMock()
        mock_layer.block_sparse_moe = mock_moe

        wrapper = OverlayMoEModel(MagicMock(), MagicMock(), MagicMock())
        result = wrapper._get_moe_block(mock_layer)

        assert result is mock_moe

    def test_get_moe_block_moe(self):
        """Test _get_moe_block with moe attribute."""
        mock_moe = MagicMock()
        mock_moe.experts = [MagicMock()]

        mock_layer = MagicMock(spec=["moe"])
        mock_layer.moe = mock_moe

        wrapper = OverlayMoEModel(MagicMock(), MagicMock(), MagicMock())
        result = wrapper._get_moe_block(mock_layer)

        assert result is mock_moe

    def test_get_moe_block_mlp(self):
        """Test _get_moe_block with mlp attribute."""
        mock_moe = MagicMock()
        mock_moe.experts = [MagicMock()]

        mock_layer = MagicMock(spec=["mlp"])
        mock_layer.mlp = mock_moe

        wrapper = OverlayMoEModel(MagicMock(), MagicMock(), MagicMock())
        result = wrapper._get_moe_block(mock_layer)

        assert result is mock_moe

    def test_get_moe_block_feed_forward(self):
        """Test _get_moe_block with feed_forward attribute."""
        mock_moe = MagicMock()
        mock_moe.experts = [MagicMock()]

        mock_layer = MagicMock(spec=["feed_forward"])
        mock_layer.feed_forward = mock_moe

        wrapper = OverlayMoEModel(MagicMock(), MagicMock(), MagicMock())
        result = wrapper._get_moe_block(mock_layer)

        assert result is mock_moe

    def test_get_moe_block_no_experts(self):
        """Test _get_moe_block when block has no experts attr (line 127-128)."""
        mock_mlp = MagicMock(spec=[])  # No experts attribute

        mock_layer = MagicMock(spec=["mlp"])
        mock_layer.mlp = mock_mlp

        wrapper = OverlayMoEModel(MagicMock(), MagicMock(), MagicMock())
        result = wrapper._get_moe_block(mock_layer)

        assert result is None

    def test_get_moe_block_none(self):
        """Test _get_moe_block with no matching attribute (line 129)."""
        mock_layer = MagicMock(spec=[])  # No matching attributes

        wrapper = OverlayMoEModel(MagicMock(), MagicMock(), MagicMock())
        result = wrapper._get_moe_block(mock_layer)

        assert result is None


class TestPatchExperts:
    """Tests for _patch_experts method."""

    def test_patch_experts_already_patched(self):
        """Test _patch_experts skips if already patched (lines 93-94)."""
        mock_overlay = MagicMock()
        mock_overlay.moe_layer_indices = [0, 1]

        wrapper = OverlayMoEModel(MagicMock(), mock_overlay, MagicMock())
        wrapper._patched = True

        # Should return early without doing anything
        wrapper._patch_experts()

        # overlay.moe_layer_indices should not be iterated
        assert wrapper._patched is True

    def test_patch_experts_layer_none(self):
        """Test _patch_experts handles None layer (lines 99-100)."""
        mock_model = MagicMock(spec=[])  # No model.model.layers
        mock_overlay = MagicMock()
        mock_overlay.moe_layer_indices = [0]

        wrapper = OverlayMoEModel(mock_model, mock_overlay, MagicMock())
        wrapper._patch_experts()

        assert wrapper._patched is True

    def test_patch_experts_moe_block_none(self):
        """Test _patch_experts handles None moe_block (lines 104-105)."""
        mock_layer = MagicMock(spec=[])  # No moe block attributes
        mock_model = MagicMock()
        mock_model.model.layers = [mock_layer]

        mock_overlay = MagicMock()
        mock_overlay.moe_layer_indices = [0]

        wrapper = OverlayMoEModel(mock_model, mock_overlay, MagicMock())
        wrapper._patch_experts()

        assert wrapper._patched is True

    def test_patch_experts_successful(self):
        """Test _patch_experts patches MoE blocks (lines 107-111)."""
        # Create mock MoE block with experts
        mock_expert = MagicMock()
        mock_moe = MagicMock()
        mock_moe.experts = [mock_expert, mock_expert]  # 2 experts

        mock_layer = MagicMock()
        mock_layer.block_sparse_moe = mock_moe

        mock_model = MagicMock()
        mock_model.model.layers = [mock_layer]

        mock_overlay = MagicMock()
        mock_overlay.moe_layer_indices = [0]

        wrapper = OverlayMoEModel(mock_model, mock_overlay, MagicMock())
        wrapper._patch_experts()

        assert wrapper._patched is True
        # Experts should be replaced
        assert len(mock_moe.experts) == 2


class TestPatchMoEBlock:
    """Tests for _patch_moe_block method."""

    def test_patch_moe_block_creates_overlay_experts(self):
        """Test _patch_moe_block creates OverlayExpert instances (lines 131-161)."""
        mock_expert = MagicMock()
        mock_moe = MagicMock()
        mock_moe.experts = [mock_expert, mock_expert, mock_expert]  # 3 experts

        mock_overlay = MagicMock()
        mock_overlay.apply_expert = MagicMock(return_value=mx.zeros((1, 10)))

        wrapper = OverlayMoEModel(MagicMock(), mock_overlay, MagicMock())
        wrapper._patch_moe_block(mock_moe, layer_idx=5)

        # Check experts were replaced
        assert len(mock_moe.experts) == 3
        # Each expert should be an OverlayExpert (has expert_idx attribute)
        for i, expert in enumerate(mock_moe.experts):
            assert hasattr(expert, "expert_idx")
            assert expert.expert_idx == i
            assert expert.layer_idx == 5


class TestOverlayExpertCall:
    """Tests for the inner OverlayExpert.__call__ method."""

    def test_overlay_expert_call(self):
        """Test OverlayExpert forward pass (lines 147-157)."""
        # Create mock overlay
        mock_overlay = MagicMock()

        # Setup return values for apply_expert
        gate_out = mx.array([[1.0, 2.0, 3.0]])
        up_out = mx.array([[0.5, 0.5, 0.5]])
        down_out = mx.array([[0.1, 0.2]])

        def mock_apply_expert(layer_idx, proj_type, expert_idx, x):
            if proj_type == "gate":
                return gate_out
            elif proj_type == "up":
                return up_out
            elif proj_type == "down":
                return down_out

        mock_overlay.apply_expert = mock_apply_expert

        # Create mock MoE block
        mock_moe = MagicMock()
        mock_moe.experts = [MagicMock()]

        wrapper = OverlayMoEModel(MagicMock(), mock_overlay, MagicMock())
        wrapper._patch_moe_block(mock_moe, layer_idx=0)

        # Call the patched expert
        x = mx.array([[1.0, 1.0]])
        result = mock_moe.experts[0](x)

        # Result should be from the down projection
        assert result.shape == down_out.shape


class TestMemoryUsage:
    """Tests for memory_usage method."""

    def test_memory_usage(self):
        """Test memory_usage returns correct statistics (lines 190-200)."""
        mock_overlay = MagicMock()
        mock_overlay.memory_usage_mb.return_value = 100.0
        mock_overlay.config.original_bytes = 500 * 1024 * 1024  # 500 MB
        mock_overlay.config.compression_ratio = 5.0

        wrapper = OverlayMoEModel(MagicMock(), mock_overlay, MagicMock())
        result = wrapper.memory_usage()

        assert result["overlay_mb"] == 100.0
        assert result["original_expert_mb"] == 500.0
        assert result["savings_mb"] == 400.0
        assert result["compression_ratio"] == 5.0


class TestGenerate:
    """Tests for generate method."""

    @patch("mlx_lm.generate")
    def test_generate_calls_mlx_generate(self, mock_generate):
        """Test generate calls mlx_lm.generate (lines 163-188)."""
        mock_generate.return_value = "Generated text"

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        wrapper = OverlayMoEModel(mock_model, MagicMock(), mock_tokenizer)

        result = wrapper.generate("Test prompt", max_tokens=50)

        mock_generate.assert_called_once_with(
            mock_model,
            mock_tokenizer,
            prompt="Test prompt",
            max_tokens=50,
        )
        assert result == "Generated text"


class TestLoad:
    """Tests for load class method."""

    @patch("mlx_lm.utils.load_tokenizer")
    @patch("chuk_lazarus.introspection.moe.moe_type.MoETypeService._load_model")
    @patch("chuk_lazarus.introspection.moe.moe_compression.OverlayExperts.load")
    def test_load_creates_wrapper(self, mock_load_overlay, mock_load_model, mock_load_tokenizer):
        """Test load creates and patches OverlayMoEModel (lines 54-89)."""
        # Setup mocks
        mock_overlay = MagicMock()
        mock_overlay.moe_layer_indices = []
        mock_overlay.num_layers = 24
        mock_overlay.num_experts = 8
        mock_overlay.config.compression_ratio = 5.0
        mock_load_overlay.return_value = mock_overlay

        mock_model = MagicMock(spec=[])  # No model.model.layers
        mock_load_model.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer

        result = OverlayMoEModel.load(
            compressed_path="/path/to/compressed",
            original_model_id="test/model",
        )

        mock_load_overlay.assert_called_once_with("/path/to/compressed")
        mock_load_model.assert_called_once_with("test/model")
        mock_load_tokenizer.assert_called_once_with("test/model")

        assert isinstance(result, OverlayMoEModel)
        assert result.overlay is mock_overlay
        assert result._patched is True


class TestGetLayerEdgeCases:
    """Edge case tests for _get_layer."""

    def test_get_layer_with_index(self):
        """Test _get_layer accesses correct index."""
        mock_layers = [MagicMock(name=f"layer_{i}") for i in range(3)]
        mock_model = MagicMock()
        mock_model.model.layers = mock_layers

        wrapper = OverlayMoEModel(mock_model, MagicMock(), MagicMock())

        assert wrapper._get_layer(0) is mock_layers[0]
        assert wrapper._get_layer(1) is mock_layers[1]
        assert wrapper._get_layer(2) is mock_layers[2]


class TestGetMoEBlockPriority:
    """Tests for _get_moe_block attribute priority."""

    def test_block_sparse_moe_has_priority(self):
        """Test block_sparse_moe is checked first."""
        mock_moe1 = MagicMock()
        mock_moe1.experts = [MagicMock()]
        mock_moe2 = MagicMock()
        mock_moe2.experts = [MagicMock()]

        mock_layer = MagicMock()
        mock_layer.block_sparse_moe = mock_moe1
        mock_layer.moe = mock_moe2

        wrapper = OverlayMoEModel(MagicMock(), MagicMock(), MagicMock())
        result = wrapper._get_moe_block(mock_layer)

        assert result is mock_moe1


class TestPatchExpertsMultipleLayers:
    """Tests for patching multiple layers."""

    def test_patch_multiple_layers(self):
        """Test _patch_experts handles multiple layers."""
        mock_moe1 = MagicMock()
        mock_moe1.experts = [MagicMock()]
        mock_moe2 = MagicMock()
        mock_moe2.experts = [MagicMock(), MagicMock()]

        mock_layer1 = MagicMock()
        mock_layer1.block_sparse_moe = mock_moe1
        mock_layer2 = MagicMock()
        mock_layer2.block_sparse_moe = mock_moe2

        mock_model = MagicMock()
        mock_model.model.layers = [mock_layer1, mock_layer2]

        mock_overlay = MagicMock()
        mock_overlay.moe_layer_indices = [0, 1]

        wrapper = OverlayMoEModel(mock_model, mock_overlay, MagicMock())
        wrapper._patch_experts()

        assert wrapper._patched is True
        assert len(mock_moe1.experts) == 1
        assert len(mock_moe2.experts) == 2
