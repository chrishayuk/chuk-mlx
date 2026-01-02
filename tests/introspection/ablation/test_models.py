"""Tests for ablation models module."""

from chuk_lazarus.introspection.ablation.models import (
    AblationResult,
    LayerSweepResult,
)


class TestAblationResult:
    """Tests for AblationResult dataclass."""

    def test_creation(self):
        """Test creating an ablation result."""
        result = AblationResult(
            layer=5,
            component="mlp",
            original_output="The capital is Paris",
            ablated_output="The capital is ???",
            original_criterion=True,
            ablated_criterion=False,
            criterion_changed=True,
        )
        assert result.layer == 5
        assert result.component == "mlp"
        assert result.original_output == "The capital is Paris"
        assert result.ablated_output == "The capital is ???"
        assert result.original_criterion is True
        assert result.ablated_criterion is False
        assert result.criterion_changed is True

    def test_default_values(self):
        """Test default values."""
        result = AblationResult(
            layer=0,
            component="attention",
            original_output="output1",
            ablated_output="output2",
            original_criterion=True,
            ablated_criterion=True,
            criterion_changed=False,
        )
        assert result.output_coherent is True
        assert result.metadata == {}

    def test_with_metadata(self):
        """Test with custom metadata."""
        result = AblationResult(
            layer=0,
            component="mlp",
            original_output="out1",
            ablated_output="out2",
            original_criterion=True,
            ablated_criterion=True,
            criterion_changed=False,
            metadata={"extra": "info"},
        )
        assert result.metadata == {"extra": "info"}


class TestLayerSweepResult:
    """Tests for LayerSweepResult dataclass."""

    def test_creation(self):
        """Test creating a layer sweep result."""
        results = [
            AblationResult(
                layer=0,
                component="mlp",
                original_output="a",
                ablated_output="a",
                original_criterion=True,
                ablated_criterion=True,
                criterion_changed=False,
            ),
            AblationResult(
                layer=1,
                component="mlp",
                original_output="a",
                ablated_output="b",
                original_criterion=True,
                ablated_criterion=False,
                criterion_changed=True,
            ),
            AblationResult(
                layer=2,
                component="mlp",
                original_output="a",
                ablated_output="a",
                original_criterion=True,
                ablated_criterion=True,
                criterion_changed=False,
            ),
        ]
        sweep = LayerSweepResult(
            task_name="test_task",
            criterion_name="has_answer",
            results=results,
        )
        assert sweep.task_name == "test_task"
        assert sweep.criterion_name == "has_answer"
        assert len(sweep.results) == 3

    def test_causal_layers_computed(self):
        """Test that causal_layers is computed from results."""
        results = [
            AblationResult(
                layer=0,
                component="mlp",
                original_output="a",
                ablated_output="a",
                original_criterion=True,
                ablated_criterion=True,
                criterion_changed=False,
            ),
            AblationResult(
                layer=1,
                component="mlp",
                original_output="a",
                ablated_output="b",
                original_criterion=True,
                ablated_criterion=False,
                criterion_changed=True,
            ),
            AblationResult(
                layer=2,
                component="mlp",
                original_output="a",
                ablated_output="c",
                original_criterion=True,
                ablated_criterion=False,
                criterion_changed=True,
            ),
        ]
        sweep = LayerSweepResult(
            task_name="test",
            criterion_name="criterion",
            results=results,
        )
        assert sweep.causal_layers == [1, 2]

    def test_no_causal_layers(self):
        """Test when no layers are causal."""
        results = [
            AblationResult(
                layer=0,
                component="mlp",
                original_output="a",
                ablated_output="a",
                original_criterion=True,
                ablated_criterion=True,
                criterion_changed=False,
            ),
        ]
        sweep = LayerSweepResult(
            task_name="test",
            criterion_name="criterion",
            results=results,
        )
        assert sweep.causal_layers == []

    def test_empty_results(self):
        """Test with empty results."""
        sweep = LayerSweepResult(
            task_name="test",
            criterion_name="criterion",
            results=[],
        )
        assert sweep.causal_layers == []
