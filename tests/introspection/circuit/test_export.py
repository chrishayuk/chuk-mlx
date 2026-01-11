"""Tests for circuit graph export utilities."""

import json
import tempfile
from pathlib import Path

import pytest

from chuk_lazarus.introspection.circuit.export import (
    CircuitEdge,
    CircuitGraph,
    CircuitNode,
    EdgeType,
    NodeType,
    create_circuit_from_ablation,
    create_circuit_from_directions,
    export_circuit_to_dot,
    export_circuit_to_html,
    export_circuit_to_json,
    export_circuit_to_mermaid,
    load_circuit,
    load_circuit_from_json,
    save_circuit,
)

# =============================================================================
# Tests for Models
# =============================================================================


class TestCircuitNode:
    """Tests for CircuitNode model."""

    def test_creation(self):
        """Test node creation."""
        node = CircuitNode(
            id="test_node",
            label="Test Node",
            node_type=NodeType.MLP,
            layer=5,
            importance=0.8,
        )
        assert node.id == "test_node"
        assert node.label == "Test Node"
        assert node.node_type == NodeType.MLP
        assert node.layer == 5

    def test_default_values(self):
        """Test default values."""
        node = CircuitNode(
            id="node",
            label="Label",
            node_type=NodeType.LAYER,
            layer=0,
        )
        assert node.importance == 1.0
        assert node.metadata == {}


class TestCircuitEdge:
    """Tests for CircuitEdge model."""

    def test_creation(self):
        """Test edge creation."""
        edge = CircuitEdge(
            source="node_a",
            target="node_b",
            edge_type=EdgeType.CAUSAL,
            weight=0.9,
            label="strong",
        )
        assert edge.source == "node_a"
        assert edge.target == "node_b"
        assert edge.weight == 0.9

    def test_default_values(self):
        """Test default values."""
        edge = CircuitEdge(
            source="a",
            target="b",
            edge_type=EdgeType.RESIDUAL,
        )
        assert edge.weight == 1.0
        assert edge.label == ""


class TestCircuitGraph:
    """Tests for CircuitGraph model."""

    def test_creation(self):
        """Test graph creation."""
        nodes = (
            CircuitNode(id="input", label="Input", node_type=NodeType.INPUT, layer=-1),
            CircuitNode(id="output", label="Output", node_type=NodeType.OUTPUT, layer=-1),
        )
        edges = (CircuitEdge(source="input", target="output", edge_type=EdgeType.RESIDUAL),)

        graph = CircuitGraph(
            name="Test Circuit",
            description="A test circuit",
            nodes=nodes,
            edges=edges,
        )

        assert graph.name == "Test Circuit"
        assert graph.num_nodes == 2
        assert graph.num_edges == 1

    def test_get_node(self):
        """Test get_node method."""
        nodes = (
            CircuitNode(id="node1", label="Node 1", node_type=NodeType.MLP, layer=0),
            CircuitNode(id="node2", label="Node 2", node_type=NodeType.ATTENTION, layer=1),
        )
        graph = CircuitGraph(name="Test", nodes=nodes)

        assert graph.get_node("node1") is not None
        assert graph.get_node("node1").label == "Node 1"
        assert graph.get_node("nonexistent") is None

    def test_get_layers(self):
        """Test get_layers method."""
        nodes = (
            CircuitNode(id="input", label="Input", node_type=NodeType.INPUT, layer=-1),
            CircuitNode(id="L0", label="Layer 0", node_type=NodeType.MLP, layer=0),
            CircuitNode(id="L5", label="Layer 5", node_type=NodeType.ATTENTION, layer=5),
            CircuitNode(id="L2", label="Layer 2", node_type=NodeType.MLP, layer=2),
        )
        graph = CircuitGraph(name="Test", nodes=nodes)

        layers = graph.get_layers()
        assert layers == [0, 2, 5]


# =============================================================================
# Tests for Circuit Creation
# =============================================================================


class TestCreateCircuitFromAblation:
    """Tests for create_circuit_from_ablation function."""

    def test_basic_creation(self):
        """Test basic circuit creation from ablation results."""
        ablation_results = [
            {"layer": 0, "component": "mlp", "effect": 0.5},
            {"layer": 5, "component": "attention", "effect": -0.3},
            {"layer": 10, "component": "mlp", "effect": 0.8},
        ]

        circuit = create_circuit_from_ablation(ablation_results, name="Ablation Circuit")

        assert circuit.name == "Ablation Circuit"
        assert circuit.num_nodes > 0
        assert circuit.num_edges > 0

    def test_threshold_filtering(self):
        """Test that low-effect components are filtered."""
        ablation_results = [
            {"layer": 0, "component": "mlp", "effect": 0.5},  # Above threshold
            {"layer": 1, "component": "mlp", "effect": 0.05},  # Below threshold
        ]

        circuit = create_circuit_from_ablation(ablation_results, threshold=0.1)

        # Should only have input, output, and L0_mlp
        layer_nodes = [n for n in circuit.nodes if n.layer >= 0]
        assert len(layer_nodes) == 1

    def test_causal_vs_inhibitory_edges(self):
        """Test that positive effects create causal edges."""
        ablation_results = [
            {"layer": 0, "component": "mlp", "effect": 0.5},  # Positive -> CAUSAL
            {"layer": 1, "component": "mlp", "effect": -0.5},  # Negative -> INHIBITORY
        ]

        circuit = create_circuit_from_ablation(ablation_results)

        causal_edges = [e for e in circuit.edges if e.edge_type == EdgeType.CAUSAL]
        inhibitory_edges = [e for e in circuit.edges if e.edge_type == EdgeType.INHIBITORY]

        assert len(causal_edges) >= 1
        assert len(inhibitory_edges) >= 1


class TestCreateCircuitFromDirections:
    """Tests for create_circuit_from_directions function."""

    def test_basic_creation(self):
        """Test basic circuit creation from directions."""
        directions = [
            {"layer": 10, "name": "arithmetic", "separation_score": 0.8},
            {"layer": 20, "name": "tool_calling", "separation_score": 0.6},
        ]

        circuit = create_circuit_from_directions(directions, name="Direction Circuit")

        assert circuit.name == "Direction Circuit"
        assert circuit.num_nodes > 0

    def test_steering_edges(self):
        """Test that direction nodes have steering edges."""
        directions = [
            {"layer": 5, "name": "test", "separation_score": 0.9},
        ]

        circuit = create_circuit_from_directions(directions)

        steering_edges = [e for e in circuit.edges if e.edge_type == EdgeType.STEERING]
        assert len(steering_edges) >= 1


# =============================================================================
# Tests for Export Functions
# =============================================================================


class TestExportToDot:
    """Tests for export_circuit_to_dot function."""

    def test_basic_export(self):
        """Test basic DOT export."""
        nodes = (
            CircuitNode(id="input", label="Input", node_type=NodeType.INPUT, layer=-1),
            CircuitNode(id="output", label="Output", node_type=NodeType.OUTPUT, layer=-1),
        )
        edges = (CircuitEdge(source="input", target="output", edge_type=EdgeType.RESIDUAL),)
        circuit = CircuitGraph(name="Test", nodes=nodes, edges=edges)

        dot_string = export_circuit_to_dot(circuit)

        assert "digraph" in dot_string
        assert "input" in dot_string
        assert "output" in dot_string
        assert "->" in dot_string

    def test_custom_colors(self):
        """Test custom color options."""
        nodes = (CircuitNode(id="mlp", label="MLP", node_type=NodeType.MLP, layer=0),)
        circuit = CircuitGraph(name="Test", nodes=nodes)

        custom_colors = {NodeType.MLP: "#FF0000"}
        dot_string = export_circuit_to_dot(circuit, node_colors=custom_colors)

        assert "#FF0000" in dot_string


class TestExportToJson:
    """Tests for export_circuit_to_json function."""

    def test_basic_export(self):
        """Test basic JSON export."""
        nodes = (CircuitNode(id="node1", label="Node 1", node_type=NodeType.MLP, layer=0),)
        edges = (CircuitEdge(source="node1", target="node1", edge_type=EdgeType.CAUSAL),)
        circuit = CircuitGraph(name="Test", nodes=nodes, edges=edges)

        json_string = export_circuit_to_json(circuit)
        data = json.loads(json_string)

        assert data["name"] == "Test"
        assert len(data["nodes"]) == 1
        assert len(data["edges"]) == 1

    def test_roundtrip(self):
        """Test export and reimport."""
        nodes = (CircuitNode(id="test", label="Test", node_type=NodeType.DIRECTION, layer=5),)
        circuit = CircuitGraph(name="Roundtrip Test", nodes=nodes)

        json_string = export_circuit_to_json(circuit)
        loaded = load_circuit_from_json(json_string)

        assert loaded.name == circuit.name
        assert len(loaded.nodes) == len(circuit.nodes)


class TestExportToMermaid:
    """Tests for export_circuit_to_mermaid function."""

    def test_basic_export(self):
        """Test basic Mermaid export."""
        nodes = (
            CircuitNode(id="input", label="Input", node_type=NodeType.INPUT, layer=-1),
            CircuitNode(id="output", label="Output", node_type=NodeType.OUTPUT, layer=-1),
        )
        edges = (CircuitEdge(source="input", target="output", edge_type=EdgeType.RESIDUAL),)
        circuit = CircuitGraph(name="Test", nodes=nodes, edges=edges)

        mermaid_string = export_circuit_to_mermaid(circuit)

        assert "graph" in mermaid_string
        assert "input" in mermaid_string
        assert "-->" in mermaid_string


class TestExportToHtml:
    """Tests for export_circuit_to_html function."""

    def test_basic_export(self):
        """Test basic HTML export."""
        nodes = (CircuitNode(id="node1", label="Node 1", node_type=NodeType.MLP, layer=0),)
        circuit = CircuitGraph(name="Test Circuit", nodes=nodes)

        html_string = export_circuit_to_html(circuit)

        assert "<!DOCTYPE html>" in html_string
        assert "Test Circuit" in html_string
        assert "vis.Network" in html_string


# =============================================================================
# Tests for File I/O
# =============================================================================


class TestSaveAndLoad:
    """Tests for save_circuit and load_circuit functions."""

    def test_save_json(self):
        """Test saving as JSON."""
        nodes = (CircuitNode(id="test", label="Test", node_type=NodeType.MLP, layer=0),)
        circuit = CircuitGraph(name="Save Test", nodes=nodes)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_circuit(circuit, path, format="json")
            loaded = load_circuit(path)

            assert loaded.name == circuit.name
        finally:
            path.unlink()

    def test_save_dot(self):
        """Test saving as DOT."""
        nodes = (CircuitNode(id="test", label="Test", node_type=NodeType.MLP, layer=0),)
        circuit = CircuitGraph(name="DOT Test", nodes=nodes)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as f:
            path = Path(f.name)

        try:
            save_circuit(circuit, path, format="dot")
            content = path.read_text()

            assert "digraph" in content
        finally:
            path.unlink()

    def test_save_mermaid(self):
        """Test saving as Mermaid."""
        nodes = (CircuitNode(id="test", label="Test", node_type=NodeType.MLP, layer=0),)
        circuit = CircuitGraph(name="Mermaid Test", nodes=nodes)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = Path(f.name)

        try:
            save_circuit(circuit, path, format="mermaid")
            content = path.read_text()

            assert "graph" in content
        finally:
            path.unlink()

    def test_save_html(self):
        """Test saving as HTML."""
        nodes = (CircuitNode(id="test", label="Test", node_type=NodeType.MLP, layer=0),)
        circuit = CircuitGraph(name="HTML Test", nodes=nodes)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            path = Path(f.name)

        try:
            save_circuit(circuit, path, format="html")
            content = path.read_text()

            assert "<!DOCTYPE html>" in content
        finally:
            path.unlink()

    def test_invalid_format(self):
        """Test error on invalid format."""
        circuit = CircuitGraph(name="Test")

        with pytest.raises(ValueError, match="Unknown format"):
            save_circuit(circuit, "test.xyz", format="xyz")
