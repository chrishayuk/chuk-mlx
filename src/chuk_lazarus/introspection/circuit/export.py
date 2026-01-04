"""
Circuit graph export utilities.

Provides tools for exporting discovered circuits to various formats:
- DOT (Graphviz)
- JSON graph format
- Mermaid diagrams
- HTML interactive visualization

Example:
    >>> from chuk_lazarus.introspection.circuit import DirectionBundle
    >>> from chuk_lazarus.introspection.circuit.export import (
    ...     export_circuit_to_dot,
    ...     export_circuit_to_json,
    ...     export_circuit_to_html,
    ... )
    >>>
    >>> # Export ablation results as a circuit graph
    >>> dot_string = export_circuit_to_dot(ablation_results, directions)
    >>> with open("circuit.dot", "w") as f:
    ...     f.write(dot_string)
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class NodeType(str, Enum):
    """Type of node in a circuit graph."""

    LAYER = "layer"
    ATTENTION = "attention"
    MLP = "mlp"
    EXPERT = "expert"
    DIRECTION = "direction"
    INPUT = "input"
    OUTPUT = "output"


class EdgeType(str, Enum):
    """Type of edge in a circuit graph."""

    RESIDUAL = "residual"
    ATTENTION_OUT = "attention_out"
    MLP_OUT = "mlp_out"
    CAUSAL = "causal"
    INHIBITORY = "inhibitory"
    STEERING = "steering"


class CircuitNode(BaseModel):
    """A node in a circuit graph."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Unique node identifier")
    label: str = Field(description="Display label")
    node_type: NodeType = Field(description="Type of node")
    layer: int = Field(ge=-1, description="Layer index (-1 for input/output)")
    importance: float = Field(ge=0, le=1, default=1.0, description="Node importance")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional data")


class CircuitEdge(BaseModel):
    """An edge in a circuit graph."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    edge_type: EdgeType = Field(description="Type of edge")
    weight: float = Field(default=1.0, description="Edge weight/strength")
    label: str = Field(default="", description="Edge label")
    metadata: dict[str, Any] = Field(default_factory=dict)


class CircuitGraph(BaseModel):
    """A complete circuit graph."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Circuit name")
    description: str = Field(default="", description="Circuit description")
    nodes: tuple[CircuitNode, ...] = Field(default_factory=tuple)
    edges: tuple[CircuitEdge, ...] = Field(default_factory=tuple)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def get_node(self, node_id: str) -> CircuitNode | None:
        """Get node by ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_layers(self) -> list[int]:
        """Get unique layer indices in the circuit."""
        layers = set()
        for node in self.nodes:
            if node.layer >= 0:
                layers.add(node.layer)
        return sorted(layers)


# =============================================================================
# Circuit Building
# =============================================================================


def create_circuit_from_ablation(
    ablation_results: list[dict[str, Any]],
    name: str = "Ablation Circuit",
    threshold: float = 0.1,
) -> CircuitGraph:
    """
    Create a circuit graph from ablation study results.

    Args:
        ablation_results: List of ablation results with layer, component, effect
        name: Name for the circuit
        threshold: Minimum effect to include in circuit

    Returns:
        CircuitGraph representing causal components
    """
    nodes: list[CircuitNode] = []
    edges: list[CircuitEdge] = []

    # Add input/output nodes
    nodes.append(
        CircuitNode(
            id="input",
            label="Input",
            node_type=NodeType.INPUT,
            layer=-1,
        )
    )
    nodes.append(
        CircuitNode(
            id="output",
            label="Output",
            node_type=NodeType.OUTPUT,
            layer=-1,
        )
    )

    # Process ablation results
    for result in ablation_results:
        layer = result.get("layer", 0)
        component = result.get("component", "mlp")
        effect = result.get("effect", 0.0)

        if abs(effect) < threshold:
            continue

        # Create node
        node_type = NodeType.ATTENTION if "attn" in component.lower() else NodeType.MLP
        node_id = f"L{layer}_{component}"

        nodes.append(
            CircuitNode(
                id=node_id,
                label=f"L{layer} {component}",
                node_type=node_type,
                layer=layer,
                importance=min(1.0, abs(effect)),
                metadata={"effect": effect, "component": component},
            )
        )

        # Create edge based on effect direction
        edge_type = EdgeType.CAUSAL if effect > 0 else EdgeType.INHIBITORY

        edges.append(
            CircuitEdge(
                source=node_id,
                target="output",
                edge_type=edge_type,
                weight=abs(effect),
                label=f"{effect:+.2f}",
            )
        )

    # Add residual connections between layers
    layer_nodes = sorted([n for n in nodes if n.layer >= 0], key=lambda n: n.layer)

    prev_node = "input"
    for node in layer_nodes:
        edges.append(
            CircuitEdge(
                source=prev_node,
                target=node.id,
                edge_type=EdgeType.RESIDUAL,
                weight=1.0,
            )
        )
        prev_node = node.id

    if prev_node != "input":
        edges.append(
            CircuitEdge(
                source=prev_node,
                target="output",
                edge_type=EdgeType.RESIDUAL,
                weight=1.0,
            )
        )

    return CircuitGraph(
        name=name,
        description=f"Circuit from ablation study ({len(nodes)} components)",
        nodes=tuple(nodes),
        edges=tuple(edges),
    )


def create_circuit_from_directions(
    directions: list[dict[str, Any]],
    name: str = "Direction Circuit",
) -> CircuitGraph:
    """
    Create a circuit graph from extracted directions.

    Args:
        directions: List of direction info with layer, separation_score, etc.
        name: Name for the circuit

    Returns:
        CircuitGraph representing direction-based circuit
    """
    nodes: list[CircuitNode] = []
    edges: list[CircuitEdge] = []

    # Add input/output
    nodes.append(
        CircuitNode(
            id="input",
            label="Input",
            node_type=NodeType.INPUT,
            layer=-1,
        )
    )
    nodes.append(
        CircuitNode(
            id="output",
            label="Output",
            node_type=NodeType.OUTPUT,
            layer=-1,
        )
    )

    # Add direction nodes
    for i, direction in enumerate(directions):
        layer = direction.get("layer", i)
        separation = direction.get("separation_score", 0.5)
        direction_name = direction.get("name", f"Direction {i}")

        node_id = f"dir_L{layer}"

        nodes.append(
            CircuitNode(
                id=node_id,
                label=f"L{layer}: {direction_name}",
                node_type=NodeType.DIRECTION,
                layer=layer,
                importance=min(1.0, abs(separation)),
                metadata=direction,
            )
        )

        # Connect to output
        edges.append(
            CircuitEdge(
                source=node_id,
                target="output",
                edge_type=EdgeType.STEERING,
                weight=abs(separation),
                label=f"sep={separation:.2f}",
            )
        )

    # Chain layers
    sorted_nodes = sorted([n for n in nodes if n.layer >= 0], key=lambda n: n.layer)

    prev = "input"
    for node in sorted_nodes:
        edges.append(
            CircuitEdge(
                source=prev,
                target=node.id,
                edge_type=EdgeType.RESIDUAL,
            )
        )
        prev = node.id

    return CircuitGraph(
        name=name,
        nodes=tuple(nodes),
        edges=tuple(edges),
    )


# =============================================================================
# DOT Export
# =============================================================================


def export_circuit_to_dot(
    circuit: CircuitGraph,
    rankdir: str = "TB",
    node_colors: dict[NodeType, str] | None = None,
    edge_colors: dict[EdgeType, str] | None = None,
) -> str:
    """
    Export circuit to DOT format for Graphviz.

    Args:
        circuit: The circuit graph
        rankdir: Graph direction (TB=top-bottom, LR=left-right)
        node_colors: Custom node colors by type
        edge_colors: Custom edge colors by type

    Returns:
        DOT format string
    """
    # Default colors
    if node_colors is None:
        node_colors = {
            NodeType.INPUT: "#90EE90",
            NodeType.OUTPUT: "#FFB6C1",
            NodeType.ATTENTION: "#87CEEB",
            NodeType.MLP: "#DDA0DD",
            NodeType.EXPERT: "#F0E68C",
            NodeType.DIRECTION: "#FFA07A",
            NodeType.LAYER: "#D3D3D3",
        }

    if edge_colors is None:
        edge_colors = {
            EdgeType.RESIDUAL: "#808080",
            EdgeType.CAUSAL: "#228B22",
            EdgeType.INHIBITORY: "#DC143C",
            EdgeType.ATTENTION_OUT: "#4169E1",
            EdgeType.MLP_OUT: "#9932CC",
            EdgeType.STEERING: "#FF8C00",
        }

    lines = [
        f'digraph "{circuit.name}" {{',
        f"    rankdir={rankdir};",
        '    node [shape=box, style="rounded,filled"];',
        "",
    ]

    # Add nodes
    for node in circuit.nodes:
        color = node_colors.get(node.node_type, "#FFFFFF")
        # Scale size by importance
        width = 0.5 + node.importance * 1.0
        height = 0.3 + node.importance * 0.5

        lines.append(
            f'    "{node.id}" ['
            f'label="{node.label}", '
            f'fillcolor="{color}", '
            f"width={width:.2f}, "
            f"height={height:.2f}"
            f"];"
        )

    lines.append("")

    # Add edges
    for edge in circuit.edges:
        color = edge_colors.get(edge.edge_type, "#000000")
        penwidth = 1.0 + edge.weight * 2.0
        style = "dashed" if edge.edge_type == EdgeType.RESIDUAL else "solid"

        label_str = f', label="{edge.label}"' if edge.label else ""

        lines.append(
            f'    "{edge.source}" -> "{edge.target}" ['
            f'color="{color}", '
            f"penwidth={penwidth:.1f}, "
            f"style={style}"
            f"{label_str}"
            f"];"
        )

    lines.append("}")

    return "\n".join(lines)


# =============================================================================
# JSON Export
# =============================================================================


def export_circuit_to_json(
    circuit: CircuitGraph,
    indent: int = 2,
) -> str:
    """
    Export circuit to JSON format.

    Args:
        circuit: The circuit graph
        indent: JSON indentation

    Returns:
        JSON string
    """
    data = {
        "name": circuit.name,
        "description": circuit.description,
        "metadata": circuit.metadata,
        "nodes": [
            {
                "id": n.id,
                "label": n.label,
                "type": n.node_type.value,
                "layer": n.layer,
                "importance": n.importance,
                "metadata": n.metadata,
            }
            for n in circuit.nodes
        ],
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "type": e.edge_type.value,
                "weight": e.weight,
                "label": e.label,
                "metadata": e.metadata,
            }
            for e in circuit.edges
        ],
    }

    return json.dumps(data, indent=indent)


def load_circuit_from_json(json_str: str) -> CircuitGraph:
    """
    Load circuit from JSON string.

    Args:
        json_str: JSON string

    Returns:
        CircuitGraph
    """
    data = json.loads(json_str)

    nodes = tuple(
        CircuitNode(
            id=n["id"],
            label=n["label"],
            node_type=NodeType(n["type"]),
            layer=n["layer"],
            importance=n.get("importance", 1.0),
            metadata=n.get("metadata", {}),
        )
        for n in data["nodes"]
    )

    edges = tuple(
        CircuitEdge(
            source=e["source"],
            target=e["target"],
            edge_type=EdgeType(e["type"]),
            weight=e.get("weight", 1.0),
            label=e.get("label", ""),
            metadata=e.get("metadata", {}),
        )
        for e in data["edges"]
    )

    return CircuitGraph(
        name=data["name"],
        description=data.get("description", ""),
        nodes=nodes,
        edges=edges,
        metadata=data.get("metadata", {}),
    )


# =============================================================================
# Mermaid Export
# =============================================================================


def export_circuit_to_mermaid(
    circuit: CircuitGraph,
    direction: str = "TB",
) -> str:
    """
    Export circuit to Mermaid diagram format.

    Args:
        circuit: The circuit graph
        direction: Diagram direction (TB, LR, BT, RL)

    Returns:
        Mermaid diagram string
    """
    lines = [f"graph {direction}"]

    # Define node shapes based on type
    shape_map = {
        NodeType.INPUT: ("([", "])"),  # Stadium
        NodeType.OUTPUT: ("([", "])"),
        NodeType.ATTENTION: ("{", "}"),  # Diamond-ish
        NodeType.MLP: ("[", "]"),  # Rectangle
        NodeType.EXPERT: ("((", "))"),  # Circle
        NodeType.DIRECTION: (">", "]"),  # Flag
        NodeType.LAYER: ("[", "]"),
    }

    # Add nodes
    for node in circuit.nodes:
        left, right = shape_map.get(node.node_type, ("[", "]"))
        # Escape quotes in label
        label = node.label.replace('"', "'")
        lines.append(f'    {node.id}{left}"{label}"{right}')

    # Add edges
    for edge in circuit.edges:
        arrow = "-->" if edge.edge_type != EdgeType.INHIBITORY else "-.->"
        if edge.label:
            lines.append(f"    {edge.source} {arrow}|{edge.label}| {edge.target}")
        else:
            lines.append(f"    {edge.source} {arrow} {edge.target}")

    return "\n".join(lines)


# =============================================================================
# HTML Export
# =============================================================================


def export_circuit_to_html(
    circuit: CircuitGraph,
    title: str | None = None,
    width: int = 800,
    height: int = 600,
) -> str:
    """
    Export circuit to interactive HTML using vis.js.

    Args:
        circuit: The circuit graph
        title: Page title
        width: Canvas width
        height: Canvas height

    Returns:
        HTML string
    """
    title = title or circuit.name

    # Build node and edge data for vis.js
    nodes_data = []
    for node in circuit.nodes:
        color_map = {
            NodeType.INPUT: "#90EE90",
            NodeType.OUTPUT: "#FFB6C1",
            NodeType.ATTENTION: "#87CEEB",
            NodeType.MLP: "#DDA0DD",
            NodeType.EXPERT: "#F0E68C",
            NodeType.DIRECTION: "#FFA07A",
            NodeType.LAYER: "#D3D3D3",
        }
        color = color_map.get(node.node_type, "#FFFFFF")
        size = 20 + node.importance * 30

        nodes_data.append(
            {
                "id": node.id,
                "label": node.label,
                "color": color,
                "size": size,
                "title": f"Layer: {node.layer}<br>Type: {node.node_type.value}<br>Importance: {node.importance:.2f}",
            }
        )

    edges_data = []
    for edge in circuit.edges:
        color_map = {
            EdgeType.RESIDUAL: "#808080",
            EdgeType.CAUSAL: "#228B22",
            EdgeType.INHIBITORY: "#DC143C",
            EdgeType.ATTENTION_OUT: "#4169E1",
            EdgeType.MLP_OUT: "#9932CC",
            EdgeType.STEERING: "#FF8C00",
        }
        color = color_map.get(edge.edge_type, "#000000")
        dashes = edge.edge_type == EdgeType.RESIDUAL

        edges_data.append(
            {
                "from": edge.source,
                "to": edge.target,
                "color": color,
                "width": 1 + edge.weight * 3,
                "dashes": dashes,
                "arrows": "to",
                "title": edge.label if edge.label else edge.edge_type.value,
            }
        )

    nodes_json = json.dumps(nodes_data)
    edges_json = json.dumps(edges_data)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        h1 {{
            margin-bottom: 10px;
        }}
        #circuit {{
            width: {width}px;
            height: {height}px;
            border: 1px solid #ccc;
        }}
        .legend {{
            margin-top: 10px;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
        }}
        .info {{
            margin-top: 10px;
            color: #666;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p class="info">{circuit.description}</p>
    <div id="circuit"></div>
    <div class="legend">
        <div class="legend-item"><div class="legend-color" style="background:#90EE90"></div>Input</div>
        <div class="legend-item"><div class="legend-color" style="background:#FFB6C1"></div>Output</div>
        <div class="legend-item"><div class="legend-color" style="background:#87CEEB"></div>Attention</div>
        <div class="legend-item"><div class="legend-color" style="background:#DDA0DD"></div>MLP</div>
        <div class="legend-item"><div class="legend-color" style="background:#F0E68C"></div>Expert</div>
        <div class="legend-item"><div class="legend-color" style="background:#FFA07A"></div>Direction</div>
    </div>
    <p class="info">Nodes: {circuit.num_nodes} | Edges: {circuit.num_edges}</p>

    <script>
        var nodes = new vis.DataSet({nodes_json});
        var edges = new vis.DataSet({edges_json});

        var container = document.getElementById('circuit');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            layout: {{
                hierarchical: {{
                    direction: 'UD',
                    sortMethod: 'directed',
                    levelSeparation: 100,
                }}
            }},
            physics: false,
            interaction: {{
                hover: true,
                tooltipDelay: 100,
            }},
        }};

        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>"""

    return html


# =============================================================================
# File I/O
# =============================================================================


def save_circuit(
    circuit: CircuitGraph,
    path: str | Path,
    format: str = "json",
) -> None:
    """
    Save circuit to file.

    Args:
        circuit: The circuit graph
        path: Output file path
        format: Output format (json, dot, mermaid, html)
    """
    path = Path(path)

    if format == "json":
        content = export_circuit_to_json(circuit)
    elif format == "dot":
        content = export_circuit_to_dot(circuit)
    elif format == "mermaid":
        content = export_circuit_to_mermaid(circuit)
    elif format == "html":
        content = export_circuit_to_html(circuit)
    else:
        raise ValueError(f"Unknown format: {format}")

    path.write_text(content)


def load_circuit(path: str | Path) -> CircuitGraph:
    """
    Load circuit from JSON file.

    Args:
        path: Input file path

    Returns:
        CircuitGraph
    """
    path = Path(path)
    content = path.read_text()
    return load_circuit_from_json(content)
