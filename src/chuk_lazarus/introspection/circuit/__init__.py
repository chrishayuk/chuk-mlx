"""
Circuit analysis tools for mechanistic interpretability.

This module provides generic infrastructure for:
- Dataset creation (any binary/multi-class task)
- Activation collection and storage
- Linear probe batteries for layer-wise analysis
- Direction extraction for activation steering
- Geometry analysis (PCA, UMAP)

Example use cases:
- Gemma arithmetic suppression circuits
- Tool-calling circuits (FunctionGemma)
- Factual consistency circuits
- Safety/alignment circuits

CLI Usage:
    circuit dataset create -o prompts.json
    circuit collect -m gemma-3-4b-it -d prompts.json -o activations
    circuit probes -m gemma-3-4b-it --layers 0,10,20,24,30
    circuit directions -a activations.safetensors --layer 24
"""

from .collector import (
    ActivationCollector,
    CollectedActivations,
    CollectorConfig,
    collect_activations,
)
from .dataset import (
    CircuitDataset,
    ContrastivePair,
    # Generic (recommended)
    LabeledPrompt,
    # Backwards compatibility
    PromptCategory,
    ToolPrompt,
    ToolPromptDataset,
    # Domain-specific datasets
    create_arithmetic_dataset,
    create_binary_dataset,
    create_code_execution_dataset,
    create_contrastive_dataset,
    create_factual_consistency_dataset,
    create_tool_calling_dataset,
    create_tool_delegation_dataset,
)
from .directions import (
    DirectionBundle,
    DirectionExtractor,
    DirectionMethod,
    ExtractedDirection,
    extract_all_directions,
    extract_direction,
)
from .export import (
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
from .probes import (
    ProbeBattery,
    ProbeDataset,
    ProbeResult,
    StratigraphyResult,
    # Pre-built probes
    create_arithmetic_probe,
    create_code_trace_probe,
    create_factual_consistency_probe,
    create_suppression_probe,
    create_tool_decision_probe,
    get_default_probe_datasets,
)

# Optional geometry imports (may require additional dependencies)
try:
    from .geometry import (  # noqa: F401
        GeometryAnalyzer,
        GeometryResult,
        ProbeType,
        compute_pca,
        compute_umap,
        train_linear_probe,
    )

    _HAS_GEOMETRY = True
except ImportError:
    _HAS_GEOMETRY = False


__all__ = [
    # Generic dataset (recommended)
    "LabeledPrompt",
    "ContrastivePair",
    "CircuitDataset",
    "create_binary_dataset",
    "create_contrastive_dataset",
    # Domain-specific datasets
    "create_arithmetic_dataset",
    "create_code_execution_dataset",
    "create_factual_consistency_dataset",
    "create_tool_delegation_dataset",
    # Backwards compatibility
    "PromptCategory",
    "ToolPrompt",
    "ToolPromptDataset",
    "create_tool_calling_dataset",
    # Collector
    "ActivationCollector",
    "CollectorConfig",
    "CollectedActivations",
    "collect_activations",
    # Directions
    "DirectionExtractor",
    "DirectionBundle",
    "ExtractedDirection",
    "DirectionMethod",
    "extract_direction",
    "extract_all_directions",
    # Export
    "CircuitGraph",
    "CircuitNode",
    "CircuitEdge",
    "NodeType",
    "EdgeType",
    "create_circuit_from_ablation",
    "create_circuit_from_directions",
    "export_circuit_to_dot",
    "export_circuit_to_json",
    "export_circuit_to_mermaid",
    "export_circuit_to_html",
    "save_circuit",
    "load_circuit",
    "load_circuit_from_json",
    # Probes
    "ProbeBattery",
    "ProbeDataset",
    "ProbeResult",
    "StratigraphyResult",
    "create_arithmetic_probe",
    "create_code_trace_probe",
    "create_factual_consistency_probe",
    "create_tool_decision_probe",
    "create_suppression_probe",
    "get_default_probe_datasets",
]

# Add geometry exports if available
if _HAS_GEOMETRY:
    __all__.extend(
        [
            "GeometryAnalyzer",
            "GeometryResult",
            "ProbeType",
            "train_linear_probe",
            "compute_pca",
            "compute_umap",
        ]
    )
