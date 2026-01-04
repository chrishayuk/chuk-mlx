"""Introspection CLI commands organized by category."""

# Core analysis commands
# Ablation study commands
from .ablation import (
    introspect_ablate,
    introspect_activation_diff,
    introspect_weight_diff,
)
from .analyze import (
    introspect_analyze,
    introspect_compare,
    introspect_hooks,
)

# Arithmetic study commands
from .arithmetic import introspect_arithmetic

# Circuit analysis commands
from .circuit import (
    introspect_circuit_capture,
    introspect_circuit_compare,
    introspect_circuit_decode,
    introspect_circuit_export,
    introspect_circuit_invoke,
    introspect_circuit_test,
    introspect_circuit_view,
)

# Clustering commands
from .clustering import introspect_activation_cluster

# Embedding analysis commands
from .embedding import (
    introspect_early_layers,
    introspect_embedding,
)

# Generation commands
from .generation import introspect_generate

# Layer analysis commands
from .layer import (
    introspect_format_sensitivity,
    introspect_layer,
)

# Memory commands
from .memory import (
    introspect_memory,
    introspect_memory_inject,
)

# MoE expert manipulation commands (modular package)
from .moe_expert import introspect_moe_expert

# Neuron and direction analysis commands
from .neurons import (
    introspect_directions,
    introspect_neurons,
    introspect_operand_directions,
)

# Causal intervention commands
from .patching import (
    introspect_commutativity,
    introspect_patch,
)

# Probing and uncertainty detection commands
from .probing import (
    introspect_metacognitive,
    introspect_probe,
    introspect_uncertainty,
)

# Steering commands
from .steering import introspect_steer

# Virtual expert commands
from .virtual_expert import introspect_virtual_expert

__all__ = [
    # Core analysis
    "introspect_analyze",
    "introspect_compare",
    "introspect_hooks",
    # Ablation
    "introspect_ablate",
    "introspect_weight_diff",
    "introspect_activation_diff",
    # Steering
    "introspect_steer",
    "introspect_neurons",
    "introspect_directions",
    "introspect_operand_directions",
    # Circuit
    "introspect_arithmetic",
    "introspect_commutativity",
    "introspect_patch",
    "introspect_circuit_capture",
    "introspect_circuit_invoke",
    "introspect_circuit_test",
    "introspect_circuit_view",
    "introspect_circuit_compare",
    "introspect_circuit_decode",
    "introspect_circuit_export",
    # Layer
    "introspect_layer",
    "introspect_format_sensitivity",
    "introspect_embedding",
    "introspect_early_layers",
    "introspect_activation_cluster",
    # Memory
    "introspect_memory",
    "introspect_memory_inject",
    # Generation
    "introspect_generate",
    "introspect_metacognitive",
    "introspect_probe",
    "introspect_uncertainty",
    # Virtual Expert
    "introspect_virtual_expert",
    # MoE Expert
    "introspect_moe_expert",
]
