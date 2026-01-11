"""Circuit capture and manipulation command handlers for introspection CLI.

This module provides thin CLI wrappers for circuit analysis commands.
All business logic is delegated to the framework layer (introspection module).
"""

from __future__ import annotations

import logging
from argparse import Namespace

from .._constants import CircuitDefaults, OutputFormat
from ._utils import (
    extract_arg,
    parse_prompts,
    parse_value_list,
    require_arg,
)

logger = logging.getLogger(__name__)


async def introspect_circuit_capture(args: Namespace) -> None:
    """Capture circuit activations and extract computational directions.

    This is a thin wrapper that:
    1. Converts CLI args to CircuitCaptureConfig
    2. Calls CircuitService.capture() which handles all logic
    3. Saves and displays results

    Args:
        args: Parsed command-line arguments
    """
    from ....introspection.circuit import CircuitCaptureConfig, CircuitService

    # Validate required args
    layer = require_arg(args, "layer", "Must specify --layer for circuit capture")
    prompts = parse_prompts(args.prompts)

    # Parse results if provided (using shared utility)
    results = None
    results_arg = extract_arg(args, "results")
    if results_arg:
        results = parse_value_list(results_arg, value_type=int)
        if len(results) != len(prompts):
            raise ValueError(f"{len(results)} results for {len(prompts)} prompts")

    config = CircuitCaptureConfig(
        model=args.model,
        prompts=prompts,
        layer=layer,
        results=results,
        extract_direction=extract_arg(args, "extract_direction", False),
        output_path=extract_arg(args, "output"),
    )

    # Run capture - all logic is in the service
    result = await CircuitService.capture(config)

    # Print formatted result
    print(result.to_display())


async def introspect_circuit_invoke(args: Namespace) -> None:
    """Invoke a captured circuit on new prompts.

    Args:
        args: Parsed command-line arguments
    """
    from ....introspection.circuit import CircuitInvokeConfig, CircuitService
    from ....introspection.enums import InvocationMethod
    from .._constants import AnalysisDefaults

    prompts = parse_prompts(args.prompts)

    # Parse method
    method = InvocationMethod(extract_arg(args, "method", InvocationMethod.STEER.value))

    # Parse coefficient - required for interpolate/extrapolate
    coefficient = extract_arg(args, "coefficient")
    if method in [InvocationMethod.INTERPOLATE, InvocationMethod.EXTRAPOLATE]:
        if coefficient is None:
            raise ValueError(f"--coefficient required for {method.value} method")

    config = CircuitInvokeConfig(
        model=args.model,
        circuit_file=args.circuit,
        prompts=prompts,
        method=method,
        coefficient=coefficient,
        layer=extract_arg(args, "layer"),
        top_k=extract_arg(args, "top_k", AnalysisDefaults.TOP_K),
    )

    # Run invocation - all logic is in the service
    result = await CircuitService.invoke(config)

    # Print formatted result
    print(result.to_display())


async def introspect_circuit_test(args: Namespace) -> None:
    """Test circuit prediction accuracy on test prompts.

    Args:
        args: Parsed command-line arguments
    """
    from ....introspection.circuit import CircuitService, CircuitTestConfig

    prompts = parse_prompts(args.prompts)

    # Parse expected results (using shared utility)
    results = None
    results_arg = extract_arg(args, "results")
    if results_arg:
        results = parse_value_list(results_arg, value_type=int)

    config = CircuitTestConfig(
        model=args.model,
        circuit_file=args.circuit,
        prompts=prompts,
        expected_results=results,
        threshold=CircuitDefaults.THRESHOLD,
    )

    # Run test - all logic is in the service
    result = await CircuitService.test(config)

    # Print formatted result
    print(result.to_display())


async def introspect_circuit_view(args: Namespace) -> None:
    """View contents of a captured circuit file.

    Args:
        args: Parsed command-line arguments
    """
    from ....introspection.circuit import CircuitService, CircuitViewConfig

    config = CircuitViewConfig(
        circuit_file=args.circuit,
        show_activations=extract_arg(args, "show_activations", False),
        show_direction=extract_arg(args, "show_direction", True),
    )

    # View circuit - all logic is in the service
    result = await CircuitService.view(config)

    # Print formatted result
    print(result.to_display())


async def introspect_circuit_compare(args: Namespace) -> None:
    """Compare two captured circuits.

    Args:
        args: Parsed command-line arguments
    """
    from ....introspection.circuit import CircuitCompareConfig, CircuitService

    config = CircuitCompareConfig(
        circuit_file_a=args.circuit_a,
        circuit_file_b=args.circuit_b,
    )

    # Compare circuits - all logic is in the service
    result = await CircuitService.compare(config)

    # Print formatted result
    print(result.to_display())


async def introspect_circuit_decode(args: Namespace) -> None:
    """Decode a circuit direction through the model's vocabulary.

    Args:
        args: Parsed command-line arguments
    """
    from ....introspection.circuit import CircuitDecodeConfig, CircuitService

    config = CircuitDecodeConfig(
        model=args.model,
        circuit_file=args.circuit,
        top_k=extract_arg(args, "top_k", 20),
    )

    # Decode circuit - all logic is in the service
    result = await CircuitService.decode(config)

    # Print formatted result
    print(result.to_display())


async def introspect_circuit_export(args: Namespace) -> None:
    """Export circuit in various formats (JSON, DOT, Mermaid).

    Args:
        args: Parsed command-line arguments
    """
    from ....introspection.circuit import CircuitExportConfig, CircuitService

    output_format = OutputFormat(extract_arg(args, "format", OutputFormat.JSON.value))

    config = CircuitExportConfig(
        circuit_file=args.circuit,
        output_path=extract_arg(args, "output"),
        output_format=output_format,
        direction=CircuitDefaults.DIRECTION,
    )

    # Export circuit - all logic is in the service
    result = await CircuitService.export(config)

    # Print or save result
    if config.output_path:
        print(f"Exported to: {config.output_path}")
    else:
        print(result.content)


__all__ = [
    "introspect_circuit_capture",
    "introspect_circuit_compare",
    "introspect_circuit_decode",
    "introspect_circuit_export",
    "introspect_circuit_invoke",
    "introspect_circuit_test",
    "introspect_circuit_view",
]
