"""Memory analysis command handlers for introspection CLI.

This module provides thin CLI wrappers for memory analysis commands.
All business logic is delegated to the framework layer (introspection module).

IMPORTANT: CLI commands should NOT contain hardcoded sample data.
Use --facts-file or framework-level dataset loaders instead.
"""

from __future__ import annotations

import logging
from argparse import Namespace

from ....datasets import FactType
from .._constants import AnalysisDefaults, LayerDepthRatio, MemoryDefaults
from ._utils import extract_arg, get_layer_depth_ratio, load_json_file

logger = logging.getLogger(__name__)


async def introspect_memory(args: Namespace) -> None:
    """Analyze model's memory of facts.

    This is a thin wrapper that:
    1. Loads facts from file or uses framework defaults
    2. Calls MemoryAnalysisService.analyze() which handles all logic
    3. Formats and prints results

    Args:
        args: Parsed command-line arguments
    """
    from ....datasets import load_facts
    from ....introspection.memory import MemoryAnalysisConfig, MemoryAnalysisService

    # Determine fact source
    fact_type_str = extract_arg(args, "facts", "multiplication")

    # Load facts from file or use framework datasets
    if fact_type_str.startswith("@"):
        facts = load_json_file(fact_type_str)
        fact_type = FactType.CUSTOM
    else:
        fact_type = FactType(fact_type_str)
        facts = load_facts(fact_type)

    # Get layer config
    layer = extract_arg(args, "layer")

    # Build config from CLI args
    config = MemoryAnalysisConfig(
        model=args.model,
        facts=facts,
        fact_type=fact_type,
        layer=layer,
        layer_depth_ratio=get_layer_depth_ratio(layer, LayerDepthRatio.DEEP),
        top_k=extract_arg(args, "top_k", AnalysisDefaults.TOP_K),
        classify=extract_arg(args, "classify", False),
        # Memorization thresholds from constants
        memorized_prob_threshold=MemoryDefaults.MEMORIZED_PROB_THRESHOLD,
        partial_prob_threshold=MemoryDefaults.PARTIAL_PROB_THRESHOLD,
        weak_prob_threshold=MemoryDefaults.WEAK_PROB_THRESHOLD,
        memorized_rank=MemoryDefaults.MEMORIZED_RANK,
        partial_rank=MemoryDefaults.PARTIAL_RANK,
        weak_rank=MemoryDefaults.WEAK_RANK,
    )

    # Run analysis - all logic is in the service
    result = await MemoryAnalysisService.analyze(config)

    # Print formatted result
    print(result.to_display())

    # Save results if requested
    output_path = extract_arg(args, "output")
    if output_path:
        result.save(output_path)
        print(f"\nDetailed results saved to: {output_path}")

    # Save plot if requested
    plot_path = extract_arg(args, "save_plot")
    if plot_path:
        result.save_plot(plot_path)
        print(f"Plot saved to: {plot_path}")


async def introspect_memory_inject(args: Namespace) -> None:
    """External memory injection for fact retrieval.

    Builds an external memory store from known facts and uses it to
    inject correct answers at inference time. This can rescue queries
    that the model would otherwise get wrong.

    Args:
        args: Parsed command-line arguments
    """
    from ....datasets import load_facts
    from ....introspection.external_memory import ExternalMemory, MemoryConfig

    from .._base import OutputMixin
    from .._constants import Delimiters

    # Configure memory layers from constants
    query_layer = extract_arg(args, "query_layer", MemoryDefaults.DEFAULT_QUERY_LAYER)
    inject_layer = extract_arg(args, "inject_layer", MemoryDefaults.DEFAULT_INJECT_LAYER)
    blend = extract_arg(args, "blend", MemoryDefaults.BLEND)
    threshold = extract_arg(args, "threshold", MemoryDefaults.SIMILARITY_THRESHOLD)

    memory_config = MemoryConfig(
        query_layer=query_layer,
        inject_layer=inject_layer,
        value_layer=query_layer,
        blend=blend,
        similarity_threshold=threshold,
    )

    # Create memory system
    memory = ExternalMemory.from_pretrained(args.model, memory_config)

    # Load facts from file or use framework datasets
    fact_type_str = extract_arg(args, "facts", "multiplication")
    if fact_type_str.startswith("@"):
        facts = load_json_file(fact_type_str)
    else:
        fact_type = FactType(fact_type_str)
        facts = load_facts(fact_type)
    memory.add_facts(facts)

    # Save/load store if requested
    save_store = extract_arg(args, "save_store")
    if save_store:
        memory.save(save_store)

    load_store = extract_arg(args, "load_store")
    if load_store:
        memory.load(load_store)

    # Process queries
    queries = []
    query_arg = extract_arg(args, "query")
    queries_arg = extract_arg(args, "queries")
    if query_arg:
        queries = [query_arg]
    elif queries_arg:
        queries = queries_arg.split(Delimiters.PROMPT_SEPARATOR)

    if not queries:
        print("\nNo queries provided. Use --query or --queries")
        print(f"Memory store has {memory.num_entries} entries")
        return

    # Run queries - all logic is in the memory system
    force = extract_arg(args, "force", False)
    results = memory.query_batch(queries, use_injection=True, force_injection=force)

    # Print formatted results
    print(OutputMixin.format_header("EXTERNAL MEMORY INJECTION", width=70))

    for query, result in zip(queries, results):
        print(f"\nQuery: '{query}'")
        print(f"  Baseline: '{result.baseline_answer}' ({result.baseline_confidence:.1%})")

        if result.used_injection:
            print(f"  Injected: '{result.injected_answer}' ({result.injected_confidence:.1%})")
            if result.matched_entry:
                print(f"  Matched:  '{result.matched_entry.query}' -> {result.matched_entry.answer}")
                print(f"  Similarity: {result.similarity:.3f}")

            if result.baseline_answer.strip() != result.injected_answer.strip():
                print("  Status: MODIFIED")
        else:
            if result.matched_entry:
                print(f"  Matched:  '{result.matched_entry.query}' (sim={result.similarity:.3f})")
                print(f"  Status: Below threshold ({threshold}), no injection")
            else:
                print("  Status: No match found")

    # Evaluate mode
    if extract_arg(args, "evaluate", False):
        metrics = memory.evaluate(facts, verbose=False)
        print(OutputMixin.format_header("EVALUATION", width=70))
        print(f"\nBaseline accuracy: {metrics['baseline_accuracy']:.1%}")
        print(f"Injected accuracy: {metrics['injected_accuracy']:.1%}")
        print(f"Rescued: {metrics['rescued']}")
        print(f"Broken: {metrics['broken']}")


__all__ = [
    "introspect_memory",
    "introspect_memory_inject",
]
