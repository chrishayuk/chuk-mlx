#!/usr/bin/env python3
"""
Tool Calling Dynamics Experiment

Main orchestration script for analyzing how GPT-OSS represents
and generates tool calls.

Usage:
    python experiment.py                    # Run all analyses
    python experiment.py --analysis intent  # Run specific analysis
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load experiment configuration."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_tool_intent_analysis(config: dict) -> dict[str, Any]:
    """
    Analyze tool intent detection across layers.

    Trains linear probes to detect "will call tool" vs "will answer directly"
    at each layer, identifying the earliest layer with reliable detection.
    """
    logger.info("Running tool intent analysis...")

    from probes.tool_intent_probe import ToolIntentProbe

    probe = ToolIntentProbe(config)
    results = probe.run()

    # Save results
    output_path = Path(config["output"]["results_dir"]) / "tool_intent_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Tool intent results saved to {output_path}")
    return results


def run_tool_selection_analysis(config: dict) -> dict[str, Any]:
    """
    Analyze tool selection mechanism.

    Trains probes to classify which tool will be called,
    and identifies expert activation patterns for each tool type.
    """
    logger.info("Running tool selection analysis...")

    from probes.tool_selection_probe import ToolSelectionProbe

    probe = ToolSelectionProbe(config)
    results = probe.run()

    # Save results
    output_path = Path(config["output"]["results_dir"]) / "tool_selection_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Tool selection results saved to {output_path}")
    return results


def run_expert_patterns_analysis(config: dict) -> dict[str, Any]:
    """
    Analyze expert activation patterns for tool-related tokens.

    Identifies which experts specialize in:
    - Function call syntax: ( ) , :
    - JSON syntax: { } [ ] "
    - Tool markers: <tool> function call
    - Parameter tokens: name arguments value
    """
    logger.info("Running expert patterns analysis...")

    from analysis.expert_patterns import ExpertPatternAnalyzer

    analyzer = ExpertPatternAnalyzer(config)
    results = analyzer.run()

    # Save results
    output_path = Path(config["output"]["results_dir"]) / "expert_patterns_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Expert patterns results saved to {output_path}")
    return results


def run_circuit_analysis(config: dict) -> dict[str, Any]:
    """
    Analyze cross-layer expert circuits for tool calling.

    Identifies stable expert combinations across layers that
    form functional units for tool-related computation.
    """
    logger.info("Running circuit analysis...")

    from analysis.circuit_analysis import CircuitAnalyzer

    analyzer = CircuitAnalyzer(config)
    results = analyzer.run()

    # Save results
    output_path = Path(config["output"]["results_dir"]) / "circuit_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Circuit results saved to {output_path}")
    return results


def run_vocab_alignment_analysis(config: dict) -> dict[str, Any]:
    """
    Test vocab alignment for tool names.

    Projects hidden states to vocabulary space to check if
    tool names (calculator, search, etc.) are represented.
    """
    logger.info("Running vocab alignment analysis...")

    from analysis.vocab_alignment import VocabAlignmentAnalyzer

    analyzer = VocabAlignmentAnalyzer(config)
    results = analyzer.run()

    # Save results
    output_path = Path(config["output"]["results_dir"]) / "vocab_alignment_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Vocab alignment results saved to {output_path}")
    return results


def run_generation_dynamics_analysis(config: dict) -> dict[str, Any]:
    """
    Analyze expert routing during tool call generation.

    Tracks how expert activations change through:
    - Intent phase (deciding to call tool)
    - Selection phase (choosing which tool)
    - Format phase (generating JSON/function syntax)
    - Argument phase (filling in parameters)
    """
    logger.info("Running generation dynamics analysis...")

    from analysis.generation_dynamics import GenerationDynamicsAnalyzer

    analyzer = GenerationDynamicsAnalyzer(config)
    results = analyzer.run()

    # Save results
    output_path = Path(config["output"]["results_dir"]) / "generation_dynamics_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Generation dynamics results saved to {output_path}")
    return results


def run_all_analyses(config: dict) -> dict[str, Any]:
    """Run all analyses and compile results."""
    results = {}

    analyses = [
        ("tool_intent", run_tool_intent_analysis),
        ("tool_selection", run_tool_selection_analysis),
        ("expert_patterns", run_expert_patterns_analysis),
        ("circuits", run_circuit_analysis),
        ("vocab_alignment", run_vocab_alignment_analysis),
        ("generation_dynamics", run_generation_dynamics_analysis),
    ]

    for name, func in analyses:
        if name in config.get("analyses", [name]):
            try:
                results[name] = func(config)
            except Exception as e:
                logger.error(f"Analysis {name} failed: {e}")
                results[name] = {"error": str(e)}

    # Save combined results
    output_path = Path(config["output"]["results_dir"]) / "all_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def print_summary(results: dict[str, Any]):
    """Print a summary of all results."""
    print("\n" + "=" * 60)
    print("TOOL CALLING DYNAMICS - RESULTS SUMMARY")
    print("=" * 60)

    if "tool_intent" in results and "error" not in results["tool_intent"]:
        intent = results["tool_intent"]
        print(f"\n1. Tool Intent Detection:")
        print(f"   Best layer: L{intent.get('best_layer', '?')}")
        print(f"   Accuracy: {intent.get('best_accuracy', 0):.1%}")
        print(f"   First reliable layer (>95%): L{intent.get('first_reliable_layer', '?')}")

    if "tool_selection" in results and "error" not in results["tool_selection"]:
        selection = results["tool_selection"]
        print(f"\n2. Tool Selection:")
        print(f"   Best layer: L{selection.get('best_layer', '?')}")
        print(f"   Accuracy: {selection.get('best_accuracy', 0):.1%}")

    if "expert_patterns" in results and "error" not in results["expert_patterns"]:
        patterns = results["expert_patterns"]
        print(f"\n3. Expert Patterns:")
        print(f"   Tool syntax experts: {patterns.get('num_syntax_experts', '?')}")
        print(f"   Specialization score: {patterns.get('avg_specialization', 0):.3f}")

    if "circuits" in results and "error" not in results["circuits"]:
        circuits = results["circuits"]
        print(f"\n4. Cross-Layer Circuits:")
        print(f"   Circuits found: {circuits.get('num_circuits', '?')}")
        print(f"   Avg consistency: {circuits.get('avg_consistency', 0):.1%}")

    if "vocab_alignment" in results and "error" not in results["vocab_alignment"]:
        vocab = results["vocab_alignment"]
        print(f"\n5. Vocab Alignment:")
        print(f"   Aligned tools: {vocab.get('num_aligned', '?')}/{vocab.get('num_tools', '?')}")
        print(f"   Best alignment: {vocab.get('best_tool', '?')} at rank {vocab.get('best_rank', '?')}")

    if "generation_dynamics" in results and "error" not in results["generation_dynamics"]:
        dynamics = results["generation_dynamics"]
        print(f"\n6. Generation Dynamics:")
        print(f"   Phase transitions detected: {dynamics.get('phases_detected', '?')}")
        print(f"   Expert consistency: {dynamics.get('expert_consistency', 0):.1%}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Tool Calling Dynamics Experiment"
    )
    parser.add_argument(
        "--analysis",
        choices=["intent", "selection", "experts", "circuits", "vocab", "dynamics", "all"],
        default="all",
        help="Which analysis to run"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override model from config"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without running"
    )
    args = parser.parse_args()

    config = load_config()

    if args.model:
        config["model"]["primary"] = args.model

    if args.dry_run:
        print(yaml.dump(config, default_flow_style=False))
        return

    # Map CLI args to analysis functions
    analysis_map = {
        "intent": run_tool_intent_analysis,
        "selection": run_tool_selection_analysis,
        "experts": run_expert_patterns_analysis,
        "circuits": run_circuit_analysis,
        "vocab": run_vocab_alignment_analysis,
        "dynamics": run_generation_dynamics_analysis,
        "all": run_all_analyses,
    }

    func = analysis_map[args.analysis]
    results = func(config)

    if args.analysis == "all":
        print_summary(results)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
