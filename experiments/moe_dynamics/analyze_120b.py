#!/usr/bin/env python3
"""
GPT-OSS-120B MoE Dynamics Analysis

Runs the same experiments performed on GPT-OSS-20B to validate compression findings.

Key experiments:
1. Cold Expert Analysis - Identify unused experts
2. Expert Interference - Verify k=4 requirement
3. Expert Circuits - Cross-layer routing patterns
4. Expert Merging - Check orthogonality (native MoE confirmation)

Expected outcomes (based on 20B):
- Cold expert rate: ~87% at 1% threshold
- k=4 required (k=1 breaks output)
- 0% mergeable pairs (orthogonal experts)
- ~90% compression potential

Usage:
    # Quick analysis (cold experts only)
    python experiments/moe_dynamics/analyze_120b.py --quick

    # Full analysis
    python experiments/moe_dynamics/analyze_120b.py --full

    # Using lazarus CLI
    lazarus introspect moe-expert cold-experts -m openai/gpt-oss-120b
    lazarus introspect moe-expert expert-interference -m openai/gpt-oss-120b
    lazarus introspect moe-expert expert-circuits -m openai/gpt-oss-120b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_ID = "openai/gpt-oss-120b"
MODEL_20B = "openai/gpt-oss-20b"

# Architecture comparison
ARCHITECTURE = {
    "20b": {"layers": 24, "experts": 32, "total": 768},
    "120b": {"layers": 36, "experts": 128, "total": 4608},
}

# Results directory
RESULTS_DIR = Path(__file__).parent / "results" / "120b"


def run_lazarus_command(command: list[str], timeout: int = 600) -> dict[str, Any]:
    """Run a lazarus CLI command and capture output."""
    full_command = ["lazarus"] + command

    logger.info(f"Running: {' '.join(full_command)}")

    try:
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return {
            "command": full_command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "command": full_command,
            "returncode": -1,
            "stdout": "",
            "stderr": "Command timed out",
            "success": False,
        }
    except Exception as e:
        return {
            "command": full_command,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False,
        }


def analyze_cold_experts() -> dict[str, Any]:
    """
    Analyze cold (rarely-activated) experts in GPT-OSS-120B.

    Expected: ~87% cold at 1% threshold (matching 20B)
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Cold Expert Analysis")
    logger.info("=" * 60)

    result = run_lazarus_command([
        "introspect", "moe-expert", "cold-experts",
        "-m", MODEL_ID,
    ])

    if result["success"]:
        logger.info("Cold expert analysis complete")
        logger.info(result["stdout"][:2000] if result["stdout"] else "No output")
    else:
        logger.error(f"Cold expert analysis failed: {result['stderr']}")

    return {
        "experiment": "cold_experts",
        "model": MODEL_ID,
        "result": result,
        "expected": "~87% cold at 1% threshold",
    }


def analyze_expert_interference() -> dict[str, Any]:
    """
    Test if k=4 routing is essential for GPT-OSS-120B.

    Expected: k=4 required, k=1 breaks output (matching 20B)
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Expert Interference (k-value test)")
    logger.info("=" * 60)

    result = run_lazarus_command([
        "introspect", "moe-expert", "expert-interference",
        "-m", MODEL_ID,
    ])

    if result["success"]:
        logger.info("Expert interference analysis complete")
        logger.info(result["stdout"][:2000] if result["stdout"] else "No output")
    else:
        logger.error(f"Expert interference analysis failed: {result['stderr']}")

    return {
        "experiment": "expert_interference",
        "model": MODEL_ID,
        "result": result,
        "expected": "k=4 required, k=1 breaks output",
    }


def analyze_expert_circuits() -> dict[str, Any]:
    """
    Analyze cross-layer expert circuits in GPT-OSS-120B.

    Expected: ~87.5% layer-pair consistency, 15+ pipelines
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: Expert Circuits")
    logger.info("=" * 60)

    result = run_lazarus_command([
        "introspect", "moe-expert", "expert-circuits",
        "-m", MODEL_ID,
    ])

    if result["success"]:
        logger.info("Expert circuits analysis complete")
        logger.info(result["stdout"][:2000] if result["stdout"] else "No output")
    else:
        logger.error(f"Expert circuits analysis failed: {result['stderr']}")

    return {
        "experiment": "expert_circuits",
        "model": MODEL_ID,
        "result": result,
        "expected": "~87.5% layer-pair consistency",
    }


def analyze_expert_merging() -> dict[str, Any]:
    """
    Check expert orthogonality (native MoE confirmation).

    Expected: 0% mergeable pairs at 0.8 threshold
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 4: Expert Merging (Orthogonality)")
    logger.info("=" * 60)

    result = run_lazarus_command([
        "introspect", "moe-expert", "expert-merging",
        "-m", MODEL_ID,
    ])

    if result["success"]:
        logger.info("Expert merging analysis complete")
        logger.info(result["stdout"][:2000] if result["stdout"] else "No output")
    else:
        logger.error(f"Expert merging analysis failed: {result['stderr']}")

    return {
        "experiment": "expert_merging",
        "model": MODEL_ID,
        "result": result,
        "expected": "0% mergeable (native MoE, orthogonal experts)",
    }


def analyze_generation_dynamics() -> dict[str, Any]:
    """
    Analyze routing dynamics during generation.

    Expected: ~45.9% consistency, middle layers most stable
    """
    logger.info("=" * 60)
    logger.info("EXPERIMENT 5: Generation Dynamics")
    logger.info("=" * 60)

    result = run_lazarus_command([
        "introspect", "moe-expert", "generation-dynamics",
        "-m", MODEL_ID,
    ], timeout=900)  # Longer timeout for generation

    if result["success"]:
        logger.info("Generation dynamics analysis complete")
        logger.info(result["stdout"][:2000] if result["stdout"] else "No output")
    else:
        logger.error(f"Generation dynamics analysis failed: {result['stderr']}")

    return {
        "experiment": "generation_dynamics",
        "model": MODEL_ID,
        "result": result,
        "expected": "~45.9% consistency, middle layers most stable",
    }


def run_quick_analysis() -> dict[str, Any]:
    """Run quick analysis (cold experts only)."""
    results = {
        "model": MODEL_ID,
        "timestamp": datetime.now().isoformat(),
        "mode": "quick",
        "experiments": {},
    }

    # Cold experts is the most informative quick test
    results["experiments"]["cold_experts"] = analyze_cold_experts()

    return results


def run_full_analysis() -> dict[str, Any]:
    """Run full analysis suite."""
    results = {
        "model": MODEL_ID,
        "timestamp": datetime.now().isoformat(),
        "mode": "full",
        "architecture": ARCHITECTURE["120b"],
        "experiments": {},
    }

    # Run all experiments
    results["experiments"]["cold_experts"] = analyze_cold_experts()
    results["experiments"]["expert_interference"] = analyze_expert_interference()
    results["experiments"]["expert_circuits"] = analyze_expert_circuits()
    results["experiments"]["expert_merging"] = analyze_expert_merging()
    results["experiments"]["generation_dynamics"] = analyze_generation_dynamics()

    return results


def run_comparison_analysis() -> dict[str, Any]:
    """Run both 20B and 120B for direct comparison."""
    logger.info("Running comparison analysis: 20B vs 120B")

    results = {
        "timestamp": datetime.now().isoformat(),
        "mode": "comparison",
        "models": {},
    }

    # Run cold expert analysis on both
    for model_id, arch_key in [(MODEL_20B, "20b"), (MODEL_ID, "120b")]:
        logger.info(f"\nAnalyzing {model_id}...")

        model_results = {
            "model": model_id,
            "architecture": ARCHITECTURE[arch_key],
        }

        # Cold experts
        result = run_lazarus_command([
            "introspect", "moe-expert", "cold-experts",
            "-m", model_id,
        ])
        model_results["cold_experts"] = result

        results["models"][arch_key] = model_results

    return results


def save_results(results: dict[str, Any], filename: str = None) -> Path:
    """Save results to JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_120b_{results.get('mode', 'unknown')}_{timestamp}.json"

    output_path = RESULTS_DIR / filename

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {output_path}")
    return output_path


def print_summary(results: dict[str, Any]) -> None:
    """Print summary of analysis results."""
    print()
    print("=" * 80)
    print("GPT-OSS-120B MoE DYNAMICS ANALYSIS SUMMARY")
    print("=" * 80)
    print()
    print(f"Model: {results.get('model', MODEL_ID)}")
    print(f"Timestamp: {results.get('timestamp', 'unknown')}")
    print(f"Mode: {results.get('mode', 'unknown')}")
    print()

    if "architecture" in results:
        arch = results["architecture"]
        print(f"Architecture: {arch['layers']} layers × {arch['experts']} experts = {arch['total']:,} total")
        print()

    if "experiments" in results:
        print("Experiments:")
        print("-" * 60)
        for name, data in results["experiments"].items():
            success = data.get("result", {}).get("success", False)
            status = "✓" if success else "✗"
            expected = data.get("expected", "N/A")
            print(f"  {status} {name}: expected {expected}")
        print()

    # Predictions based on 20B findings
    print("Predictions (based on GPT-OSS-20B findings):")
    print("-" * 60)
    print("  Cold experts:     ~87% at 1% threshold")
    print("  k-value:          k=4 essential (k=1 breaks output)")
    print("  Expert merging:   0% (orthogonal, native MoE)")
    print("  Layer consistency: ~87.5% layer-pair")
    print()
    print("  Expected compression: 85-92%")
    print("  Expected speedup:     4-8x")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="GPT-OSS-120B MoE Dynamics Analysis"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick analysis (cold experts only)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full analysis suite",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare 20B and 120B",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename",
    )

    args = parser.parse_args()

    # Default to quick if nothing specified
    if not any([args.quick, args.full, args.compare]):
        args.quick = True

    # Run analysis
    if args.compare:
        results = run_comparison_analysis()
    elif args.full:
        results = run_full_analysis()
    else:
        results = run_quick_analysis()

    # Save and print results
    output_path = save_results(results, args.output)
    print_summary(results)

    print(f"Full results saved to: {output_path}")


if __name__ == "__main__":
    main()
