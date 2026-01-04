"""Handler for 'layer-sweep' action - comprehensive sweep analysis across all layers."""

from __future__ import annotations

import asyncio
from argparse import Namespace
from collections import Counter
from dataclasses import dataclass, field

from ......introspection.datasets import get_layer_sweep_tests
from ......introspection.moe import ExpertRouter
from ..formatters import format_header


@dataclass
class TokenContext:
    """Context information for a token."""

    token: str
    position: int
    preceding_type: str  # "number", "word", "operator", "punct", "start"
    subcategory: str  # from test dataset


@dataclass
class LayerPatternProfile:
    """Pattern profile for a single layer."""

    layer_idx: int
    layer_fraction: float  # position in model (0.0 = first, 1.0 = last)
    total_activations: int
    expert_counts: Counter[int] = field(default_factory=Counter)
    pattern_experts: dict[str, Counter[int]] = field(default_factory=dict)
    workhorse_threshold: float = 0.05  # >5% of activations

    @property
    def workhorses(self) -> list[int]:
        """Experts with >threshold of activations."""
        if self.total_activations == 0:
            return []
        return [
            exp
            for exp, count in self.expert_counts.items()
            if count / self.total_activations > self.workhorse_threshold
        ]

    @property
    def spectators(self) -> list[int]:
        """Experts with 0 activations."""
        all_experts = set(range(32))  # Assume 32 experts max, will be refined
        active = set(self.expert_counts.keys())
        return sorted(all_experts - active)

    def top_patterns(self, n: int = 3) -> list[tuple[str, int]]:
        """Get top N patterns by expert concentration."""
        pattern_scores: list[tuple[str, int]] = []
        for pattern, experts in self.pattern_experts.items():
            if experts:
                # Score = how concentrated the pattern is (top expert count)
                top_count = experts.most_common(1)[0][1] if experts else 0
                pattern_scores.append((pattern, top_count))
        pattern_scores.sort(key=lambda x: x[1], reverse=True)
        return pattern_scores[:n]


def classify_token_context(token: str, position: int, prompt: str) -> str:
    """Classify the context type for a token based on position and preceding token."""
    if position == 0:
        return "SEQUENCE_START"

    # Get preceding token (simplified - just look at what's before this token)
    tokens = prompt.split()
    if position > 0 and position < len(tokens):
        preceding = tokens[position - 1] if position <= len(tokens) else ""
    else:
        # For single tokens or edge cases
        preceding = ""

    # Classify preceding token type
    if preceding.replace(".", "").replace("-", "").isdigit():
        return "AFTER_NUMBER"
    elif preceding in "+-*/=<>!&|":
        return "AFTER_OPERATOR"
    elif preceding in ".,;:!?()[]{}\"'":
        return "AFTER_PUNCT"
    elif preceding.lower() in ("def", "class", "import", "return", "if", "for", "while"):
        return "AFTER_CODE_KW"
    elif preceding.isalpha():
        return "AFTER_WORD"
    else:
        return "MIXED"


def handle_layer_sweep(args: Namespace) -> None:
    """Handle the 'layer-sweep' action - comprehensive sweep across all MoE layers.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID
            Optional:
            - layers: Comma-separated layer indices or "all"
            - verbose: Show detailed per-expert breakdown

    Example:
        lazarus introspect moe-expert layer-sweep -m openai/gpt-oss-20b
        lazarus introspect moe-expert layer-sweep -m openai/gpt-oss-20b --verbose
    """
    asyncio.run(_async_layer_sweep(args))


async def _async_layer_sweep(args: Namespace) -> None:
    """Async implementation of enhanced layer_sweep handler."""
    model_id = args.model
    verbose = getattr(args, "verbose", False)
    layers_arg = getattr(args, "layers", None)

    print(f"Loading model: {model_id}")

    # Load test dataset
    try:
        test_data = get_layer_sweep_tests()
    except Exception as e:
        print(f"Warning: Could not load layer sweep tests: {e}")
        print("Using fallback prompts...")
        test_data = None

    async with await ExpertRouter.from_pretrained(model_id) as router:
        info = router.info
        num_experts = info.num_experts

        # Determine which layers to analyze
        if layers_arg:
            if layers_arg == "all":
                target_layers = list(info.moe_layers)
            else:
                target_layers = [int(x) for x in layers_arg.split(",")]
        else:
            # Default: sample key layers (early, early-mid, mid, late-mid, late)
            n_layers = len(info.moe_layers)
            if n_layers <= 5:
                target_layers = list(info.moe_layers)
            else:
                indices = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
                target_layers = [info.moe_layers[i] for i in indices]

        print(format_header("LAYER SWEEP ANALYSIS"))
        print(f"Model: {model_id}")
        print(f"Architecture: {info.architecture.value}")
        print(f"Total MoE layers: {len(info.moe_layers)}")
        print(f"Experts per layer: {num_experts}")
        print(f"Analyzing layers: {target_layers}")
        print()

        # Collect prompts
        if test_data:
            all_prompts = test_data.get_all_prompts()
            prompts = [(subcat, prompt) for _, subcat, prompt in all_prompts]
        else:
            # Fallback prompts
            prompts = [
                ("position_0", "the"),
                ("position_0", "127"),
                ("num_after_num", "111 127"),
                ("word_after_word", "the cat"),
                ("num_after_word", "abc 127"),
                ("math_expr", "127 * 89 ="),
                ("code", "def foo():"),
            ]

        total_prompts = len(prompts)
        batch_size = 50  # Report progress every 50 prompts
        print(f"Testing with {total_prompts} prompts...")
        print()

        # Analyze each layer
        layer_profiles: list[LayerPatternProfile] = []

        for layer_num, layer_idx in enumerate(target_layers, 1):
            layer_fraction = layer_idx / max(info.moe_layers) if info.moe_layers else 0

            profile = LayerPatternProfile(
                layer_idx=layer_idx,
                layer_fraction=layer_fraction,
                total_activations=0,
            )

            print(f"[Layer {layer_num}/{len(target_layers)}] L{layer_idx}: ", end="", flush=True)
            processed = 0
            errors = 0

            for i, (subcat, prompt) in enumerate(prompts):
                try:
                    weights = await router.capture_router_weights(prompt, layers=[layer_idx])
                    if not weights:
                        continue

                    for layer_weights in weights:
                        for pos in layer_weights.positions:
                            # Classify context
                            context_type = classify_token_context(
                                pos.token or "", pos.position_idx, prompt
                            )

                            # Track expert activations
                            for exp_idx in pos.expert_indices:
                                profile.expert_counts[exp_idx] += 1
                                profile.total_activations += 1

                                # Track by pattern
                                if context_type not in profile.pattern_experts:
                                    profile.pattern_experts[context_type] = Counter()
                                profile.pattern_experts[context_type][exp_idx] += 1

                                # Also track by subcategory
                                if subcat not in profile.pattern_experts:
                                    profile.pattern_experts[subcat] = Counter()
                                profile.pattern_experts[subcat][exp_idx] += 1

                    processed += 1

                except Exception:
                    errors += 1
                    continue

                # Progress indicator every batch_size prompts
                if (i + 1) % batch_size == 0:
                    print(".", end="", flush=True)

            # Summary for this layer
            print(f" {processed}/{total_prompts} OK", end="")
            if errors > 0:
                print(f" ({errors} errors)", end="")
            print(f" | {profile.total_activations} activations, {len(profile.workhorses)} workhorses")

            layer_profiles.append(profile)

        # Output summary table
        print(
            f"{'Layer':<8} {'Fraction':<10} {'Workhorses':<12} {'Spectators':<12} {'Top Patterns'}"
        )
        print("-" * 80)

        for profile in layer_profiles:
            # Calculate spectators based on actual num_experts
            active_experts = set(profile.expert_counts.keys())
            spectator_count = num_experts - len(active_experts)

            top_patterns = profile.top_patterns(3)
            patterns_str = ", ".join(p[0] for p in top_patterns) if top_patterns else "NONE"

            print(
                f"L{profile.layer_idx:<6} "
                f"{profile.layer_fraction:<10.2f} "
                f"{len(profile.workhorses):<12} "
                f"{spectator_count:<12} "
                f"{patterns_str}"
            )

        # Pattern Evolution section
        print()
        print(format_header("PATTERN EVOLUTION"))

        # Collect all patterns across layers
        all_patterns: set[str] = set()
        for profile in layer_profiles:
            all_patterns.update(profile.pattern_experts.keys())

        # Filter to context-based patterns (uppercase)
        context_patterns = sorted([p for p in all_patterns if p.isupper()])

        if context_patterns:
            print(f"{'Pattern':<20} {'Layers Active':<20} {'Peak Layer':<12} {'Peak Expert'}")
            print("-" * 70)

            for pattern in context_patterns:
                active_layers = []
                peak_layer = -1
                peak_count = 0
                peak_expert = -1

                for profile in layer_profiles:
                    if pattern in profile.pattern_experts:
                        experts = profile.pattern_experts[pattern]
                        if experts:
                            active_layers.append(profile.layer_idx)
                            top_exp, top_count = experts.most_common(1)[0]
                            if top_count > peak_count:
                                peak_count = top_count
                                peak_layer = profile.layer_idx
                                peak_expert = top_exp

                if active_layers:
                    layers_str = (
                        f"L{min(active_layers)}-L{max(active_layers)}"
                        if len(active_layers) > 1
                        else f"L{active_layers[0]}"
                    )
                    print(f"{pattern:<20} {layers_str:<20} L{peak_layer:<11} E{peak_expert}")

        # Expert Lifecycle section (if verbose)
        if verbose:
            print()
            print(format_header("EXPERT LIFECYCLE"))

            # Track how top experts change roles across layers
            expert_roles: dict[int, list[tuple[int, str, int]]] = {}

            for profile in layer_profiles:
                for pattern, experts in profile.pattern_experts.items():
                    if not pattern.isupper():  # Only context patterns
                        continue
                    for exp, count in experts.most_common(3):
                        if exp not in expert_roles:
                            expert_roles[exp] = []
                        expert_roles[exp].append((profile.layer_idx, pattern, count))

            # Show top experts that appear across multiple layers
            for exp in sorted(expert_roles.keys())[:10]:
                roles = expert_roles[exp]
                if len(roles) >= 2:
                    role_str = " -> ".join(
                        f"L{l}:{p}" for l, p, _ in sorted(roles, key=lambda x: x[0])
                    )
                    print(f"E{exp:2d}: {role_str}")

        # Summary stats
        print()
        print(format_header("SUMMARY"))

        total_workhorses = sum(len(p.workhorses) for p in layer_profiles)
        avg_workhorses = total_workhorses / len(layer_profiles) if layer_profiles else 0

        total_spectators = sum(num_experts - len(p.expert_counts) for p in layer_profiles)
        avg_spectators = total_spectators / len(layer_profiles) if layer_profiles else 0

        print(f"Average workhorses per layer: {avg_workhorses:.1f}")
        print(f"Average spectators per layer: {avg_spectators:.1f}")
        print(f"Total patterns discovered: {len(context_patterns)}")

        # Show expected vs actual patterns by layer position
        if test_data:
            print()
            print("Layer Position Expectations:")
            for profile in layer_profiles:
                expectation = test_data.get_layer_expectation(profile.layer_fraction)
                if expectation:
                    actual = [p[0] for p in profile.top_patterns(3)]
                    expected = expectation.expected_patterns[:3]
                    match_count = len(set(actual) & set(expected))
                    print(
                        f"  L{profile.layer_idx} ({expectation.description}): "
                        f"{match_count}/{len(expected)} expected patterns found"
                    )

        print("=" * 80)
