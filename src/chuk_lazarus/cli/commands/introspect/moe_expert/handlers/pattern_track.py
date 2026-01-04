"""Handler for 'pattern-track' action - track a specific pattern across all layers."""

from __future__ import annotations

import asyncio
from argparse import Namespace
from collections import Counter
from dataclasses import dataclass, field

from ......introspection.datasets import get_layer_sweep_tests
from ......introspection.moe import ExpertRouter
from ..formatters import format_header


@dataclass
class PatternLayerProfile:
    """Profile of a pattern at a specific layer."""

    layer_idx: int
    top_expert: int
    top_expert_count: int
    total_pattern_activations: int
    example_prompts: list[str] = field(default_factory=list)
    expert_distribution: Counter[int] = field(default_factory=Counter)

    @property
    def confidence(self) -> float:
        """How confident is the top expert for this pattern (0-100%)."""
        if self.total_pattern_activations == 0:
            return 0.0
        return (self.top_expert_count / self.total_pattern_activations) * 100

    @property
    def specialization_level(self) -> str:
        """Categorize: SPECIALIST (>80%), TRANSITION (50-80%), GENERALIST (<50%)."""
        conf = self.confidence
        if conf >= 80:
            return "SPECIALIST"
        elif conf >= 50:
            return "TRANSITION"
        else:
            return "GENERALIST"

    def confidence_bar(self, width: int = 20) -> str:
        """Visual bar for confidence level."""
        filled = int((self.confidence / 100) * width)
        empty = width - filled
        return "█" * filled + "░" * empty


def classify_token_context(token: str, position: int, prompt: str) -> str:
    """Classify the context type for a token based on position and preceding token."""
    if position == 0:
        return "SEQUENCE_START"

    tokens = prompt.split()
    if position > 0 and position < len(tokens):
        preceding = tokens[position - 1] if position <= len(tokens) else ""
    else:
        preceding = ""

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


def handle_pattern_track(args: Namespace) -> None:
    """Handle the 'pattern-track' action - track a pattern across layers.

    Args:
        args: Parsed CLI arguments. Required:
            - model: Model ID
            - pattern: Pattern to track (e.g., SEQUENCE_START, AFTER_NUMBER)
            Optional:
            - layers: Comma-separated layer indices or "all"
            - examples: Number of example prompts to show (default 4)

    Example:
        lazarus introspect moe-expert pattern-track -m openai/gpt-oss-20b --pattern SEQUENCE_START
        lazarus introspect moe-expert pattern-track -m openai/gpt-oss-20b --pattern AFTER_NUMBER --layers all
    """
    asyncio.run(_async_pattern_track(args))


async def _async_pattern_track(args: Namespace) -> None:
    """Async implementation of pattern-track handler."""
    model_id = args.model
    pattern = getattr(args, "pattern", "SEQUENCE_START").upper()
    layers_arg = getattr(args, "layers", None)
    num_examples = getattr(args, "examples", 4)

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
            target_layers = list(info.moe_layers)

        print(format_header(f"PATTERN TRACKING: {pattern}"))
        print(f"Model: {model_id}")
        print(f"Tracking pattern: {pattern}")
        print(f"Across layers: {len(target_layers)} MoE layers")
        print()

        # Collect prompts
        if test_data:
            all_prompts = test_data.get_all_prompts()
            prompts = [(subcat, prompt) for _, subcat, prompt in all_prompts]
        else:
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
        print(f"Analyzing {total_prompts} prompts per layer...")
        print()

        # Track pattern across all layers
        layer_profiles: list[PatternLayerProfile] = []

        # Track which expert handles this pattern at each layer
        expert_history: dict[int, list[tuple[int, float]]] = {}  # expert -> [(layer, confidence)]

        for layer_num, layer_idx in enumerate(target_layers, 1):
            expert_counts: Counter[int] = Counter()
            example_prompts: list[str] = []
            total_activations = 0

            print(f"[Layer {layer_num}/{len(target_layers)}] L{layer_idx}: ", end="", flush=True)
            processed = 0

            for i, (subcat, prompt) in enumerate(prompts):
                try:
                    weights = await router.capture_router_weights(prompt, layers=[layer_idx])
                    if not weights:
                        continue

                    for layer_weights in weights:
                        for pos in layer_weights.positions:
                            context_type = classify_token_context(
                                pos.token or "", pos.position_idx, prompt
                            )

                            if context_type == pattern:
                                for exp_idx in pos.expert_indices:
                                    expert_counts[exp_idx] += 1
                                    total_activations += 1

                                # Collect example prompts
                                if len(example_prompts) < num_examples and prompt not in example_prompts:
                                    example_prompts.append(prompt)

                    processed += 1

                except Exception:
                    continue

                if (i + 1) % 50 == 0:
                    print(".", end="", flush=True)

            # Build profile for this layer
            if expert_counts:
                top_expert, top_count = expert_counts.most_common(1)[0]
            else:
                top_expert, top_count = -1, 0

            profile = PatternLayerProfile(
                layer_idx=layer_idx,
                top_expert=top_expert,
                top_expert_count=top_count,
                total_pattern_activations=total_activations,
                example_prompts=example_prompts,
                expert_distribution=expert_counts,
            )
            layer_profiles.append(profile)

            # Track expert history
            if top_expert >= 0:
                if top_expert not in expert_history:
                    expert_history[top_expert] = []
                expert_history[top_expert].append((layer_idx, profile.confidence))

            print(f" E{top_expert} @ {profile.confidence:.0f}% ({profile.specialization_level})")

        # Summary visualization
        print()
        print(format_header(f"{pattern} HANDLER BY LAYER"))
        print()
        print(f"{'Layer':<8} {'Expert':<8} {'Confidence':<12} {'Specialization':<14} {'Visual'}")
        print("-" * 70)

        peak_layer = None
        peak_confidence = 0

        for profile in layer_profiles:
            if profile.confidence > peak_confidence:
                peak_confidence = profile.confidence
                peak_layer = profile

            marker = " ← peak" if profile.confidence == peak_confidence and profile == peak_layer else ""
            print(
                f"L{profile.layer_idx:<7} "
                f"E{profile.top_expert:<7} "
                f"{profile.confidence:>5.0f}%       "
                f"{profile.specialization_level:<14} "
                f"{profile.confidence_bar()}{marker}"
            )

        # Categorize layers
        specialists = [p for p in layer_profiles if p.specialization_level == "SPECIALIST"]
        transitions = [p for p in layer_profiles if p.specialization_level == "TRANSITION"]
        generalists = [p for p in layer_profiles if p.specialization_level == "GENERALIST"]

        print()
        print(format_header("LAYER ZONES"))

        if specialists:
            layers = [p.layer_idx for p in specialists]
            print(f"Specialists (>80%):  L{min(layers)}-L{max(layers)} ({len(specialists)} layers)")
        else:
            print("Specialists (>80%):  None")

        if transitions:
            layers = [p.layer_idx for p in transitions]
            print(f"Transition (50-80%): L{min(layers)}-L{max(layers)} ({len(transitions)} layers)")
        else:
            print("Transition (50-80%): None")

        if generalists:
            layers = [p.layer_idx for p in generalists]
            print(f"Generalists (<50%):  L{min(layers)}-L{max(layers)} ({len(generalists)} layers)")
        else:
            print("Generalists (<50%):  None")

        # Expert ownership changes
        print()
        print(format_header("EXPERT HANDOFFS"))

        current_expert = None
        for profile in layer_profiles:
            if profile.top_expert != current_expert and profile.top_expert >= 0:
                if current_expert is not None:
                    print(f"L{profile.layer_idx}: E{current_expert} → E{profile.top_expert} (handoff)")
                current_expert = profile.top_expert

        # Example tokens
        print()
        print(format_header(f"EXAMPLE TOKENS FOR {pattern}"))

        # Get examples from peak layer
        if peak_layer and peak_layer.example_prompts:
            print(f"From L{peak_layer.layer_idx} (E{peak_layer.top_expert}):")
            for prompt in peak_layer.example_prompts[:num_examples]:
                print(f'  "{prompt}"')

        # Video-ready summary
        print()
        print(format_header("VIDEO SUMMARY"))
        print()
        print(f'"Watch {pattern} move through the model"')
        print()

        # Show 5 evenly spaced layers
        n = len(layer_profiles)
        indices = [0, n // 4, n // 2, 3 * n // 4, n - 1] if n >= 5 else list(range(n))

        for idx in indices:
            profile = layer_profiles[idx]
            bar = profile.confidence_bar()
            print(f"Layer {profile.layer_idx:>2}:  E{profile.top_expert:<2} {bar} {profile.confidence:.0f}%")

        if generalists and specialists:
            print()
            print(f'"The specialist dies. By layer {generalists[0].layer_idx}, nobody owns it."')

        print()
        print("=" * 70)
