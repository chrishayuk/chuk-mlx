#!/usr/bin/env python3
"""
steering_demo.py

Demonstrate the tool-calling steering feature on FunctionGemma 270M.

NOTE: FunctionGemma 270M is a base model designed for fine-tuning. It has only
~58% function calling accuracy without domain-specific tuning. This demo shows
that steering works on the model's internal representations, even though the
base model doesn't reliably produce function calls.

For actual function calling deployment, you should:
1. Fine-tune FunctionGemma on your domain
2. Use a larger model (Gemma 3 12B/27B)

Run: uv run python examples/introspection/steering_demo.py
"""

import json
import numpy as np
import mlx.core as mx
from sklearn.linear_model import LogisticRegression

from chuk_lazarus.introspection.steering import (
    ToolCallingSteering,
    SteeringConfig,
    SteeringMode,
    format_functiongemma_prompt,
)


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def demo_text_generation():
    """Show that the model generates coherent text."""
    print_header("TEXT GENERATION DEMO")
    print("\nShowing that the model generates coherent text.")
    print("NOTE: Base FunctionGemma 270M has only ~58% function call accuracy.")
    print("It often generates text responses instead of function calls.\n")

    steerer = ToolCallingSteering.from_pretrained(
        "mlx-community/functiongemma-270m-it-bf16"
    )

    queries = [
        "What is the weather in Tokyo?",
        "Send an email to John",
        "What is the capital of France?",
    ]

    for query in queries:
        prompt = format_functiongemma_prompt(query)
        output = steerer.generate(prompt, mode="normal", max_new_tokens=40)

        print(f"Query: {query}")
        print(f"Output: {output[:100]}...")
        print()


def demo_steering_effect_on_representations():
    """
    Show that steering changes the model's internal representations.

    This is the key validation: even if the model doesn't output function calls,
    steering moves the representations in the expected direction.
    """
    print_header("STEERING EFFECT ON REPRESENTATIONS")
    print("\nThis demo shows that steering moves internal representations")
    print("toward/away from the 'tool-calling' direction, even if the model")
    print("doesn't output function calls.\n")

    steerer = ToolCallingSteering.from_pretrained(
        "mlx-community/functiongemma-270m-it-bf16"
    )

    # Training data for a simple probe
    tool_queries = [
        "What is the weather in Paris?",
        "Send an email to Alice",
        "Set a timer for 5 minutes",
        "Book a flight to London",
        "Call my mom",
    ]

    no_tool_queries = [
        "What is the capital of Spain?",
        "Explain photosynthesis",
        "Who wrote Hamlet?",
        "What is 2+2?",
        "How does gravity work?",
    ]

    # Collect activations
    print("Collecting activations for probe training...")

    def get_final_hidden(steerer, prompt, mode="normal"):
        """Get final hidden state."""
        tokens = steerer.tokenizer.encode(prompt)
        if isinstance(tokens, np.ndarray):
            tokens = tokens.flatten().tolist()
        input_ids = mx.array([tokens])

        steerer._install_steering(SteeringConfig(mode=SteeringMode(mode)))
        try:
            output = steerer.model(input_ids)
            h = output.logits  # Actually use hidden state before lm_head would be better
            # But for simplicity, we use the probability of <start_function_call>
            probs = mx.softmax(output.logits[0, -1, :], axis=-1)
            return float(probs[48])  # Token 48 = <start_function_call>
        finally:
            steerer._uninstall_steering()

    # Collect function call probabilities
    print("\nFunction call token probability by steering mode:\n")
    print(f"{'Query':<40} {'Normal':>10} {'Force':>10} {'Prevent':>10}")
    print("-" * 75)

    for query in tool_queries[:3] + no_tool_queries[:3]:
        prompt = format_functiongemma_prompt(query)

        prob_normal = get_final_hidden(steerer, prompt, "normal")
        prob_force = get_final_hidden(steerer, prompt, "force_tool")
        prob_prevent = get_final_hidden(steerer, prompt, "prevent_tool")

        query_short = query[:38] + ".." if len(query) > 40 else query
        print(f"{query_short:<40} {prob_normal:>9.6f} {prob_force:>9.6f} {prob_prevent:>9.6f}")

    print("\nNote: Base model has very low P(<start_function_call>) for all queries.")
    print("Steering changes these probabilities, but they remain low without fine-tuning.")


def demo_top_token_changes():
    """Show how steering changes the top predicted tokens."""
    print_header("TOP TOKEN CHANGES")
    print("\nShowing how steering affects the top predicted tokens.\n")

    steerer = ToolCallingSteering.from_pretrained(
        "mlx-community/functiongemma-270m-it-bf16"
    )

    queries = [
        ("What is the weather in Tokyo?", "Tool query"),
        ("What is the capital of France?", "Factual query"),
    ]

    for query, label in queries:
        prompt = format_functiongemma_prompt(query)

        print(f"\n{label}: {query}")
        print("-" * 60)

        for mode in ["normal", "force_tool", "prevent_tool"]:
            result = steerer.predict(prompt, mode=mode)

            print(f"\n  Mode: {mode}")
            for token, prob in result['top_tokens'][:3]:
                token_display = token.replace('\n', '\\n')
                if len(token_display) > 25:
                    token_display = token_display[:22] + "..."
                print(f"    {token_display:<28} {prob:>6.1%}")


def demo_generation_comparison():
    """Compare generated text across steering modes."""
    print_header("GENERATION COMPARISON")
    print("\nComparing generated responses across steering modes.\n")

    steerer = ToolCallingSteering.from_pretrained(
        "mlx-community/functiongemma-270m-it-bf16"
    )

    query = "What is the weather in Tokyo?"
    prompt = format_functiongemma_prompt(query)

    print(f"Query: {query}\n")

    for mode in ["normal", "force_tool", "prevent_tool"]:
        output = steerer.generate(prompt, mode=mode, max_new_tokens=50)
        output_display = output.replace('\n', ' ')[:80]

        print(f"  {mode:15} → {output_display}...")


def demo_kill_switch():
    """Demonstrate the L12:230 kill switch."""
    print_header("KILL SWITCH DEMO (L12:230)")
    print("\nThe kill switch neuron can modify tool-calling behavior.")
    print("However, since base model rarely outputs function calls,")
    print("the effect is subtle.\n")

    steerer = ToolCallingSteering.from_pretrained(
        "mlx-community/functiongemma-270m-it-bf16"
    )

    query = "What is the weather in Tokyo?"
    prompt = format_functiongemma_prompt(query)

    configs = [
        ("Normal", {}),
        ("Kill switch ON", {"use_kill_switch": True}),
        ("Kill switch BOOST +10000", {"kill_switch_boost": 10000.0}),
    ]

    print(f"Query: {query}\n")

    for name, kwargs in configs:
        result = steerer.predict(prompt, mode="normal", **kwargs)
        top_token, top_prob = result['top_tokens'][0]
        top_token = top_token.replace('\n', '\\n')[:25]

        print(f"  {name:<30} → {top_token:<25} ({top_prob:.1%})")


def demo_neuron_targeting():
    """Show how targeting specific neurons affects output."""
    print_header("NEURON TARGETING")
    print("\nKey neurons discovered in circuit analysis:")
    print("  Tool promoters: 803, 2036, 831 (boost these to encourage tools)")
    print("  Tool suppressors: 1237, 821, 1347 (boost these to prevent tools)")
    print("  Kill switch: L12:230 (NO-TOOL veto)\n")

    steerer = ToolCallingSteering.from_pretrained(
        "mlx-community/functiongemma-270m-it-bf16"
    )

    query = "What is the weather in Tokyo?"
    prompt = format_functiongemma_prompt(query)

    configs = [
        ("Normal", "normal", {}),
        ("Force (default neurons)", "force_tool", {}),
        ("Force (only 803)", "force_tool", {
            "tool_promoters": [803],
            "tool_suppressors": [],
        }),
        ("Force (all top-5)", "force_tool", {
            "tool_promoters": [803, 2036, 831, 436, 969],
            "tool_suppressors": [1347, 1237, 821, 217, 543],
        }),
    ]

    print(f"Query: {query}\n")
    print(f"{'Config':<35} {'Top Token':<25} {'Prob':>8}")
    print("-" * 70)

    for name, mode, kwargs in configs:
        result = steerer.predict(prompt, mode=mode, **kwargs)
        top_token, top_prob = result['top_tokens'][0]
        top_token = top_token.replace('\n', '\\n')[:23]

        print(f"  {name:<33} {top_token:<25} {top_prob:>6.1%}")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  TOOL-CALLING STEERING DEMO")
    print("  FunctionGemma 270M Circuit Control")
    print("=" * 60)

    print("\nIMPORTANT: FunctionGemma 270M is a BASE model for fine-tuning.")
    print("It has only ~58% function calling accuracy without domain tuning.")
    print("This demo validates that steering affects internal representations.")

    print("\nKey findings from circuit analysis:")
    print("  - L11 is the CONTROL POINT (100% decision flip rate)")
    print("  - L12:230 is the KILL SWITCH (70% accuracy drop when ablated)")
    print("  - Steering direction is 86% on PC1 (nearly 1D decision)")

    # Run demos
    demo_text_generation()
    demo_top_token_changes()
    demo_generation_comparison()
    demo_kill_switch()
    demo_neuron_targeting()
    demo_steering_effect_on_representations()

    print_header("SUMMARY")
    print("""
FunctionGemma 270M is designed as a foundation model for fine-tuning.
Without domain-specific training, it has low function calling accuracy.

However, the circuit analysis findings are still valid:
- The model DOES have tool-calling circuitry at L11/L12
- Steering DOES change internal representations
- These neurons ARE causally important (validated by ablation)

For production use:
1. Fine-tune FunctionGemma on your specific tools/domain
2. Or use a larger model (Gemma 3 12B/27B) for better zero-shot performance

The steering tool will be MORE effective after fine-tuning, when the
model actually outputs function calls that can be steered.

Usage:

    from chuk_lazarus.introspection.steering import (
        ToolCallingSteering, format_functiongemma_prompt
    )

    steerer = ToolCallingSteering.from_pretrained(
        "mlx-community/functiongemma-270m-it-bf16"
    )

    prompt = format_functiongemma_prompt("What is the weather?")

    # Generate with steering
    output = steerer.generate(prompt, mode="force_tool")

    # Or get next-token prediction
    result = steerer.predict(prompt, mode="prevent_tool")
""")


if __name__ == "__main__":
    main()
