#!/usr/bin/env python
"""
Test end-to-end CSP Virtual Expert with model generation.

Tests whether:
1. Model can generate CSP-formatted output given a scheduling prompt
2. CSP extractor can parse the generated output
3. Solver can solve the extracted problem
"""

from __future__ import annotations

import argparse
import mlx.core as mx


def generate_text(model, tokenizer, prompt: str, max_tokens: int = 200) -> str:
    """Simple greedy generation."""
    tokens = tokenizer.encode(prompt)

    generated = list(tokens)

    for _ in range(max_tokens):
        # Forward pass
        x = mx.array([generated])

        # Use model's forward method if available
        if hasattr(model, '__call__'):
            output = model(x)
            # Handle different output formats
            if hasattr(output, 'logits'):
                logits = output.logits
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
        else:
            # Manual forward
            if hasattr(model, 'model'):
                hidden = model.model.embed_tokens(x)
                for layer in model.model.layers:
                    output = layer(hidden)
                    hidden = output[0] if isinstance(output, tuple) else output
                hidden = model.model.norm(hidden)
                logits = model.lm_head(hidden)
            else:
                hidden = model.embed_tokens(x)
                for layer in model.layers:
                    output = layer(hidden)
                    hidden = output[0] if isinstance(output, tuple) else output
                hidden = model.norm(hidden)
                logits = model.lm_head(hidden)

        # Get next token (greedy)
        next_token = mx.argmax(logits[0, -1, :]).item()
        generated.append(next_token)

        # Check for EOS or generation patterns
        eos_id = getattr(tokenizer, 'eos_token_id', None)
        if eos_id and next_token == eos_id:
            break

        # Check if we've generated a trigger pattern
        text_so_far = tokenizer.decode(generated[len(tokens):])
        if "SOLVE:" in text_so_far or "Solution:" in text_so_far:
            # Continue a bit more to see if model adds anything
            pass

    return tokenizer.decode(generated[len(tokens):])


def main():
    parser = argparse.ArgumentParser(description="Test CSP generation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    print("=" * 70)
    print("CSP Virtual Expert - Generation Test")
    print("=" * 70)

    # Load model
    print(f"\nLoading model: {args.model}")
    from chuk_lazarus.models_v2.loader import load_model

    loaded = load_model(args.model)
    model = loaded.model
    tokenizer = loaded.tokenizer
    print(f"Loaded: {loaded.family_type}")

    # Import CSP components
    from experiments.csp_virtual_expert.expert.csp_plugin import CSPVirtualExpertPlugin
    from experiments.csp_virtual_expert.extraction.csp_extractor import extract_csp_spec

    plugin = CSPVirtualExpertPlugin()

    # Test prompts
    test_prompts = [
        {
            "name": "Structured prompt",
            "prompt": """Format your scheduling response as:
TASKS: [name:duration, ...]
CONSTRAINTS: [constraint, ...]
SOLVE:

Schedule meetings: Alice needs 2 hours, Bob needs 1 hour. They can't overlap.
Response:""",
        },
        {
            "name": "Natural prompt",
            "prompt": "Schedule my day: gym (1 hour), lunch (1.5 hours), meeting (2 hours). What's the optimal order?",
        },
        {
            "name": "Direct CSP format (no generation needed)",
            "prompt": """TASKS: [Alice:2hr, Bob:1hr, Carol:1.5hr]
CONSTRAINTS: [no_overlap(Alice, Bob), no_overlap(Bob, Carol)]
OBJECTIVE: minimize_makespan
SOLVE:""",
        },
    ]

    print("\n" + "=" * 70)

    for test in test_prompts:
        print(f"\n--- Test: {test['name']} ---")
        print(f"Prompt: {test['prompt'][:100]}...")

        # For direct format tests, skip generation
        if "SOLVE:" in test['prompt']:
            output = test['prompt']
            print(f"\n[Using direct input - no generation]")
        else:
            print(f"\n[Generating...]")
            output = test['prompt'] + generate_text(model, tokenizer, test['prompt'], args.max_tokens)
            print(f"Generated:\n{output[-300:]}")

        # Test CSP detection
        can_handle = plugin.can_handle(output)
        print(f"\nCSP Plugin can_handle: {can_handle}")

        if can_handle:
            # Test extraction
            spec = extract_csp_spec(output)
            if spec:
                print(f"Extracted {len(spec.tasks)} tasks, {len(spec.constraints)} constraints")

                # Test solving
                result = plugin.execute(output)
                print(f"Solver result:\n{result}")
            else:
                print("Extraction failed - trying loose extraction")
                spec = plugin._extract_loose(output)
                if spec:
                    print(f"Loose extracted {len(spec.tasks)} tasks")

        print("-" * 50)

    # Final: Test the full plugin directly with known-good input
    print("\n" + "=" * 70)
    print("Direct Plugin Test (Known Good Input)")
    print("=" * 70)

    direct_test = """
TASKS: [Meeting:1hr, Coding:3hr, Review:1hr, Lunch:1hr]
WINDOW: [9:00, 17:00]
CONSTRAINTS: [Meeting before Coding, Lunch before Review]
OBJECTIVE: minimize_makespan
SOLVE:
"""
    print(f"Input:\n{direct_test}")
    result = plugin.execute(direct_test)
    print(f"Result:\n{result}")


if __name__ == "__main__":
    main()
