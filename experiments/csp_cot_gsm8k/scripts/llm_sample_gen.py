#!/usr/bin/env python3
"""
Sample generation using spec generator + LLM expansion.

Uses chuk-lazarus UnifiedPipeline for inference with gpt-oss models.
"""

import json
from spec_generator import SpecGenerator


def build_expansion_prompt(spec: dict, use_raw: bool = False) -> str:
    """Build a prompt for LLM to expand spec into natural language."""
    # Extract key values
    agent1 = spec.get("agent1", "Person A")
    item = spec.get("item", "items")
    time_unit = spec.get("time_unit", "hour")
    time_units = spec.get("time_units", "hours")
    formula = spec.get("formula", "")
    answer = spec.get("answer", 0)

    # Build variable descriptions
    var_parts = []
    skip_keys = {"schema", "domain", "description", "agent1", "agent2", "item",
                 "time_unit", "time_units", "verb", "verbs", "formula", "answer", "trace"}
    for k, v in spec.items():
        if k not in skip_keys and isinstance(v, (int, float)):
            var_parts.append(f"{k}={int(v) if isinstance(v, float) and v.is_integer() else v}")

    variables_str = ", ".join(var_parts) if var_parts else ""

    if use_raw:
        # For raw generation (gpt-oss style)
        prompt = f"""Write a short math word problem (GSM-8K style).

Details:
- Person: {agent1}
- Item: {item}
- Numbers: {variables_str}
- The answer must be {int(answer) if isinstance(answer, float) and answer == int(answer) else answer}

Write only the problem (2-3 sentences + question), no solution:

"""
    else:
        # For chat-based models
        prompt = f"""Write a math word problem using these details:

Person: {agent1}
Item: {item}
Values: {variables_str}
The answer is {int(answer) if isinstance(answer, float) and answer == int(answer) else answer}

Write a short, natural word problem (2-3 sentences) ending with a question. Only output the problem text."""

    return prompt


SYSTEM_PROMPT = """You are a math teacher creating word problems.
Write natural problems using the EXACT numbers provided.
Be concise - 2-3 sentences plus a question. No solution or answer."""


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate word problems from specs using LLM")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", help="Model ID")
    parser.add_argument("--schema", default=None, help="Specific schema to use")
    parser.add_argument("--domain", default=None, help="Specific domain to use")
    parser.add_argument("--count", type=int, default=3, help="Number of samples")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--spec-only", action="store_true", help="Only show specs, don't run LLM")
    parser.add_argument("--raw", action="store_true", help="Use raw generation instead of chat (better for gpt-oss)")
    args = parser.parse_args()

    # Initialize spec generator
    gen = SpecGenerator()

    print("=" * 70)
    print("SPEC-BASED WORD PROBLEM GENERATION")
    print("=" * 70)
    print(f"Schemas available: {len(gen._schemas)}")
    print(f"Domains available: {len(gen._domains)}")
    print()

    # Generate specs
    schemas = [args.schema] if args.schema else list(gen._schemas.keys())

    specs = []
    for i in range(args.count):
        schema = schemas[i % len(schemas)] if args.schema else schemas[i % len(schemas)]
        spec = gen.generate_spec(schema, args.domain)
        specs.append(spec)

    if args.spec_only:
        print("Generated Specs (--spec-only mode):")
        print("-" * 70)
        for i, spec in enumerate(specs):
            print(f"\n[{i+1}] Schema: {spec['schema']}, Domain: {spec['domain']}")
            print(f"    Formula: {spec['formula']} = {spec['answer']}")
            print(f"    Prompt preview:")
            prompt = build_expansion_prompt(spec)
            # Show first few lines
            for line in prompt.split('\n')[:10]:
                print(f"      {line}")
            print("      ...")
        return

    # Load model
    print(f"Loading model: {args.model}")
    print("-" * 70)

    from chuk_lazarus.inference import UnifiedPipeline, UnifiedPipelineConfig

    config = UnifiedPipelineConfig(
        default_max_tokens=args.max_tokens,
        default_temperature=args.temperature,
    )

    pipeline = UnifiedPipeline.from_pretrained(args.model, config, verbose=True)

    # Auto-detect if we should use raw mode for gpt-oss
    use_raw = args.raw or "gpt-oss" in args.model.lower() or "gpt_oss" in args.model.lower()

    # Generate word problems
    print("\n" + "=" * 70)
    print("GENERATED WORD PROBLEMS")
    print("=" * 70)
    print(f"Mode: {'raw generation' if use_raw else 'chat'}")

    for i, spec in enumerate(specs):
        print(f"\n[{i+1}] Schema: {spec['schema']}, Domain: {spec['domain']}")
        print(f"    Formula: {spec['formula']} = {spec['answer']}")
        print("-" * 50)

        prompt = build_expansion_prompt(spec, use_raw=use_raw)

        if use_raw:
            result = pipeline.generate(
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )
        else:
            result = pipeline.chat(
                prompt,
                system_message=SYSTEM_PROMPT,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )

        # Clean up the output
        text = result.text.strip()
        # Stop at solution markers if present
        for marker in ["\n\nSolution:", "\n\n**Solution", "\n\nAnswer:", "**Answer"]:
            if marker in text:
                text = text.split(marker)[0].strip()

        print(f"Generated problem:")
        print(text)
        print()


if __name__ == "__main__":
    main()
