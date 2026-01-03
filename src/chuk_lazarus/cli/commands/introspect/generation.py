"""Generation commands for introspection CLI.

This module contains functions for token generation analysis.
"""

import logging

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class AnswerOnsetResult(BaseModel):
    """Result of finding answer onset in generated output."""

    model_config = ConfigDict(frozen=True)

    onset_index: int | None = Field(
        default=None, description="Token index where answer first appears"
    )
    onset_token: str | None = Field(default=None, description="Token at onset position")
    is_answer_first: bool | None = Field(
        default=None, description="Whether answer appears in first token"
    )
    answer_found: bool = Field(default=False, description="Whether answer was found at all")


def introspect_generate(args):
    """Generate multiple tokens to test next-token lock hypothesis.

    Tests whether format issues (like missing trailing space) cause:
    A) Simple next-token lock: model completes format, then computes
    B) Answer-onset routing: model changes WHEN to emit answer
    C) Computation blocked: model can't produce correct answer at all
    """
    from mlx_lm import generate, load

    from ....introspection import apply_chat_template, extract_expected_answer

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model)

    # Load external chat template if available (e.g., GPT-OSS)
    _load_external_chat_template(tokenizer, args.model)

    # Check if using raw mode (no chat template)
    use_raw = getattr(args, "raw", False)
    has_chat_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template

    if use_raw:
        print("Mode: RAW (no chat template)")
    elif has_chat_template:
        print("Mode: CHAT (using chat template)")
        print("  Add --raw to test direct prompts without chat formatting")
    else:
        print("Mode: RAW (model has no chat template)")

    # Parse prompts
    if args.prompts.startswith("@"):
        with open(args.prompts[1:]) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [p.strip() for p in args.prompts.split("|")]

    # If comparing format, create with/without space variants
    if args.compare_format:
        expanded = []
        for p in prompts:
            base = p.rstrip()
            expanded.append(base)  # without trailing space
            expanded.append(base + " ")  # with trailing space
        prompts = expanded

    print(f"\nGenerating {args.max_tokens} tokens per prompt")
    print(f"Temperature: {args.temperature}")
    print()

    results = []
    for prompt in prompts:
        # Apply chat template unless --raw is specified
        formatted_prompt = prompt
        if not use_raw and has_chat_template:
            formatted_prompt = apply_chat_template(tokenizer, prompt)

        if args.temperature == 0:
            output = generate(
                model,
                tokenizer,
                prompt=formatted_prompt,
                max_tokens=args.max_tokens,
                verbose=False,
            )
        else:
            output = generate(
                model,
                tokenizer,
                prompt=formatted_prompt,
                max_tokens=args.max_tokens,
                temp=args.temperature,
                verbose=False,
            )

        # Compute expected answer and find onset
        expected = extract_expected_answer(prompt)
        onset_info = _find_answer_onset(output, expected, tokenizer)

        # Show results
        has_space = prompt.endswith(" ")
        marker = "[space]" if has_space else "[no-space]"
        print(f"{marker} {prompt!r}")
        print(f"  -> {output!r}")

        # Show answer onset info
        if expected:
            if onset_info.answer_found:
                onset_str = f"onset={onset_info.onset_index}"
                if onset_info.is_answer_first:
                    onset_str += " (answer-first)"
                else:
                    onset_str += " (delayed)"
                print(f"  Expected: {expected}, {onset_str}")
            else:
                print(f"  Expected: {expected}, NOT FOUND in output")

        # Token-by-token breakdown if requested
        if args.show_tokens:
            prompt_ids = tokenizer.encode(formatted_prompt)
            output_ids = tokenizer.encode(formatted_prompt + output)
            gen_ids = output_ids[len(prompt_ids) :]

            print("  Tokens: ", end="")
            for i, tid in enumerate(gen_ids[:10]):
                tok = tokenizer.decode([tid])
                # Highlight the onset token
                if expected and onset_info.onset_index == i:
                    print(f"[{tok!r}] ", end="")
                else:
                    print(f"{tok!r} ", end="")
            if len(gen_ids) > 10:
                print("...")
            else:
                print()
        print()

        results.append(
            {
                "prompt": prompt,
                "has_trailing_space": has_space,
                "output": output,
                "expected_answer": expected,
                **onset_info,
            }
        )

    # Summary if comparing format
    if args.compare_format and len(results) >= 2:
        print("=== Format Comparison Summary ===")
        print()
        print(f"{'Prompt':<20} {'No-Space':<12} {'With-Space':<12} {'Diagnosis'}")
        print("-" * 70)

        for i in range(0, len(results), 2):
            no_space = results[i]
            with_space = results[i + 1]
            base_prompt = no_space["prompt"][:18]

            # Determine diagnosis based on onset patterns
            ns_onset = no_space.get("onset_index")
            ws_onset = with_space.get("onset_index")
            ns_found = no_space.get("answer_found", False)
            ws_found = with_space.get("answer_found", False)

            # Format onset display
            ns_str = f"onset={ns_onset}" if ns_onset is not None else "not found"
            ws_str = f"onset={ws_onset}" if ws_onset is not None else "not found"

            # Classify the behavior
            if not ns_found and not ws_found:
                diagnosis = "BOTH FAIL"
            elif not ns_found and ws_found:
                diagnosis = "COMPUTE BLOCKED"
            elif ns_found and not ws_found:
                diagnosis = "WEIRD (no-space works?)"
            elif ns_onset == ws_onset or (ns_onset <= 1 and ws_onset <= 1):
                diagnosis = "SPACE-LOCK ONLY"
            elif ns_onset is not None and ws_onset is not None and ns_onset > ws_onset + 2:
                diagnosis = "ONSET ROUTING"
            else:
                diagnosis = "MINOR DIFFERENCE"

            print(f"{base_prompt:<20} {ns_str:<12} {ws_str:<12} {diagnosis}")

        print()
        print("Legend:")
        print("  SPACE-LOCK ONLY  = Just adds space token, same answer timing")
        print("  ONSET ROUTING    = Answer delayed (mode/style switch)")
        print("  COMPUTE BLOCKED  = Answer not produced without space")
        print("  MINOR DIFFERENCE = Small onset difference")

    # Save if requested
    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def _normalize_number(s: str) -> str:
    """Normalize a number string by removing formatting characters."""
    import re

    # Remove commas, thin spaces (unicode \u202f), regular spaces, and other separators
    return re.sub(r"[\s,\u202f\u00a0]+", "", s)


def _find_answer_onset(output: str, expected_answer: str | None, tokenizer) -> AnswerOnsetResult:
    """Find where the answer first appears in the output.

    Returns:
        AnswerOnsetResult with onset information
    """
    if expected_answer is None:
        return AnswerOnsetResult()

    # Normalize expected answer (remove any formatting)
    expected_normalized = _normalize_number(expected_answer)

    # Tokenize output
    tokens = []
    output_ids = tokenizer.encode(output)
    for tid in output_ids:
        tokens.append(tokenizer.decode([tid]))

    # Find first position where expected answer appears
    # Check both in individual tokens and cumulative string
    cumulative = ""
    for i, tok in enumerate(tokens):
        cumulative += tok
        # Check if answer appears in cumulative output (normalized)
        if expected_normalized in _normalize_number(cumulative):
            return AnswerOnsetResult(
                onset_index=i,
                onset_token=tok,
                is_answer_first=i <= 1,  # Answer in first 2 tokens
                answer_found=True,
            )

    return AnswerOnsetResult(is_answer_first=False)


def _load_external_chat_template(tokenizer, model_path: str) -> None:
    """Load external chat template from model directory if available.

    Some models (like GPT-OSS) store the chat template in a separate
    chat_template.jinja file rather than in tokenizer_config.json.
    """
    from pathlib import Path

    from huggingface_hub import snapshot_download

    # Try to find model path
    try:
        # If it's a HF model ID, get the local cache path
        local_path = Path(snapshot_download(model_path, allow_patterns=["chat_template.jinja"]))
    except Exception:
        local_path = Path(model_path)

    chat_template_path = local_path / "chat_template.jinja"
    if chat_template_path.exists() and not tokenizer.chat_template:
        try:
            with open(chat_template_path) as f:
                tokenizer.chat_template = f.read()
        except Exception:
            pass


__all__ = [
    "introspect_generate",
]
