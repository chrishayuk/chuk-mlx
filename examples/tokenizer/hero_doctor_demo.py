#!/usr/bin/env python3
"""
Hero Demo 1: Tokenizer Doctor + Chat Template Sanity Check

This demo shows how to:
1. Run a comprehensive health check on a tokenizer
2. Detect and validate chat template format
3. Auto-patch missing or malformed chat templates
4. Save patched tokenizers for production use

Usage:
    python examples/tokenizer/hero_doctor_demo.py

Or via CLI:
    uvx chuk-lazarus tokenizer doctor -t "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --verbose
    uvx chuk-lazarus tokenizer doctor -t "model-name" --fix --format chatml --output ./patched
"""

from chuk_lazarus.data.tokenizers.fingerprint import compute_fingerprint
from chuk_lazarus.data.tokenizers.runtime.chat_templates import (
    ChatTemplateRegistry,
    TemplateFormat,
    patch_chat_template,
    suggest_template_for_model,
    validate_chat_template,
)
from chuk_lazarus.utils.tokenizer_loader import load_tokenizer


def run_health_check(model_name: str) -> dict:
    """Run comprehensive tokenizer health check."""
    print(f"\n{'=' * 60}")
    print(f"Tokenizer Doctor: {model_name}")
    print(f"{'=' * 60}")

    results = {"model": model_name, "healthy": True, "issues": [], "warnings": []}

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    try:
        tokenizer = load_tokenizer(model_name)
        print(f"   Vocab size: {len(tokenizer.get_vocab()):,}")
    except Exception as e:
        results["healthy"] = False
        results["issues"].append(f"Load failed: {e}")
        return results

    # Check special tokens
    print("\n2. Checking special tokens...")
    for attr, name in [
        ("bos_token_id", "BOS"),
        ("eos_token_id", "EOS"),
        ("pad_token_id", "PAD"),
        ("unk_token_id", "UNK"),
    ]:
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            print(f"   {name}: {token_id}")
        else:
            print(f"   {name}: NOT SET")
            if name in ("BOS", "EOS"):
                results["warnings"].append(f"Missing {name} token")

    # Validate chat template
    print("\n3. Validating chat template...")
    validation = validate_chat_template(tokenizer)

    if validation.is_valid:
        print(f"   Status: VALID")
        print(f"   Format: {validation.format.value}")
        caps = [c.value for c in validation.capabilities]
        print(f"   Capabilities: {', '.join(caps) if caps else 'basic'}")
    else:
        print(f"   Status: INVALID or MISSING")
        results["healthy"] = False
        for issue in validation.issues:
            results["issues"].append(issue.message)
            print(f"   ERROR: {issue.message}")

    # Test roundtrip encoding
    print("\n4. Testing encode/decode roundtrip...")
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Special chars: @#$%^&*()",
        "Unicode: 你好 🎉",
    ]

    failures = 0
    for text in test_texts:
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(encoded, skip_special_tokens=True)
            # Normalize for comparison
            if " ".join(text.split()) != " ".join(decoded.split()):
                failures += 1
        except Exception:
            failures += 1

    if failures == 0:
        print(f"   All {len(test_texts)} tests: PASS")
    else:
        print(f"   {len(test_texts) - failures}/{len(test_texts)} tests: PASS")
        results["warnings"].append(f"{failures} roundtrip tests had issues")

    # Compute fingerprint
    print("\n5. Computing fingerprint...")
    try:
        fp = compute_fingerprint(tokenizer)
        print(f"   Fingerprint: {fp.fingerprint}")
        print(f"   Vocab hash:  {fp.vocab_hash}")
        results["fingerprint"] = fp.fingerprint
    except Exception as e:
        print(f"   Error: {e}")
        results["issues"].append(f"Fingerprint failed: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"{'=' * 60}")

    if results["healthy"] and not results["warnings"]:
        print("   Status: HEALTHY")
    elif results["healthy"]:
        print(f"   Status: HEALTHY with {len(results['warnings'])} warnings")
        for w in results["warnings"]:
            print(f"   WARN: {w}")
    else:
        print(f"   Status: ISSUES FOUND ({len(results['issues'])})")
        for i in results["issues"]:
            print(f"   ERROR: {i}")

    return results


def demo_auto_fix(model_name: str):
    """Demonstrate auto-patching a tokenizer."""
    print(f"\n{'=' * 60}")
    print(f"Auto-Fix Demo: {model_name}")
    print(f"{'=' * 60}")

    tokenizer = load_tokenizer(model_name)

    # Check if template exists
    has_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template

    if has_template:
        print(f"\nTokenizer already has chat template")
        registry = ChatTemplateRegistry()
        format = registry.detect_format(tokenizer.chat_template)
        print(f"Format: {format.value}")
        return

    # Find best template
    print("\nNo chat template found. Suggesting fix...")
    suggested = suggest_template_for_model(model_name)

    if suggested:
        print(f"Suggested format: {suggested.format.value}")
        print(f"Description: {suggested.description}")

        # Apply patch
        print("\nApplying patch...")
        success = patch_chat_template(tokenizer, template_format=suggested.format)

        if success:
            print("Patch applied successfully!")

            # Validate new template
            validation = validate_chat_template(tokenizer)
            print(f"Validation: {'PASS' if validation.is_valid else 'FAIL'}")

            # Test chat
            messages = [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
            ]
            result = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            print(f"\nSample output:\n{result[:200]}...")
    else:
        print("Could not determine appropriate template format")
        print("Use --format flag to specify: chatml, llama, phi, gemma, etc.")


def demo_template_registry():
    """Demonstrate the chat template registry."""
    print(f"\n{'=' * 60}")
    print("Chat Template Registry")
    print(f"{'=' * 60}")

    registry = ChatTemplateRegistry()

    print("\nAvailable Templates:")
    print("-" * 40)

    for template in registry.list_templates():
        print(f"\n{template.format.value.upper()}")
        print(f"  Description: {template.description}")
        print(f"  System msgs: {'Yes' if template.supports_system else 'No'}")
        print(f"  Model families: {', '.join(template.model_families)}")

    # Test model family detection
    print("\n" + "=" * 60)
    print("Model Family Detection")
    print("-" * 40)

    test_models = [
        "meta-llama/Llama-2-7b-chat-hf",
        "Qwen/Qwen2-0.5B",
        "microsoft/Phi-3-mini-4k-instruct",
        "google/gemma-2b",
        "mistralai/Mistral-7B-Instruct-v0.1",
    ]

    for model in test_models:
        template = registry.get_for_model_family(model)
        if template:
            print(f"  {model} -> {template.format.value}")
        else:
            print(f"  {model} -> Unknown")


if __name__ == "__main__":
    print("=" * 60)
    print("HERO DEMO 1: Tokenizer Doctor + Chat Template Sanity Check")
    print("=" * 60)

    # Demo 1: Health check on TinyLlama
    run_health_check("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Demo 2: Show template registry
    demo_template_registry()

    # Demo 3: Show auto-fix capability
    # (TinyLlama already has a template, so this is just illustrative)
    # demo_auto_fix("some-model-without-template")

    print("\n" + "=" * 60)
    print("Demo complete! Try these CLI commands:")
    print("=" * 60)
    print("  # Basic health check:")
    print('  uvx chuk-lazarus tokenizer doctor -t "TinyLlama/TinyLlama-1.1B-Chat-v1.0"')
    print("\n  # Auto-fix missing template:")
    print('  uvx chuk-lazarus tokenizer doctor -t "model" --fix')
    print("\n  # Save patched tokenizer:")
    print('  uvx chuk-lazarus tokenizer doctor -t "model" --fix --output ./patched')
