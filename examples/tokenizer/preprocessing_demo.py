"""
Tokenizer Preprocessing Demo

Demonstrates the preprocessing module features:
- Numeric normalization
- Structure token injection
- Pre/post tokenization hooks
- Tokenizer profiles
- Byte fallback

These features improve tokenization for math/reasoning and tool use.
"""

from chuk_lazarus.data.tokenizers.preprocessing import (
    # Hooks
    HookedTokenizer,
    # Profiles
    ProfiledTokenizer,
    create_inference_profile,
    create_math_pipeline,
    create_standard_pipeline,
    create_tool_pipeline,
    create_training_profile,
    # Numeric
    detect_numbers,
    # Structure
    detect_structures,
    inject_structure_tokens,
    normalize_numbers,
    restore_numbers,
    restore_structures,
    # Fallback
    wrap_with_fallback,
)
from chuk_lazarus.utils.tokenizer_loader import load_tokenizer


def demo_numeric_detection():
    """Demonstrate number detection and normalization."""
    print("=" * 60)
    print("Numeric Detection and Normalization")
    print("=" * 60)

    texts = [
        "Pi is 3.14159",
        "Avogadro's number: 6.022e23",
        "Hex color: 0xFF00FF",
        "Success rate: 95%",
        "Half is 1/2",
        "Sum: 10 + 20 = 30",
    ]

    print("\nDetecting numbers:")
    for text in texts:
        spans = detect_numbers(text)
        print(f"\n  '{text}'")
        for span in spans:
            print(f"    -> {span.original} ({span.format.value}) = {span.value}")

    print("\n\nNormalizing numbers with placeholders:")
    text = "The answer is 42 and pi is 3.14159"
    encoding = normalize_numbers(text)
    print(f"  Original: {text}")
    print(f"  Encoded:  {encoding.encoded_text}")
    print(f"  Mapping:  {encoding.mapping}")

    # Restore
    restored = restore_numbers(encoding.encoded_text, encoding.mapping)
    print(f"  Restored: {restored}")

    # Token savings
    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    math_text = "Calculate: 3.14159 * 2.71828 = 8.539734"

    from chuk_lazarus.data.tokenizers.preprocessing.numeric import (
        get_numeric_token_savings,
    )

    savings = get_numeric_token_savings(math_text, tokenizer)
    print("\n\nToken savings for math expression:")
    print(f"  Original: {savings['original_tokens']} tokens")
    print(f"  Normalized: {savings['normalized_tokens']} tokens")
    print(f"  Savings: {savings['savings']} tokens ({savings['savings_percent']:.1f}%)")


def demo_structure_detection():
    """Demonstrate structure detection and injection."""
    print("\n" + "=" * 60)
    print("Structure Detection and Injection")
    print("=" * 60)

    texts = [
        "ID: 550e8400-e29b-41d4-a716-446655440000",
        "Email: user@example.com",
        "Visit https://api.example.com/v1/data",
        "Server: 192.168.1.1",
        "Date: 2024-01-15T14:30:00Z",
        "Path: /usr/local/bin/python",
    ]

    print("\nDetecting structures:")
    for text in texts:
        spans = detect_structures(text)
        print(f"\n  '{text}'")
        for span in spans:
            print(f"    -> {span.structure_type.value}: {span.original[:30]}...")

    print("\n\nInjecting structure tokens:")
    complex_text = (
        "User 550e8400-e29b-41d4-a716-446655440000 at 192.168.1.1 sent email to admin@example.com"
    )
    encoding = inject_structure_tokens(complex_text)
    print(f"  Original: {complex_text}")
    print(f"  Encoded:  {encoding.encoded_text}")

    # Restore
    restored = restore_structures(encoding.encoded_text, encoding.mapping)
    print(f"  Restored: {restored}")
    print(f"  Match: {restored == complex_text}")


def demo_hook_pipeline():
    """Demonstrate the hook pipeline."""
    print("\n" + "=" * 60)
    print("Hook Pipeline")
    print("=" * 60)

    _ = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # Verify tokenizer loads

    # Standard pipeline
    print("\nStandard pipeline (numeric + structure):")
    pipeline = create_standard_pipeline(numeric=True, structure=True)

    text = "User 550e8400-e29b-41d4-a716-446655440000 has balance $99.99"
    transformed = pipeline.pre_tokenize(text)
    print(f"  Original:    {text}")
    print(f"  Transformed: {transformed}")

    # Restore
    restored = pipeline.post_decode(transformed)
    print(f"  Restored:    {restored}")

    # Math pipeline
    print("\n\nMath pipeline:")
    math_pipeline = create_math_pipeline()
    math_text = "f(x) = x^2 + 3.14 * x"
    math_transformed = math_pipeline.pre_tokenize(math_text)
    print(f"  Original:    {math_text}")
    print(f"  Transformed: {math_transformed}")

    # Tool pipeline
    print("\n\nTool pipeline:")
    tool_pipeline = create_tool_pipeline()
    tool_text = "Call API at https://api.example.com with ID 550e8400-e29b-41d4-a716-446655440000"
    tool_transformed = tool_pipeline.pre_tokenize(tool_text)
    print(f"  Original:    {tool_text}")
    print(f"  Transformed: {tool_transformed}")


def demo_hooked_tokenizer():
    """Demonstrate the hooked tokenizer."""
    print("\n" + "=" * 60)
    print("Hooked Tokenizer")
    print("=" * 60)

    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    pipeline = create_standard_pipeline()
    hooked = HookedTokenizer(tokenizer, pipeline)

    text = "Value is 3.14159 at IP 192.168.1.1"
    print(f"\nText: {text}")

    # Compare tokens
    original_tokens = tokenizer.encode(text, add_special_tokens=False)
    hooked_tokens = hooked.encode(text, add_special_tokens=False)

    print(f"\nOriginal tokenization: {len(original_tokens)} tokens")
    print(f"Hooked tokenization:   {len(hooked_tokens)} tokens")
    print(f"Token reduction:       {len(original_tokens) - len(hooked_tokens)}")


def demo_profiles():
    """Demonstrate tokenizer profiles."""
    print("\n" + "=" * 60)
    print("Tokenizer Profiles")
    print("=" * 60)

    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Training profile
    print("\nTraining profile:")
    training_profile = create_training_profile(
        normalize_numbers=True,
        inject_structures=True,
        max_length=128,
    )
    profiled_training = ProfiledTokenizer(tokenizer, training_profile)
    print(f"  Mode: {training_profile.mode.value}")
    print(f"  Normalize numbers: {training_profile.normalize_numbers}")
    print(f"  Inject structures: {training_profile.inject_structures}")
    print(f"  Max length: {training_profile.max_length}")

    # Inference profile
    print("\nInference profile:")
    inference_profile = create_inference_profile()
    profiled_inference = ProfiledTokenizer(tokenizer, inference_profile)
    print(f"  Mode: {inference_profile.mode.value}")
    print(f"  Normalize numbers: {inference_profile.normalize_numbers}")
    print(f"  Byte fallback: {inference_profile.byte_fallback}")

    # Encode with different profiles
    text = "Result: 3.14159 at 192.168.1.1"
    print(f"\n\nEncoding '{text}':")

    training_tokens = profiled_training.encode(text, add_special_tokens=False)
    inference_tokens = profiled_inference.encode(text, add_special_tokens=False)

    print(f"  Training tokens: {len(training_tokens)}")
    print(f"  Inference tokens: {len(inference_tokens)}")

    # Switch profiles
    print("\nSwitching profiles on same tokenizer:")
    profiled_training.set_profile(inference_profile)
    print(f"  Now using: {profiled_training.profile.name}")


def demo_byte_fallback():
    """Demonstrate byte fallback."""
    print("\n" + "=" * 60)
    print("Byte Fallback")
    print("=" * 60)

    tokenizer = load_tokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    wrapper = wrap_with_fallback(tokenizer)

    test_strings = [
        "Hello, world!",
        "Café au lait",
        "日本語テスト",
        "Emoji: 🎉🎊",
        "Price: €100",
    ]

    print("\nTesting byte safety:")
    for text in test_strings:
        stats = wrapper.get_fallback_stats(text)
        tokens = wrapper.encode(text, add_special_tokens=False)
        _ = wrapper.decode(tokens)  # Verify roundtrip works

        print(f"\n  Input:    '{text}'")
        print(f"  Tokens:   {len(tokens)}")
        print(f"  Fallback: {stats.fallback_chars} chars ({stats.fallback_ratio:.1%})")

    # Stress test
    print("\n\nRunning stress test:")
    from chuk_lazarus.data.tokenizers.preprocessing.fallback import run_byte_safety_tests

    results = run_byte_safety_tests(tokenizer)
    print(f"  Total tests: {results['total_tests']}")
    print(f"  Passed: {results['passed']}")
    print(f"  Failed: {results['failed']}")


def main():
    """Run all preprocessing demos."""
    print("Tokenizer Preprocessing Demo")
    print("=" * 60)

    demo_numeric_detection()
    demo_structure_detection()
    demo_hook_pipeline()
    demo_hooked_tokenizer()
    demo_profiles()
    demo_byte_fallback()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
