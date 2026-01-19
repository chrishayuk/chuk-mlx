"""
Generator with Compiler Expert.

Integrates the compiler virtual expert into generation, detecting completed
code blocks and injecting execution results back into the context.

Based on the MoE bypass pattern from ir_attention_routing/moe_bypass.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from compiler_plugin import CompilerExpertPlugin, ExecutionResult


# =============================================================================
# CODE DETECTION CLASSIFIER
# =============================================================================


class CodeBlockClassifier(nn.Module):
    """
    Classifier to detect "code execution intent" from hidden states.

    Trained on hidden states at ``` positions to distinguish:
    - Code block that should be executed
    - Code block that's just being explained
    - Incomplete code block
    """

    def __init__(self, hidden_dim: int, hidden_size: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)  # execute vs don't execute
        self.dropout = nn.Dropout(0.1)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


# =============================================================================
# HIDDEN STATE EXTRACTION
# =============================================================================


def get_hidden_at_position(
    model, input_ids: mx.array, position: int, layer: int
) -> mx.array:
    """Extract hidden state at a specific position and layer."""
    hidden = model.model.embed_tokens(input_ids)

    for i, layer_module in enumerate(model.model.layers):
        output = layer_module(hidden, mask=None)
        if hasattr(output, "hidden_states"):
            hidden = output.hidden_states
        elif isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        if i == layer:
            break

    mx.eval(hidden)
    return hidden[0, position, :]


def get_hidden_dim(model) -> int:
    """Get hidden dimension from model."""
    embed = model.model.embed_tokens
    if hasattr(embed, "weight"):
        weight = embed.weight
        if hasattr(weight, "shape"):
            return weight.shape[1]
    params = embed.parameters()
    if isinstance(params, dict):
        for val in params.values():
            if hasattr(val, "shape") and len(val.shape) == 2:
                return val.shape[1]
    raise ValueError("Cannot determine hidden dim")


# =============================================================================
# TRAINING DATA GENERATION
# =============================================================================


def generate_training_data(n_samples: int = 1000) -> list[dict]:
    """
    Generate training examples for code execution detection.

    Returns examples with:
    - text: The prompt text
    - label: 1 if should execute, 0 if not
    - trigger_pos: Position of the closing ```
    """
    import random

    data = []

    # Positive examples: complete code blocks that should execute
    positive_templates = [
        "```python\n{code}\n```",
        "Here's the code:\n```python\n{code}\n```",
        "Let me test this:\n```python\n{code}\n```",
        "Running:\n```\n{code}\n```",
    ]

    code_snippets = [
        "print('hello')",
        "x = 5\nprint(x)",
        "def add(a, b):\n    return a + b\nprint(add(2, 3))",
        "result = sum(range(10))\nprint(result)",
        "for i in range(3):\n    print(i)",
        "x = [1, 2, 3]\nprint(sorted(x))",
    ]

    for _ in range(n_samples // 2):
        template = random.choice(positive_templates)
        code = random.choice(code_snippets)
        text = template.format(code=code)

        data.append({"text": text, "label": 1, "is_complete": True})

    # Negative examples: incomplete blocks, explanations, etc.
    negative_templates = [
        "In Python, you write:\n```python\n{code}",  # Incomplete
        "The syntax is `{inline_code}`",  # Inline, not block
        "Let me explain: {explanation}",  # No code
        "```python\n# Just a comment\n```",  # Comment only
        "What does `{inline_code}` do?",  # Question about code
    ]

    explanations = [
        "A function is defined with def",
        "Lists are created with square brackets",
        "The for loop iterates over a sequence",
    ]

    inline_codes = [
        "print()",
        "for i in range(n)",
        "def foo():",
    ]

    for _ in range(n_samples // 2):
        template = random.choice(negative_templates)

        if "{code}" in template:
            text = template.format(code=random.choice(code_snippets))
        elif "{explanation}" in template:
            text = template.format(explanation=random.choice(explanations))
        elif "{inline_code}" in template:
            text = template.format(inline_code=random.choice(inline_codes))
        else:
            text = template

        data.append({"text": text, "label": 0, "is_complete": False})

    random.shuffle(data)
    return data


def train_code_classifier(
    model, tokenizer, hidden_dim: int, layer: int = 10
) -> CodeBlockClassifier:
    """Train the code execution classifier."""
    print("  Training code execution classifier...")

    data = generate_training_data(1000)
    classifier = CodeBlockClassifier(hidden_dim)
    optimizer = optim.Adam(learning_rate=1e-3)

    def loss_fn(clf, x, y):
        logits = clf(x)
        return nn.losses.cross_entropy(logits, y).mean()

    loss_and_grad_fn = nn.value_and_grad(classifier, loss_fn)

    # Extract hidden states at ``` positions
    X = []
    y = []

    for example in data[:500]:
        tokens = tokenizer.encode(example["text"])
        input_ids = mx.array([tokens])

        # Find last ``` position
        text = example["text"]
        last_backtick = text.rfind("```")
        if last_backtick == -1:
            continue

        # Find token position for this character position
        # Approximate: tokenize up to that point
        prefix = text[: last_backtick + 3]
        prefix_tokens = tokenizer.encode(prefix)
        trigger_pos = len(prefix_tokens) - 1

        if trigger_pos >= len(tokens):
            trigger_pos = len(tokens) - 1

        hidden = get_hidden_at_position(model, input_ids, trigger_pos, layer)
        X.append(hidden)
        y.append(example["label"])

    if len(X) < 100:
        print("  Warning: Not enough training examples")
        return classifier

    X = mx.stack(X)
    y = mx.array(y)
    mx.eval(X, y)

    # Train
    batch_size = 32
    n_samples = len(X)

    for epoch in range(10):
        import random

        perm = mx.array(random.sample(range(n_samples), n_samples))
        X_shuffled = X[perm]
        y_shuffled = y[perm]

        epoch_loss = 0
        for i in range(0, n_samples, batch_size):
            batch_x = X_shuffled[i : i + batch_size]
            batch_y = y_shuffled[i : i + batch_size]

            loss, grads = loss_and_grad_fn(classifier, batch_x, batch_y)
            optimizer.update(classifier, grads)
            mx.eval(classifier.parameters(), optimizer.state)
            epoch_loss += float(loss.item())

    # Validate
    classifier.eval()
    logits = classifier(X)
    preds = mx.argmax(logits, axis=-1)
    acc = float((preds == y).mean().item())
    print(f"  Classifier accuracy: {acc:.1%}")

    return classifier


# =============================================================================
# GENERATOR WITH COMPILER EXPERT
# =============================================================================


@dataclass
class GenerationResult:
    """Result from generation with compiler expert."""

    prompt: str
    full_output: str
    compiler_triggered: bool
    code_block: str | None
    execution_result: str | None
    confidence: float | None


class GeneratorWithCompiler:
    """
    Generator that invokes compiler expert on completed code blocks.

    When generating and a code block is completed:
    1. Check classifier confidence for "should execute"
    2. If high confidence, extract and execute code
    3. Inject execution results back into context
    4. Continue generation
    """

    def __init__(
        self,
        model,
        tokenizer,
        classifier: CodeBlockClassifier,
        layer: int = 10,
        confidence_threshold: float = 0.8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.classifier = classifier
        self.layer = layer
        self.threshold = confidence_threshold
        self.compiler = CompilerExpertPlugin()

    def _find_code_block_end(self, text: str) -> int | None:
        """Find position of closing ``` if code block just completed."""
        # Check if text ends with closing ```
        if not text.rstrip().endswith("```"):
            return None

        # Find the matching opening
        # Count ``` occurrences
        positions = [m.start() for m in re.finditer(r"```", text)]

        if len(positions) < 2:
            return None

        # Check if last one closes an open block
        if len(positions) % 2 == 0:
            return positions[-1]

        return None

    def _classify_hidden(
        self, input_ids: mx.array, position: int
    ) -> tuple[bool, float]:
        """Classify if code should be executed from hidden state."""
        hidden = get_hidden_at_position(self.model, input_ids, position, self.layer)

        self.classifier.eval()
        logits = self.classifier(hidden.reshape(1, -1))
        probs = mx.softmax(logits, axis=-1)
        mx.eval(probs)

        execute_prob = float(probs[0, 1].item())
        should_execute = execute_prob >= self.threshold

        return should_execute, execute_prob

    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
    ) -> GenerationResult:
        """Generate with compiler expert integration."""
        tokens = self.tokenizer.encode(prompt)
        input_ids = mx.array([tokens])

        generated = []
        compiler_triggered = False
        code_block = None
        execution_result = None
        confidence = None

        for step in range(max_tokens):
            # Forward pass
            output = self.model(input_ids)
            logits = output.logits if hasattr(output, "logits") else output

            # Get next token
            next_token = mx.argmax(logits[0, -1, :])
            token_id = int(next_token.item())

            if token_id == self.tokenizer.eos_token_id:
                break

            generated.append(token_id)

            # Check if we just completed a code block
            current_text = self.tokenizer.decode(tokens + generated)
            block_end = self._find_code_block_end(current_text)

            if block_end is not None and not compiler_triggered:
                # Get current position
                current_pos = len(tokens) + len(generated) - 1

                # Classify
                should_execute, conf = self._classify_hidden(input_ids, current_pos)
                confidence = conf

                if should_execute:
                    # Extract code block
                    code_block = self._extract_last_code_block(current_text)

                    if code_block and self.compiler.can_handle(
                        f"```python\n{code_block}\n```"
                    ):
                        # Execute
                        exec_result = self.compiler.execute(
                            f"```python\n{code_block}\n```"
                        )

                        if exec_result:
                            compiler_triggered = True
                            execution_result = exec_result

                            # Inject result into generation
                            injection = f"\n\n{exec_result}\n"
                            result_tokens = self.tokenizer.encode(injection)
                            # Skip BOS token if present
                            result_tokens = [
                                t
                                for t in result_tokens
                                if t != self.tokenizer.bos_token_id
                            ]

                            generated.extend(result_tokens)
                            input_ids = mx.array([tokens + generated])
                            continue

            # Normal update
            input_ids = mx.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

            # Stop on double newline (end of response)
            current_text = self.tokenizer.decode(generated[-10:] if len(generated) > 10 else generated)
            if "\n\n\n" in current_text:
                break

        full_output = self.tokenizer.decode(generated)

        return GenerationResult(
            prompt=prompt,
            full_output=full_output,
            compiler_triggered=compiler_triggered,
            code_block=code_block,
            execution_result=execution_result,
            confidence=confidence,
        )

    def _extract_last_code_block(self, text: str) -> str | None:
        """Extract the last code block content."""
        pattern = r"```(?:python|py)?\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[-1].strip()
        return None


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================


def main(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    print("=" * 70)
    print("COMPILER VIRTUAL EXPERT - Generation Integration")
    print("=" * 70)

    # Load model
    print(f"\n1. Loading model: {model_name}...")
    from chuk_lazarus.models_v2.loader import load_model

    loaded = load_model(model_name)
    model = loaded.model
    tokenizer = loaded.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mx.eval(model.parameters())
    print("   Model loaded.")

    hidden_dim = get_hidden_dim(model)
    print(f"   Hidden dim: {hidden_dim}")

    # Train classifier
    print("\n2. Training code execution classifier...")
    classifier = train_code_classifier(model, tokenizer, hidden_dim, layer=10)

    # Create generator
    print("\n3. Creating generator with compiler expert...")
    generator = GeneratorWithCompiler(
        model, tokenizer, classifier, layer=10, confidence_threshold=0.7
    )

    # Test prompts
    test_prompts = [
        "Write a Python function to add two numbers and test it:\n```python\n",
        "Here's a simple factorial function:\n```python\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\nprint(factorial(5))\n```",
        "Let me calculate the sum of 1 to 10:\n```python\nprint(sum(range(1, 11)))\n```",
    ]

    print("\n4. Testing generation with compiler expert...")
    print("-" * 70)

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt[:60]}...")
        result = generator.generate(prompt, max_tokens=100)

        print(f"Output: {result.full_output[:100]}...")
        print(f"Compiler triggered: {result.compiler_triggered}")
        if result.execution_result:
            print(f"Execution result: {result.execution_result}")
        print(f"Confidence: {result.confidence}")
        print("-" * 70)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        """
The compiler virtual expert:
1. Detects completed code blocks via learned classifier
2. Executes code in sandboxed environment
3. Injects results back into generation context
4. Model can continue with feedback

This enables:
- Immediate feedback on generated code
- Self-correction when errors occur
- Verified outputs (syntax + runtime)
"""
    )


if __name__ == "__main__":
    import sys

    model_name = (
        sys.argv[1] if len(sys.argv) > 1 else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    main(model_name)
