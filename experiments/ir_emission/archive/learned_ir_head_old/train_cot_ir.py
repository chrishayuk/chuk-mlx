"""
Train CoT→IR on small sample to prove the paradigm.

Training format:
  Input:  Q: <question>
  Output: <cot> [IR] <ir_code> [ANSWER] <number>

Reward: IR execution matches expected answer.
"""

import re
import json
import random
from pathlib import Path
from collections import Counter

import functools
print = functools.partial(print, flush=True)

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


# =============================================================================
# IR EXECUTION
# =============================================================================

def execute_ir(ir_code: str) -> float:
    """Execute IR and return final value."""
    env = {}
    steps = ir_code.strip().split('|')

    for step in steps:
        step = step.strip()
        if step == '[END]' or not step:
            continue

        if '=' not in step:
            continue

        var, expr = step.split('=', 1)
        var = var.strip()
        expr = expr.strip()

        # Substitute previous steps
        for prev_var, prev_val in env.items():
            expr = expr.replace(prev_var, str(prev_val))

        try:
            result = eval(expr)
            env[var] = result
        except:
            return None

    if not env:
        return None
    return list(env.values())[-1]


def extract_ir_from_output(text: str) -> str:
    """Extract IR code from model output."""
    match = re.search(r'\[IR\]\s*(.+?)\s*\[ANSWER\]', text)
    if match:
        return match.group(1).strip()

    # Try without [ANSWER]
    match = re.search(r'\[IR\]\s*(.+?)(?:\[|$)', text)
    if match:
        return match.group(1).strip()

    return None


def extract_answer_from_output(text: str) -> float:
    """Extract numeric answer from model output."""
    match = re.search(r'\[ANSWER\]\s*(-?[\d.]+)', text)
    if match:
        return float(match.group(1))
    return None


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cot_ir_data(n_train: int = 200, n_test: int = 50):
    """Load CoT→IR training data."""
    train_path = Path(__file__).parent / "cot_ir_data" / "train.json"
    test_path = Path(__file__).parent / "cot_ir_data" / "test.json"

    with open(train_path) as f:
        train_data = json.load(f)

    with open(test_path) as f:
        test_data = json.load(f)

    # Sample
    random.shuffle(train_data)
    random.shuffle(test_data)

    return train_data[:n_train], test_data[:n_test]


# =============================================================================
# TOKENIZER
# =============================================================================

class SimpleTokenizer:
    """Character-level tokenizer with special tokens."""

    def __init__(self):
        self.special_tokens = ['<PAD>', '<BOS>', '<EOS>', '<UNK>',
                               '[IR]', '[ANSWER]', '[END]', 'step1', 'step2',
                               'step3', 'step4', 'step5', 'step6', 'step7', 'step8']
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

    def build_vocab(self, texts: list[str]):
        """Build vocabulary from texts."""
        # Special tokens first
        for tok in self.special_tokens:
            self.char_to_idx[tok] = len(self.char_to_idx)
            self.idx_to_char[len(self.idx_to_char)] = tok

        # Characters
        chars = set()
        for text in texts:
            # Remove special tokens temporarily
            clean = text
            for tok in self.special_tokens:
                clean = clean.replace(tok, ' ')
            chars.update(clean)

        for char in sorted(chars):
            if char not in self.char_to_idx:
                self.char_to_idx[char] = len(self.char_to_idx)
                self.idx_to_char[len(self.idx_to_char)] = char

        self.vocab_size = len(self.char_to_idx)
        print(f"Vocabulary size: {self.vocab_size}")

    def encode(self, text: str) -> list[int]:
        """Encode text to indices."""
        tokens = []

        i = 0
        while i < len(text):
            # Check special tokens
            matched = False
            for tok in self.special_tokens:
                if text[i:].startswith(tok):
                    tokens.append(self.char_to_idx[tok])
                    i += len(tok)
                    matched = True
                    break

            if not matched:
                char = text[i]
                tokens.append(self.char_to_idx.get(char, self.char_to_idx['<UNK>']))
                i += 1

        return tokens

    def decode(self, indices: list[int]) -> str:
        """Decode indices to text."""
        chars = []
        for idx in indices:
            if idx in self.idx_to_char:
                tok = self.idx_to_char[idx]
                if tok not in ['<PAD>', '<BOS>', '<EOS>']:
                    chars.append(tok)
        return ''.join(chars)


# =============================================================================
# MODEL
# =============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiHeadAttention(dim, n_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x, mask=None):
        # Self-attention
        attn_out = self.attention(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


class SmallTransformer(nn.Module):
    """Small transformer for CoT→IR generation."""

    def __init__(self, vocab_size: int, dim: int = 256, n_layers: int = 4,
                 n_heads: int = 4, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)

        self.layers = [TransformerBlock(dim, n_heads, dropout) for _ in range(n_layers)]
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def __call__(self, x, mask=None):
        B, T = x.shape

        # Embeddings
        tok_emb = self.embed(x)
        pos = mx.arange(T)
        pos_emb = self.pos_embed(pos)

        x = self.dropout(tok_emb + pos_emb)

        # Causal mask
        if mask is None:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        logits = self.head(x)

        return logits

    def generate(self, prompt_ids: list[int], max_new_tokens: int = 200,
                 temperature: float = 0.7, eos_id: int = None) -> list[int]:
        """Generate tokens autoregressively."""
        ids = list(prompt_ids)

        for _ in range(max_new_tokens):
            # Truncate if too long
            context = ids[-self.max_len:]
            x = mx.array([context])

            logits = self(x)
            next_logits = logits[0, -1, :]

            # Temperature sampling
            if temperature > 0:
                probs = mx.softmax(next_logits / temperature)
                next_id = int(mx.random.categorical(mx.log(probs + 1e-8)))
            else:
                next_id = int(mx.argmax(next_logits))

            ids.append(next_id)

            if eos_id is not None and next_id == eos_id:
                break

        return ids


# =============================================================================
# TRAINING
# =============================================================================

def create_training_batch(examples: list, tokenizer: SimpleTokenizer,
                          max_len: int = 512) -> tuple:
    """Create a training batch."""
    inputs = []
    targets = []

    for ex in examples:
        # Format: Q: <question>\n<output>
        text = f"Q: {ex['input']}\n{ex['output']}<EOS>"
        ids = tokenizer.encode(text)

        if len(ids) > max_len:
            ids = ids[:max_len]

        # Pad
        pad_id = tokenizer.char_to_idx['<PAD>']
        padded = ids + [pad_id] * (max_len - len(ids))

        inputs.append(padded[:-1])
        targets.append(padded[1:])

    return mx.array(inputs), mx.array(targets)


def compute_loss(model, inputs, targets, pad_id: int):
    """Compute cross-entropy loss."""
    logits = model(inputs)

    # Reshape for cross-entropy
    B, T, V = logits.shape
    logits = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)

    # Mask padding
    mask = (targets_flat != pad_id).astype(mx.float32)

    # Cross-entropy
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    losses = -log_probs[mx.arange(len(targets_flat)), targets_flat]
    loss = (losses * mask).sum() / mask.sum()

    return loss


def train_epoch(model, optimizer, train_data, tokenizer, batch_size: int = 16):
    """Train for one epoch."""
    random.shuffle(train_data)
    total_loss = 0
    n_batches = 0
    pad_id = tokenizer.char_to_idx['<PAD>']

    def loss_fn(model, inputs, targets):
        return compute_loss(model, inputs, targets, pad_id)

    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i+batch_size]
        inputs, targets = create_training_batch(batch, tokenizer)

        loss, grads = nn.value_and_grad(model, loss_fn)(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        total_loss += float(loss)
        n_batches += 1

    return total_loss / n_batches


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, test_data, tokenizer, n_samples: int = 10):
    """Evaluate model on test data."""
    correct_ir = 0
    correct_answer = 0
    valid_ir = 0
    total = 0

    eos_id = tokenizer.char_to_idx['<EOS>']

    results = []

    for ex in test_data[:n_samples]:
        prompt = f"Q: {ex['input']}\n"
        prompt_ids = tokenizer.encode(prompt)

        # Generate
        output_ids = model.generate(prompt_ids, max_new_tokens=300,
                                    temperature=0.3, eos_id=eos_id)
        output_text = tokenizer.decode(output_ids)

        # Extract IR
        generated_ir = extract_ir_from_output(output_text)
        expected_ir = ex['ir']
        expected_answer = ex['answer']

        # Evaluate
        ir_result = None
        if generated_ir:
            valid_ir += 1
            ir_result = execute_ir(generated_ir)

            if ir_result is not None:
                if abs(ir_result - expected_answer) < 0.01:
                    correct_answer += 1

        results.append({
            'question': ex['input'][:60],
            'expected_answer': expected_answer,
            'expected_ir': expected_ir,
            'generated_ir': generated_ir,
            'ir_result': ir_result,
            'correct': ir_result is not None and abs(ir_result - expected_answer) < 0.01,
        })

        total += 1

    return {
        'total': total,
        'valid_ir': valid_ir,
        'correct_answer': correct_answer,
        'valid_ir_rate': valid_ir / total if total > 0 else 0,
        'accuracy': correct_answer / total if total > 0 else 0,
        'results': results,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    random.seed(42)
    mx.random.seed(42)

    print("=" * 70)
    print("  COT→IR TRAINING PROOF OF CONCEPT")
    print("  Training small transformer on GSM8K CoT→IR format")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    train_data, test_data = load_cot_ir_data(n_train=300, n_test=50)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # Build tokenizer
    print("\nBuilding tokenizer...")
    tokenizer = SimpleTokenizer()
    all_texts = [f"Q: {ex['input']}\n{ex['output']}" for ex in train_data + test_data]
    tokenizer.build_vocab(all_texts)

    # Create model
    print("\nCreating model...")
    model = SmallTransformer(
        vocab_size=tokenizer.vocab_size,
        dim=256,
        n_layers=4,
        n_heads=4,
        max_len=512,
        dropout=0.1
    )

    def count_params(params):
        total = 0
        for v in params.values():
            if isinstance(v, dict):
                total += count_params(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        total += count_params(item)
                    else:
                        total += item.size
            else:
                total += v.size
        return total

    n_params = count_params(model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = optim.AdamW(learning_rate=3e-4)

    # Training
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    n_epochs = 30
    best_acc = 0

    for epoch in range(n_epochs):
        loss = train_epoch(model, optimizer, train_data, tokenizer, batch_size=16)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1}/{n_epochs}, Loss: {loss:.4f}")

            # Quick eval
            eval_result = evaluate(model, test_data, tokenizer, n_samples=20)
            print(f"  Valid IR: {eval_result['valid_ir_rate']:.1%}")
            print(f"  Accuracy: {eval_result['accuracy']:.1%}")

            if eval_result['accuracy'] > best_acc:
                best_acc = eval_result['accuracy']

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    final_eval = evaluate(model, test_data, tokenizer, n_samples=len(test_data))

    print(f"\nTotal test examples: {final_eval['total']}")
    print(f"Valid IR generated:  {final_eval['valid_ir']} ({final_eval['valid_ir_rate']:.1%})")
    print(f"Correct answers:     {final_eval['correct_answer']} ({final_eval['accuracy']:.1%})")

    # Show examples
    print("\n" + "=" * 70)
    print("SAMPLE RESULTS")
    print("=" * 70)

    print("\nCorrect predictions:")
    for r in [x for x in final_eval['results'] if x['correct']][:5]:
        print(f"\n  Q: {r['question']}...")
        print(f"  Expected: {r['expected_ir']} → {r['expected_answer']}")
        print(f"  Generated: {r['generated_ir']} → {r['ir_result']}")

    print("\nIncorrect predictions:")
    for r in [x for x in final_eval['results'] if not x['correct']][:5]:
        print(f"\n  Q: {r['question']}...")
        print(f"  Expected: {r['expected_ir']} → {r['expected_answer']}")
        print(f"  Generated: {r['generated_ir']} → {r['ir_result']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
  Training samples: {len(train_data)}
  Test samples: {len(test_data)}
  Model parameters: {n_params:,}

  Results:
    Valid IR rate: {final_eval['valid_ir_rate']:.1%}
    Accuracy:      {final_eval['accuracy']:.1%}

  The model learns to:
    1. Generate natural language CoT reasoning
    2. Emit valid IR code after [IR] token
    3. Execute IR to get verifiable answer
""")


if __name__ == "__main__":
    main()
