"""
IR Sequence-to-Sequence Training

Trains an autoregressive decoder to emit IR opcode sequences from NL input.

Architecture:
  NL Input → LLM Encoder → Hidden State → IR Decoder → IR Opcode Sequence

Unlike the classifier approach, this generates opcodes token-by-token,
enabling potential generalization to novel programs.
"""

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ir_emission.shared import IROpcode
from programs import ALL_PROGRAMS as PROGRAMS, compile_program, WASMRuntime


# =============================================================================
# MODEL
# =============================================================================

@dataclass
class DecoderConfig:
    """Configuration for IR sequence decoder."""
    hidden_dim: int = 2048       # LLM hidden dimension
    embed_dim: int = 256         # Decoder embedding dimension
    num_heads: int = 4           # Attention heads
    num_layers: int = 3          # Decoder layers
    max_seq_len: int = 64        # Maximum IR sequence length
    vocab_size: int = 64         # Number of IR opcodes
    dropout: float = 0.1


class IRSequenceDecoder(nn.Module):
    """
    Autoregressive decoder for IR opcode sequences.

    Takes LLM hidden state as conditioning, generates opcodes one at a time.
    """

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config

        # Project LLM hidden state to decoder dimension
        self.hidden_proj = nn.Linear(config.hidden_dim, config.embed_dim)

        # Opcode embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)

        # Position embeddings
        self.pos_embed = nn.Embedding(config.max_seq_len, config.embed_dim)

        # Transformer decoder layers
        self.layers = [
            nn.TransformerEncoderLayer(
                dims=config.embed_dim,
                num_heads=config.num_heads,
                mlp_dims=config.embed_dim * 4,
            )
            for _ in range(config.num_layers)
        ]

        # Output projection
        self.output_proj = nn.Linear(config.embed_dim, config.vocab_size)

        self.dropout = nn.Dropout(config.dropout)

    def __call__(
        self,
        hidden_state: mx.array,
        target_seq: mx.array | None = None,
    ) -> mx.array:
        """
        Forward pass with teacher forcing.

        Args:
            hidden_state: LLM hidden state (batch, hidden_dim)
            target_seq: Target IR sequence for teacher forcing (batch, seq_len)

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size = hidden_state.shape[0]

        # Project hidden state as first token representation
        h_proj = self.hidden_proj(hidden_state)  # (batch, embed_dim)

        if target_seq is not None:
            # Teacher forcing mode
            seq_len = target_seq.shape[1]

            # Get token embeddings for target (shifted right)
            # Input: [h_proj, tok_0, tok_1, ..., tok_{n-1}]
            # Target: [tok_0, tok_1, ..., tok_n]
            tok_embeds = self.token_embed(target_seq[:, :-1])  # (batch, seq-1, embed)

            # Combine: first position is projected hidden state
            h = mx.concatenate([h_proj[:, None, :], tok_embeds], axis=1)  # (batch, seq, embed)

            # Add position embeddings
            positions = mx.arange(seq_len)
            pos_embeds = self.pos_embed(positions)  # (seq, embed)
            h = h + pos_embeds

            h = self.dropout(h)

            # Causal attention mask
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)

            # Transform through layers
            for layer in self.layers:
                h = layer(h, mask=mask)

            # Project to vocabulary
            logits = self.output_proj(h)  # (batch, seq, vocab)

        else:
            # Inference mode - not used during training
            raise NotImplementedError("Use generate() for inference")

        return logits

    def generate(
        self,
        hidden_state: mx.array,
        max_length: int | None = None,
        temperature: float = 0.0,
    ) -> list[int]:
        """
        Generate IR sequence autoregressively.

        Args:
            hidden_state: LLM hidden state (1, hidden_dim)
            max_length: Maximum sequence length
            temperature: Sampling temperature (0 = greedy)

        Returns:
            List of opcode indices
        """
        max_len = max_length or self.config.max_seq_len

        # Start with projected hidden state
        h_proj = self.hidden_proj(hidden_state)  # (1, embed_dim)

        generated = [int(IROpcode.START)]
        h_seq = h_proj[:, None, :]  # (1, 1, embed)

        for t in range(max_len - 1):
            # Current sequence length
            seq_len = h_seq.shape[1]

            # Add position embeddings
            positions = mx.arange(seq_len)
            pos_embeds = self.pos_embed(positions)
            h_with_pos = h_seq + pos_embeds

            # Create causal mask
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)

            # Transform with mask
            h_t = h_with_pos
            for layer in self.layers:
                h_t = layer(h_t, mask=mask)

            # Get logits for last position
            logits = self.output_proj(h_t[:, -1, :])  # (1, vocab)

            # Sample or greedy
            if temperature == 0:
                next_idx = int(mx.argmax(logits, axis=-1).item())
            else:
                probs = mx.softmax(logits / temperature, axis=-1)
                next_idx = int(mx.random.categorical(mx.log(probs)).item())

            generated.append(next_idx)

            # Stop at END token
            if next_idx == int(IROpcode.END):
                break

            # Add new token embedding to sequence
            next_embed = self.token_embed(mx.array([[next_idx]]))  # (1, 1, embed)
            h_seq = mx.concatenate([h_seq, next_embed], axis=1)

        return generated


# =============================================================================
# DATA
# =============================================================================

def load_dataset(path: Path) -> list[dict]:
    """Load dataset from JSON."""
    with open(path) as f:
        data = json.load(f)
    return data["examples"]


def prepare_sequence(opcodes: list[int], max_len: int) -> mx.array:
    """Prepare opcode sequence with padding."""
    # Ensure START and END tokens
    if opcodes[0] != IROpcode.START:
        opcodes = [IROpcode.START] + opcodes
    if opcodes[-1] != IROpcode.END:
        opcodes = opcodes + [IROpcode.END]

    # Truncate or pad
    if len(opcodes) > max_len:
        opcodes = opcodes[:max_len-1] + [IROpcode.END]
    else:
        opcodes = opcodes + [IROpcode.PAD] * (max_len - len(opcodes))

    return mx.array(opcodes)


def extract_operands(text: str) -> list[int]:
    """Extract numeric operands from NL text."""
    numbers = re.findall(r'\b(\d+)\b', text)
    return [int(n) for n in numbers]


# =============================================================================
# TRAINING
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 50
    hidden_layer: int = 12
    eval_every: int = 10
    max_seq_len: int = 64


def get_hidden_state(model, tokenizer, text: str, layer: int) -> mx.array:
    """Get hidden state from model at specified layer."""
    tokens = tokenizer.encode(text)
    input_ids = mx.array([tokens])

    hidden = model.model.embed_tokens(input_ids)

    for i, block in enumerate(model.model.layers[:layer]):
        hidden = block(hidden, mask=None, cache=None)
        if hasattr(hidden, 'hidden_states'):
            hidden = hidden.hidden_states
        elif isinstance(hidden, tuple):
            hidden = hidden[0]

    return hidden[0, -1, :]


def train_epoch(
    decoder: IRSequenceDecoder,
    optimizer: optim.Optimizer,
    model,
    tokenizer,
    examples: list[dict],
    config: TrainingConfig,
) -> float:
    """Train for one epoch."""
    import random
    random.shuffle(examples)

    total_loss = 0
    num_batches = 0

    def loss_fn(decoder, hidden_states, target_seqs):
        logits = decoder(hidden_states, target_seqs)
        # Cross-entropy loss, ignoring PAD tokens
        # logits: (batch, seq, vocab), target: (batch, seq)
        batch_size, seq_len, vocab_size = logits.shape

        # Flatten for loss computation
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = target_seqs.reshape(-1)

        # Compute loss only on non-PAD tokens
        mask = targets_flat != IROpcode.PAD
        loss = nn.losses.cross_entropy(logits_flat, targets_flat, reduction='none')
        loss = (loss * mask).sum() / mask.sum()

        return loss

    loss_and_grad = nn.value_and_grad(decoder, loss_fn)

    for i in range(0, len(examples), config.batch_size):
        batch = examples[i:i + config.batch_size]

        # Get hidden states and target sequences
        hidden_states = []
        target_seqs = []

        for ex in batch:
            hidden = get_hidden_state(model, tokenizer, ex["nl_input"], config.hidden_layer)
            hidden_states.append(hidden)

            target = prepare_sequence(ex["ir_opcodes"], config.max_seq_len)
            target_seqs.append(target)

        hidden_states = mx.stack(hidden_states)
        target_seqs = mx.stack(target_seqs)

        # Compute loss and gradients
        loss, grads = loss_and_grad(decoder, hidden_states, target_seqs)

        # Update
        optimizer.update(decoder, grads)
        mx.eval(decoder.parameters(), optimizer.state)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate_generation(
    decoder: IRSequenceDecoder,
    model,
    tokenizer,
    examples: list[dict],
    config: TrainingConfig,
) -> dict:
    """Evaluate sequence generation accuracy."""
    runtime = WASMRuntime(use_native=True)

    exact_match = 0
    exec_correct = 0
    total = 0

    details = []

    for ex in examples[:50]:  # Limit for speed
        total += 1

        # Get hidden state
        hidden = get_hidden_state(model, tokenizer, ex["nl_input"], config.hidden_layer)

        # Generate sequence
        generated = decoder.generate(hidden[None, :], temperature=0.0)

        # Check exact match (ignoring PAD)
        target = ex["ir_opcodes"]
        gen_trimmed = [x for x in generated if x not in (IROpcode.PAD,)]
        target_trimmed = [x for x in target if x not in (IROpcode.PAD,)]

        is_exact = (gen_trimmed == target_trimmed)
        if is_exact:
            exact_match += 1

        # Try execution
        exec_ok = False
        try:
            program = PROGRAMS.get(ex["program_name"])
            if program:
                operands = extract_operands(ex["nl_input"])
                # Try with generated sequence
                from programs import IRProgram, compile_program as compile_prog

                # Create temp program with generated opcodes
                gen_program = IRProgram(
                    name="generated",
                    description="",
                    opcodes=generated,
                    num_operands=len(operands),
                    test_cases=[],
                )

                wasm = compile_prog(gen_program, operands[:program.num_operands])
                result = runtime.execute(wasm)

                if result.success and result.result == ex["expected_result"]:
                    exec_ok = True
                    exec_correct += 1
        except Exception as e:
            pass

        details.append({
            "nl": ex["nl_input"][:50],
            "program": ex["program_name"],
            "exact_match": is_exact,
            "exec_correct": exec_ok,
            "generated_len": len(generated),
            "target_len": len(target),
        })

    return {
        "exact_match": exact_match / total if total > 0 else 0,
        "exec_accuracy": exec_correct / total if total > 0 else 0,
        "total": total,
        "details": details,
    }


def main():
    print("=" * 60)
    print("IR Program Synthesis - Sequence-to-Sequence Training")
    print("=" * 60)

    # Load model
    print("\n1. Loading base model...")
    from chuk_lazarus.models_v2.loader import load_model

    loaded = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = loaded.model
    tokenizer = loaded.tokenizer
    model.freeze()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mx.eval(model.parameters())
    print("   Model loaded.")

    # Load datasets
    print("\n2. Loading datasets...")
    results_dir = Path(__file__).parent / "results"
    train_examples = load_dataset(results_dir / "train_dataset.json")
    test_examples = load_dataset(results_dir / "test_dataset.json")
    print(f"   Train: {len(train_examples)} examples")
    print(f"   Test: {len(test_examples)} examples")

    # Get hidden dimension
    hidden_dim = model.model.layers[0].self_attn.q_proj.weight.shape[0]
    print(f"   Hidden dim: {hidden_dim}")

    # Find max sequence length needed
    max_opcode_len = max(len(ex["ir_opcodes"]) for ex in train_examples + test_examples)
    print(f"   Max opcode sequence: {max_opcode_len}")

    # Create decoder
    print("\n3. Creating sequence decoder...")
    config = TrainingConfig(max_seq_len=max_opcode_len + 4)  # Add buffer for START/END

    decoder_config = DecoderConfig(
        hidden_dim=hidden_dim,
        embed_dim=256,
        num_heads=4,
        num_layers=3,
        max_seq_len=config.max_seq_len,
        vocab_size=64,
    )
    decoder = IRSequenceDecoder(decoder_config)

    # Optimizer
    optimizer = optim.Adam(learning_rate=config.learning_rate)

    # Training loop
    print("\n4. Training...")
    print("-" * 60)

    best_train_acc = 0

    for epoch in range(config.epochs):
        start = time.perf_counter()

        # Train
        train_loss = train_epoch(
            decoder, optimizer, model, tokenizer,
            train_examples, config
        )

        elapsed = time.perf_counter() - start

        # Evaluate periodically
        if (epoch + 1) % config.eval_every == 0 or epoch == 0:
            train_eval = evaluate_generation(
                decoder, model, tokenizer,
                train_examples[:100], config
            )
            test_eval = evaluate_generation(
                decoder, model, tokenizer,
                test_examples, config
            )

            print(f"Epoch {epoch+1:2d} | Loss: {train_loss:.4f} | "
                  f"Train Exact: {train_eval['exact_match']:.1%} | "
                  f"Train Exec: {train_eval['exec_accuracy']:.1%} | "
                  f"Test Exact: {test_eval['exact_match']:.1%} | "
                  f"Test Exec: {test_eval['exec_accuracy']:.1%} | "
                  f"Time: {elapsed:.1f}s")

            if train_eval['exact_match'] > best_train_acc:
                best_train_acc = train_eval['exact_match']
        else:
            print(f"Epoch {epoch+1:2d} | Loss: {train_loss:.4f} | Time: {elapsed:.1f}s")

    print("-" * 60)

    # Final evaluation
    print("\n5. Final Evaluation...")

    print("\n   Training set:")
    train_final = evaluate_generation(
        decoder, model, tokenizer,
        train_examples, config
    )
    print(f"   Exact Match: {train_final['exact_match']:.1%}")
    print(f"   Execution: {train_final['exec_accuracy']:.1%}")

    print("\n   Test set (Collatz - held out):")
    test_final = evaluate_generation(
        decoder, model, tokenizer,
        test_examples, config
    )
    print(f"   Exact Match: {test_final['exact_match']:.1%}")
    print(f"   Execution: {test_final['exec_accuracy']:.1%}")

    # Show some examples
    print("\n   Sample generations:")
    for detail in test_final["details"][:5]:
        status = "exact" if detail["exact_match"] else ("exec" if detail["exec_correct"] else "fail")
        print(f"   [{status}] '{detail['nl']}...'")
        print(f"         gen_len={detail['generated_len']}, target_len={detail['target_len']}")

    # Save results
    print("\n6. Saving results...")
    results = {
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "hidden_layer": config.hidden_layer,
            "max_seq_len": config.max_seq_len,
            "decoder": {
                "embed_dim": decoder_config.embed_dim,
                "num_heads": decoder_config.num_heads,
                "num_layers": decoder_config.num_layers,
            }
        },
        "train": {
            "exact_match": train_final["exact_match"],
            "exec_accuracy": train_final["exec_accuracy"],
            "total": train_final["total"],
        },
        "test": {
            "exact_match": test_final["exact_match"],
            "exec_accuracy": test_final["exec_accuracy"],
            "total": test_final["total"],
            "details": test_final["details"],
        },
    }

    results_path = results_dir / "seq2seq_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   Saved to: {results_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
Sequence-to-Sequence Training Results:

Training Set:
  Exact Match: {train_final['exact_match']:.1%}
  Execution: {train_final['exec_accuracy']:.1%}

Test Set (Collatz - UNSEEN):
  Exact Match: {test_final['exact_match']:.1%}
  Execution: {test_final['exec_accuracy']:.1%}

Key Question: Can seq2seq generalize to Collatz?
Answer: {"YES" if test_final['exec_accuracy'] > 0.1 else "UNCERTAIN" if test_final['exact_match'] > 0 else "NO"}
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
