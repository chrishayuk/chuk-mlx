#!/usr/bin/env python3
"""
Example 2: Sequence Packing with Segment-Aware Attention

Demonstrates:
- Packing multiple short sequences into longer ones
- Creating segment-aware attention masks
- Comparing packing algorithms (first-fit, best-fit, greedy)
- Measuring packing efficiency gains

Run:
    python examples/batching/02_sequence_packing.py
"""

import numpy as np

from chuk_lazarus.data import (
    PackedSequence,
    PackingConfig,
    PackingMode,
    SequenceToPack,
    compute_packing_metrics,
    create_segment_attention_mask,
    pack_sequences,
)


def create_sequences(num_sequences: int = 20) -> list[SequenceToPack]:
    """Create synthetic sequences with varying lengths."""
    import random

    random.seed(42)
    sequences = []

    for i in range(num_sequences):
        # Simulate realistic distribution: mostly short sequences
        if random.random() < 0.7:
            length = random.randint(50, 150)
        else:
            length = random.randint(150, 300)

        # Create token IDs (for demo, just use sequential numbers)
        input_ids = list(range(1000 + i * 1000, 1000 + i * 1000 + length))

        # SFT-style loss mask: first 20% is prompt
        prompt_len = int(length * 0.2)
        loss_mask = [0] * prompt_len + [1] * (length - prompt_len)

        seq = SequenceToPack(
            sample_id=f"seq_{i:03d}",
            input_ids=input_ids,
            loss_mask=loss_mask,
        )
        sequences.append(seq)

    return sequences


def visualize_attention_mask(mask: np.ndarray, segment_ids: tuple, max_display: int = 20):
    """Visualize attention mask with segment boundaries."""
    size = min(len(segment_ids), max_display)
    print(f"\n   Attention mask (first {size}x{size} positions):")
    print("   " + "".join(f"{i % 10}" for i in range(size)))

    for i in range(size):
        row = ""
        for j in range(size):
            if mask[i, j] > 0:
                row += "█"
            else:
                row += "·"
        seg_marker = f" seg={segment_ids[i]}"
        print(f"   {row}{seg_marker}")


def visualize_packed_sequence(packed: PackedSequence, max_tokens: int = 40):
    """Show packed sequence structure."""
    print(f"\n   Packed sequence structure (first {max_tokens} tokens):")

    # Show segment boundaries
    n = min(len(packed.segment_ids), max_tokens)

    # Segment IDs row
    seg_row = "".join(str(s % 10) for s in packed.segment_ids[:n])
    print(f"   Segments: {seg_row}")

    # Loss mask row
    loss_row = "".join("█" if m else "·" for m in packed.loss_mask[:n])
    print(f"   Loss:     {loss_row}")

    # Padding indicator
    pad_start = packed.total_tokens
    if pad_start < n:
        pad_row = "·" * pad_start + "P" * (n - pad_start)
        print(f"   Padding:  {pad_row}")


def main():
    print("=" * 60)
    print("Sequence Packing Demo")
    print("=" * 60)

    # 1. Create sequences
    print("\n1. Creating sequences...")
    sequences = create_sequences(num_sequences=20)
    lengths = [s.length for s in sequences]
    print(f"   Created {len(sequences)} sequences")
    print(f"   Length range: {min(lengths)} - {max(lengths)}")
    print(f"   Total tokens: {sum(lengths)}")

    # 2. Configure packing
    print("\n2. Configuring packing...")
    max_length = 512
    config = PackingConfig(
        mode=PackingMode.FIRST_FIT,
        max_length=max_length,
        pad_to_max=True,
    )
    print(f"   Max packed length: {max_length}")
    print(f"   Packing mode: {config.mode.value}")
    print(f"   Pad to max: {config.pad_to_max}")

    # 3. Pack sequences
    print("\n3. Packing sequences (FIRST_FIT)...")
    packed = pack_sequences(sequences, config, pad_token_id=0)
    print(f"   Packed {len(sequences)} sequences into {len(packed)} packed sequences")

    # 4. Show packed sequence details
    print("\n4. Packed sequence details:")
    for i, p in enumerate(packed[:3]):  # Show first 3
        print(f"\n   Pack {i}:")
        print(f"     Samples: {list(p.sample_ids)}")
        print(f"     Num segments: {p.num_segments}")
        print(f"     Total tokens: {p.total_tokens}")
        print(f"     Padding: {p.padding_tokens}")
        print(f"     Efficiency: {p.efficiency:.1%}")
        print(f"     Loss tokens: {p.num_loss_tokens}")

        visualize_packed_sequence(p)

    # 5. Compute metrics
    print("\n5. Packing metrics:")
    metrics = compute_packing_metrics(packed)
    print(f"   Original samples: {metrics.num_original_samples}")
    print(f"   Packed sequences: {metrics.num_packed_sequences}")
    print(f"   Packing ratio: {metrics.packing_ratio:.2f}x")
    print(f"   Efficiency: {metrics.efficiency:.1%}")
    print(f"   Loss efficiency: {metrics.loss_efficiency:.1%}")

    # Compare with no packing
    no_pack_tokens = len(sequences) * max_length
    pack_tokens = len(packed) * max_length
    print(
        f"\n   Without packing: {len(sequences)} batches × {max_length} = {no_pack_tokens} tokens"
    )
    print(f"   With packing:    {len(packed)} batches × {max_length} = {pack_tokens} tokens")
    print(f"   Token reduction: {(1 - pack_tokens / no_pack_tokens):.1%}")

    # 6. Create segment attention mask
    print("\n6. Creating segment-aware attention mask...")
    first_pack = packed[0]
    mask = create_segment_attention_mask(first_pack.segment_ids, use_mlx=False)
    print(f"   Mask shape: {mask.shape}")

    visualize_attention_mask(mask, first_pack.segment_ids)

    # Verify attention isolation
    print("\n   Verifying attention isolation:")
    num_segments = first_pack.num_segments
    for seg in range(min(num_segments, 3)):
        seg_positions = [i for i, s in enumerate(first_pack.segment_ids) if s == seg]
        if len(seg_positions) >= 2:
            # Check within-segment attention
            p1, p2 = seg_positions[0], seg_positions[-1]
            can_attend = mask[p2, p1] > 0
            print(f"   Segment {seg}: position {p2} can attend to {p1}? {can_attend}")

    # Check cross-segment blocking
    if num_segments >= 2:
        seg0_last = max(i for i, s in enumerate(first_pack.segment_ids) if s == 0)
        seg1_first = min(i for i, s in enumerate(first_pack.segment_ids) if s == 1)
        blocked = mask[seg1_first, seg0_last] == 0
        print(
            f"   Cross-segment: position {seg1_first} (seg 1) blocked from {seg0_last} (seg 0)? {blocked}"
        )

    # 7. Compare packing algorithms
    print("\n7. Comparing packing algorithms...")

    results = {}
    for mode in [PackingMode.FIRST_FIT, PackingMode.BEST_FIT, PackingMode.GREEDY]:
        config = PackingConfig(mode=mode, max_length=max_length, pad_to_max=True)
        packed_result = pack_sequences(sequences, config, pad_token_id=0)
        metrics = compute_packing_metrics(packed_result)
        results[mode] = {
            "num_packs": len(packed_result),
            "efficiency": metrics.efficiency,
            "packing_ratio": metrics.packing_ratio,
        }

    print(f"\n   {'Algorithm':<12} {'Packs':<8} {'Efficiency':<12} {'Ratio':<8}")
    print("   " + "-" * 40)
    for mode, data in results.items():
        print(
            f"   {mode.value:<12} {data['num_packs']:<8} "
            f"{data['efficiency']:.1%}        {data['packing_ratio']:.2f}x"
        )

    # 8. Packing with separators
    print("\n8. Packing with separator tokens...")
    config_sep = PackingConfig(
        mode=PackingMode.FIRST_FIT,
        max_length=max_length,
        pad_to_max=True,
        add_separator=True,
        separator_token_id=2,  # e.g., </s> token
    )
    packed_sep = pack_sequences(sequences, config_sep, pad_token_id=0)
    print(f"   With separators: {len(packed_sep)} packed sequences")

    # Show separator in first pack
    if packed_sep:
        first = packed_sep[0]
        print("\n   First pack with separators:")
        print(f"     Samples: {list(first.sample_ids)}")

        # Find separator positions
        sep_positions = [i for i, t in enumerate(first.input_ids) if t == 2]
        print(f"     Separator positions: {sep_positions[:5]}...")  # First 5

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
