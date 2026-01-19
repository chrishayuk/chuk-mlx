#!/usr/bin/env python3
"""
Interference Co-Activation Matrix

Research question: Which facts suppress which others?

For each multiplication fact, capture which "wrong" answers appear
in the top-k predictions. Build an N×N matrix showing interference patterns.

This reveals:
- Mutual suppression (A↔B compete)
- Asymmetric interference (A suppresses B but not vice versa)
- Cluster interference (whole groups compete)
"""

import json
from collections import defaultdict
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class InterferenceMatrix:
    """Co-activation matrix for fact interference."""

    facts: list[str]  # fact queries
    answers: list[str]  # correct answers
    matrix: np.ndarray  # [i,j] = prob of answer_j when querying fact_i
    top_interferers: dict[str, list[tuple[str, float]]]  # per fact, top wrong answers


def build_interference_matrix(
    model_id: str = "openai/gpt-oss-20b",
    layer: int = 22,
    top_k: int = 30,
) -> InterferenceMatrix:
    """Build interference matrix for multiplication facts."""
    import json

    from chuk_lazarus.inference.loader import DType, HFLoader
    from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info

    print(f"Loading model: {model_id}")

    result = HFLoader.download(model_id)
    model_path = result.model_path

    config_path = model_path / "config.json"
    with open(config_path) as f:
        config_data = json.load(f)

    family_type = detect_model_family(config_data)
    family_info = get_family_info(family_type)
    config = family_info.config_class.from_hf_config(config_data)
    model = family_info.model_class(config)

    HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
    tokenizer = HFLoader.load_tokenizer(model_path)

    # Generate all single-digit multiplication facts
    facts = []
    answers = []
    answer_to_idx = {}

    for a in range(2, 10):
        for b in range(2, 10):
            query = f"{a}*{b}="
            answer = str(a * b)
            facts.append(query)
            answers.append(answer)
            if answer not in answer_to_idx:
                answer_to_idx[answer] = len(answer_to_idx)

    n_facts = len(facts)
    n_answers = len(answer_to_idx)

    print(f"Analyzing {n_facts} facts, {n_answers} unique answers")
    print(f"Target layer: {layer}")

    # Build matrix
    matrix = np.zeros((n_facts, n_answers))

    def get_layers():
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return list(model.model.layers)
        return list(model.layers)

    def get_embed():
        if hasattr(model, "model"):
            return model.model.embed_tokens
        return model.embed_tokens

    def get_norm():
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            return model.model.norm
        if hasattr(model, "norm"):
            return model.norm
        return None

    def get_lm_head():
        if hasattr(model, "lm_head"):
            return model.lm_head
        return None

    def get_scale():
        return getattr(config, "embedding_scale", None)

    def get_predictions(prompt: str) -> dict[str, float]:
        """Get probabilities for all answer tokens."""
        input_ids = mx.array(tokenizer.encode(prompt))[None, :]
        layers = get_layers()
        embed = get_embed()
        norm = get_norm()
        lm_head = get_lm_head()
        scale = get_scale()

        h = embed(input_ids)
        if scale:
            h = h * scale

        seq_len = input_ids.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

        for idx, lyr in enumerate(layers):
            try:
                out = lyr(h, mask=mask)
            except TypeError:
                out = lyr(h)
            h = (
                out.hidden_states
                if hasattr(out, "hidden_states")
                else (out[0] if isinstance(out, tuple) else out)
            )
            if idx == layer:
                break

        if norm is not None:
            h = norm(h)
        if lm_head is not None:
            outputs = lm_head(h)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
        else:
            logits = h @ embed.weight.T

        probs = mx.softmax(logits[0, -1, :], axis=-1)

        # Get probabilities for known answers
        result = {}
        for answer in answer_to_idx:
            token_ids = tokenizer.encode(answer)
            if len(token_ids) == 1:
                result[answer] = float(probs[token_ids[0]])
            else:
                # Multi-token: use first token probability as approximation
                result[answer] = float(probs[token_ids[0]])

        return result

    top_interferers = {}

    for i, (fact, correct) in enumerate(zip(facts, answers)):
        if (i + 1) % 10 == 0:
            print(f"  Processing {i + 1}/{n_facts}...")

        probs = get_predictions(fact)

        for answer, prob in probs.items():
            j = answer_to_idx[answer]
            matrix[i, j] = prob

        # Find top interferers (wrong answers with high probability)
        wrong_answers = [(ans, prob) for ans, prob in probs.items() if ans != correct]
        wrong_answers.sort(key=lambda x: -x[1])
        top_interferers[fact] = wrong_answers[:5]

    # Convert to more interpretable format
    answer_list = [""] * n_answers
    for answer, idx in answer_to_idx.items():
        answer_list[idx] = answer

    return InterferenceMatrix(
        facts=facts,
        answers=answer_list,
        matrix=matrix,
        top_interferers=top_interferers,
    )


def analyze_interference(im: InterferenceMatrix):
    """Analyze interference patterns."""
    print("\n" + "=" * 70)
    print("INTERFERENCE ANALYSIS")
    print("=" * 70)

    # 1. Top attractors (answers that appear most often as wrong answers)
    print("\n1. ATTRACTOR NODES (most frequently interfering answers)")

    # Sum probability mass each answer gets across all facts
    answer_totals = defaultdict(float)
    answer_counts = defaultdict(int)

    for i, fact in enumerate(im.facts):
        correct = im.answers[np.argmax(im.matrix[i])]

        for j, answer in enumerate(im.answers):
            if answer and answer != correct:
                prob = im.matrix[i, j]
                if prob > 0.001:  # threshold
                    answer_totals[answer] += prob
                    answer_counts[answer] += 1

    attractors = sorted(answer_totals.items(), key=lambda x: -x[1])[:15]
    for answer, total_prob in attractors:
        count = answer_counts[answer]
        print(f"  '{answer}': total_prob={total_prob:.3f}, appears in {count} facts")

    # 2. Mutual suppression pairs
    print("\n2. MUTUAL SUPPRESSION PAIRS")

    # Find pairs where A appears when querying B and vice versa
    mutual_pairs = []

    fact_to_answer = {fact: ans for fact, ans in zip(im.facts, im.answers) if ans}

    # Build answer to facts mapping
    answer_to_facts = defaultdict(list)
    for fact, ans in zip(im.facts, im.answers):
        answer_to_facts[ans].append(fact)

    for i, fact_i in enumerate(im.facts):
        ans_i = im.answers[np.argmax(im.matrix[i])] if im.matrix[i].max() > 0 else None
        if not ans_i:
            continue

        for j, fact_j in enumerate(im.facts):
            if j <= i:
                continue

            ans_j = im.answers[np.argmax(im.matrix[j])] if im.matrix[j].max() > 0 else None
            if not ans_j:
                continue

            # Check if ans_i appears when querying fact_j
            idx_i = im.answers.index(ans_i) if ans_i in im.answers else -1
            idx_j = im.answers.index(ans_j) if ans_j in im.answers else -1

            if idx_i >= 0 and idx_j >= 0:
                prob_i_in_j = im.matrix[j, idx_i]
                prob_j_in_i = im.matrix[i, idx_j]

                if prob_i_in_j > 0.01 and prob_j_in_i > 0.01:
                    mutual_pairs.append((fact_i, ans_i, fact_j, ans_j, prob_i_in_j, prob_j_in_i))

    mutual_pairs.sort(key=lambda x: -(x[4] + x[5]))
    print(f"  Found {len(mutual_pairs)} mutual suppression pairs")
    for fact_i, ans_i, fact_j, ans_j, p_i, p_j in mutual_pairs[:10]:
        print(f"    {fact_i}({ans_i}) ↔ {fact_j}({ans_j}): {p_i:.3f}/{p_j:.3f}")

    # 3. Cluster interference (do certain rows interfere?)
    print("\n3. ROW-BASED INTERFERENCE")

    # Group facts by first operand
    row_groups = defaultdict(list)
    for fact in im.facts:
        a = int(fact[0])
        row_groups[a].append(fact)

    for row, row_facts in sorted(row_groups.items()):
        # Average interference within this row
        within_interference = []
        for fact in row_facts:
            correct = fact.split("=")[0].split("*")
            correct_ans = str(int(correct[0]) * int(correct[1]))

            for other in row_facts:
                if other == fact:
                    continue
                other_ans = str(int(other[0]) * int(other[2]))

                # Does other_ans appear when querying fact?
                if other_ans in im.answers:
                    idx_f = im.facts.index(fact)
                    idx_a = im.answers.index(other_ans)
                    within_interference.append(im.matrix[idx_f, idx_a])

        avg_within = np.mean(within_interference) if within_interference else 0
        print(f"  Row {row}x: avg within-row interference = {avg_within:.4f}")

    # 4. Hardest facts (most total interference)
    print("\n4. MOST INTERFERED-WITH FACTS")

    fact_interference = []
    for i, fact in enumerate(im.facts):
        correct_idx = im.facts.index(fact)
        correct_prob = im.matrix[i, correct_idx] if correct_idx < im.matrix.shape[1] else 0

        # Total wrong probability
        total_wrong = sum(
            im.matrix[i, j]
            for j in range(len(im.answers))
            if im.answers[j] != im.answers[np.argmax(im.matrix[i])]
        )
        fact_interference.append((fact, correct_prob, total_wrong))

    fact_interference.sort(key=lambda x: -x[2])
    for fact, correct_p, wrong_p in fact_interference[:10]:
        interferers = im.top_interferers.get(fact, [])[:3]
        interf_str = ", ".join(f"{a}({p:.3f})" for a, p in interferers)
        print(f"  {fact}: correct={correct_p:.3f}, wrong={wrong_p:.3f} [{interf_str}]")


def main():
    im = build_interference_matrix(
        model_id="openai/gpt-oss-20b",
        layer=22,
        top_k=30,
    )

    analyze_interference(im)

    # Save matrix
    np.savez(
        "interference_matrix.npz",
        facts=im.facts,
        answers=im.answers,
        matrix=im.matrix,
    )
    print("\nMatrix saved to interference_matrix.npz")

    # Also save as JSON for easy viewing
    output = {
        "facts": im.facts,
        "answers": im.answers,
        "top_interferers": im.top_interferers,
    }
    with open("interference_analysis.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Analysis saved to interference_analysis.json")


if __name__ == "__main__":
    main()
