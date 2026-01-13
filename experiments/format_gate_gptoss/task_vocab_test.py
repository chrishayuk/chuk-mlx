"""
Task-level vocab alignment test for GPT-OSS.

Tests if L13 projects onto TASK tokens (arithmetic, synonyms, positive, responses)
not operation tokens (multiply, add).

IMPORTANT: Applies layer normalization before vocab projection (standard logit lens).
"""

import logging
import json
from pathlib import Path
from datetime import datetime

import mlx.core as mx
import yaml

logger = logging.getLogger(__name__)


# Test prompts organized by expected task token
TEST_PROMPTS = {
    # Math - expect 'arithmetic', 'integer', 'sum', 'subtract', 'calculated'
    "math_calculator": [
        "45 * 45 = ",
        "100 + 37 = ",
        "100 - 37 = ",
        "144 / 12 = ",
        "0.35 * 6 = ",
        "2 ^ 10 = ",
    ],

    # Synonym task - expect 'synonyms'
    "synonym": [
        "A synonym for happy is",
        "A synonym for fast is",
        "A synonym for big is",
        "What is another word for angry?",
    ],

    # Antonym task - expect 'opposite'
    "antonym": [
        "The opposite of hot is",
        "The opposite of good is",
        "The opposite of fast is",
        "What is the antonym of happy?",
    ],

    # Sentiment - expect 'positive' or 'negative'
    "sentiment": [
        "This movie is great! Sentiment:",
        "I hate this product. Sentiment:",
        "The food was amazing! Sentiment:",
        "Terrible service. Sentiment:",
    ],

    # General/default - expect 'responses'
    "general": [
        "Capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "def add(a, b):",
        "x = 45 * 45",
    ],

    # Word problems (semantic math) - what does this get?
    "word_problem": [
        "Janet has 45 apples and buys 45 more. How many total?",
        "If you have 100 dollars and spend 37, how much remains?",
        "A store has 144 items split into 12 boxes. How many per box?",
    ],
}

# Expected task tokens based on prior experiments
# Note: Many tokens in LLM vocabs have a leading space - we check both variants
TASK_TOKENS = {
    "math_calculator": [" arithmetic", " integer", " sum", " subtract", " calculated", " decimal", " multiplied"],
    "synonym": [" synonyms", " synonym"],
    "antonym": [" opposite", " antonym", " antonyms"],
    "sentiment": [" positive", " negative", " sentiment", " classification"],
    "general": [" responses", " response", " answer", " answers", " answering"],
    "word_problem": [" arithmetic", " responses", " calculated", " math"],
}


def run_task_vocab_test():
    """Run task-level vocab alignment test."""
    from chuk_lazarus.models_v2.loader import load_model

    # Load model
    logger.info("Loading GPT-OSS 20B...")
    loaded = load_model("openai/gpt-oss-20b")
    model = loaded.model
    tokenizer = loaded.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mx.eval(model.parameters())
    num_layers = len(model.model.layers)
    logger.info(f"Model loaded: {num_layers} layers")

    # Get final layer norm (CRITICAL for proper logit lens)
    final_norm = None
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        final_norm = model.model.norm
        logger.info("Found final layer norm: model.model.norm")
    else:
        logger.warning("Could not find final layer norm - results may be incorrect!")

    # Get LM head for projection
    lm_head_module = model.lm_head
    logger.info(f"LM head type: {type(lm_head_module)}")

    def get_hidden_state(prompt: str, layer: int) -> mx.array:
        tokens = tokenizer(prompt, return_tensors="np")
        input_ids = mx.array(tokens["input_ids"])
        output = model(input_ids, output_hidden_states=True)
        return output.hidden_states[layer][0, -1, :]

    # Test at L13 (the reported vocab alignment layer)
    test_layer = 13
    results = {
        "model": "openai/gpt-oss-20b",
        "layer": test_layer,
        "tasks": {}
    }

    print("\n" + "=" * 80)
    print(f"TASK-LEVEL VOCAB ALIGNMENT TEST (Layer {test_layer})")
    print("=" * 80)

    for task_name, prompts in TEST_PROMPTS.items():
        logger.info(f"\n=== {task_name.upper()} ===")
        task_results = {"prompts": []}

        for prompt in prompts:
            # Get hidden state
            h = get_hidden_state(prompt, test_layer)

            # Apply layer norm BEFORE projection (critical for logit lens!)
            if final_norm is not None:
                h = final_norm(h)

            # Project to vocab using lm_head
            h_batched = h[None, None, :]  # Add batch and seq dims
            head_out = lm_head_module(h_batched)
            if hasattr(head_out, 'logits'):
                logits = head_out.logits[0, 0, :]
            else:
                logits = head_out[0, 0, :]
            probs = mx.softmax(logits)

            # Top 10 tokens
            top_indices = mx.argsort(logits)[-10:][::-1]
            top_tokens = []
            for idx in top_indices:
                token_id = int(idx.item())
                token_str = tokenizer.decode([token_id]).strip()
                prob = probs[token_id].item()
                top_tokens.append({"token": token_str, "prob": prob})

            # Check expected tokens
            expected = TASK_TOKENS.get(task_name, [])
            expected_ranks = {}
            for exp_token in expected:
                token_ids = tokenizer.encode(exp_token, add_special_tokens=False)
                if token_ids:
                    token_id = token_ids[0]
                    rank = int((logits >= logits[token_id]).sum().item())
                    prob = probs[token_id].item()
                    expected_ranks[exp_token] = {"rank": rank, "prob": prob}

            prompt_result = {
                "prompt": prompt,
                "top_5": top_tokens[:5],
                "expected_tokens": expected_ranks,
            }
            task_results["prompts"].append(prompt_result)

            # Print
            top_token = top_tokens[0]
            print(f"\n'{prompt[:40]}...' " if len(prompt) > 40 else f"\n'{prompt}'")
            print(f"  Top: '{top_token['token']}' ({top_token['prob']:.1%})")
            print(f"  Top 5: {[t['token'] for t in top_tokens[:5]]}")

            # Show expected token ranks
            for exp, data in expected_ranks.items():
                marker = "✓" if data["rank"] <= 10 else ""
                print(f"  '{exp}': rank {data['rank']}, prob {data['prob']:.2%} {marker}")

        results["tasks"][task_name] = task_results

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"task_vocab_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY BY TASK")
    print("=" * 80)

    for task_name, task_data in results["tasks"].items():
        print(f"\n{task_name.upper()}:")
        # Collect all top tokens across prompts
        all_top = []
        for p in task_data["prompts"]:
            all_top.append(p["top_5"][0]["token"])

        # Most common top token
        from collections import Counter
        most_common = Counter(all_top).most_common(3)
        print(f"  Most common top tokens: {most_common}")

        # Average rank of expected tokens
        for exp_token in TASK_TOKENS.get(task_name, [])[:3]:
            ranks = []
            for p in task_data["prompts"]:
                if exp_token in p["expected_tokens"]:
                    ranks.append(p["expected_tokens"][exp_token]["rank"])
            if ranks:
                avg_rank = sum(ranks) / len(ranks)
                print(f"  '{exp_token}' avg rank: {avg_rank:.0f}")

    print("\n" + "=" * 80)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_task_vocab_test()
