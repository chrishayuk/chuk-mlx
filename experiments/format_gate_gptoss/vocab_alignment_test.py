"""
Quick vocab alignment test for GPT-OSS.

Tests if hidden states at L13 project onto operation tokens in vocab space.

IMPORTANT: Applies layer normalization before vocab projection (standard logit lens).
"""

import logging
import json
from pathlib import Path
from datetime import datetime

import mlx.core as mx
import yaml

logger = logging.getLogger(__name__)


def run_vocab_alignment_test():
    """Run just the vocab alignment test."""
    from chuk_lazarus.models_v2.loader import load_model

    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load model
    logger.info(f"Loading model: {config['model']}")
    loaded = load_model(config["model"])
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

    # Test layers
    test_layers = [12, 13, 14]  # Around the 50-54% depth mark
    vocab_tokens = config["parameters"]["vocab_alignment_tokens"]
    train_data = config["parameters"]["train_data"]

    def get_hidden_state(prompt: str, layer: int) -> mx.array:
        tokens = tokenizer(prompt, return_tensors="np")
        input_ids = mx.array(tokens["input_ids"])
        output = model(input_ids, output_hidden_states=True)
        return output.hidden_states[layer][0, -1, :]

    def infer_task(prompt: str) -> str:
        if "*" in prompt or "times" in prompt.lower() or "multiply" in prompt.lower():
            return "multiplication"
        elif "+" in prompt or "add" in prompt.lower() or "more" in prompt.lower():
            return "addition"
        elif "-" in prompt or "subtract" in prompt.lower() or "gave away" in prompt.lower():
            return "subtraction"
        elif "/" in prompt or "divide" in prompt.lower() or "split" in prompt.lower():
            return "division"
        return "unknown"

    results = {"model": config["model"], "layers": {}}

    for layer in test_layers:
        if layer >= num_layers:
            continue

        logger.info(f"\n=== Layer {layer} ({layer/num_layers*100:.0f}% depth) ===")
        layer_results = {}

        for task, expected_tokens in vocab_tokens.items():
            # Get prompts for this task
            task_prompts = [
                item["prompt"] for item in train_data
                if infer_task(item["prompt"]) == task
            ]

            if not task_prompts:
                continue

            # Get mean hidden state
            hiddens = [get_hidden_state(p, layer) for p in task_prompts]
            mean_hidden = mx.mean(mx.stack(hiddens), axis=0)

            # Apply layer norm BEFORE projection (critical for logit lens!)
            if final_norm is not None:
                mean_hidden = final_norm(mean_hidden)

            # Project to vocab using lm_head
            h_batched = mean_hidden[None, None, :]  # Add batch and seq dims
            head_out = lm_head_module(h_batched)
            if hasattr(head_out, 'logits'):
                logits = head_out.logits[0, 0, :]
            else:
                logits = head_out[0, 0, :]
            probs = mx.softmax(logits)

            # Top 10 tokens
            top_k = 10
            top_indices = mx.argsort(logits)[-top_k:][::-1]
            top_tokens = []
            for idx in top_indices:
                token_id = int(idx.item())
                token_str = tokenizer.decode([token_id]).strip()
                prob = probs[token_id].item()
                top_tokens.append((token_str, prob))

            # Check expected tokens
            expected_ranks = {}
            for exp_token in expected_tokens:
                token_ids = tokenizer.encode(exp_token, add_special_tokens=False)
                if token_ids:
                    token_id = token_ids[0]
                    rank = int((logits >= logits[token_id]).sum().item())
                    expected_ranks[exp_token] = {
                        "rank": rank,
                        "prob": probs[token_id].item(),
                    }

            layer_results[task] = {
                "top_tokens": top_tokens,
                "expected_tokens": expected_ranks,
            }

            logger.info(f"  {task}:")
            logger.info(f"    Top 5: {[t[0] for t in top_tokens[:5]]}")
            for exp, data in expected_ranks.items():
                logger.info(f"    '{exp}' rank: {data['rank']}")

        results["layers"][layer] = layer_results

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"vocab_alignment_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("VOCAB ALIGNMENT RESULTS")
    print("=" * 70)

    for layer, layer_data in results["layers"].items():
        print(f"\nLayer {layer}:")
        for task, task_data in layer_data.items():
            top = task_data["top_tokens"][0]
            print(f"  {task}: top='{top[0]}' ({top[1]:.4f})")
            for exp, exp_data in task_data["expected_tokens"].items():
                in_top = "✓" if exp_data["rank"] <= 10 else ""
                print(f"    '{exp}' rank {exp_data['rank']} {in_top}")

    print("\n" + "=" * 70)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_vocab_alignment_test()
