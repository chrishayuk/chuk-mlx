#!/usr/bin/env python3
"""
Attention Binding Analysis for Tool Calling

Investigates HOW the model binds query content to JSON arguments:
1. Which query tokens receive attention during argument generation?
2. Do argument-relevant tokens (e.g., "Tokyo") get special attention?
3. How does attention flow differ for syntax vs content positions?
4. Does attention explain the "re-encoding" finding?
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class AttentionBindingAnalyzer:
    def __init__(self, model_path: str = "openai/gpt-oss-20b"):
        print(f"Loading model: {model_path}")
        self.model, self.tokenizer = load(model_path)
        self.model.eval()

        # Examples with clear argument binding
        self.binding_examples = [
            {
                "query": "What's the weather in Tokyo?",
                "json": '{"name": "get_weather", "arguments": {"location": "Tokyo"}}',
                "arg_word": "Tokyo",
                "param": "location"
            },
            {
                "query": "What's the weather in Paris?",
                "json": '{"name": "get_weather", "arguments": {"location": "Paris"}}',
                "arg_word": "Paris",
                "param": "location"
            },
            {
                "query": "What is 25 * 4?",
                "json": '{"name": "calculator", "arguments": {"expression": "25 * 4"}}',
                "arg_word": "25",
                "param": "expression"
            },
            {
                "query": "Calculate 100 divided by 5",
                "json": '{"name": "calculator", "arguments": {"expression": "100 / 5"}}',
                "arg_word": "100",
                "param": "expression"
            },
            {
                "query": "Search for machine learning tutorials",
                "json": '{"name": "search", "arguments": {"query": "machine learning tutorials"}}',
                "arg_word": "machine",
                "param": "query"
            },
            {
                "query": "Find information about quantum computing",
                "json": '{"name": "search", "arguments": {"query": "quantum computing"}}',
                "arg_word": "quantum",
                "param": "query"
            },
            {
                "query": "What's the temperature in London?",
                "json": '{"name": "get_weather", "arguments": {"location": "London"}}',
                "arg_word": "London",
                "param": "location"
            },
            {
                "query": "How much is 999 plus 111?",
                "json": '{"name": "calculator", "arguments": {"expression": "999 + 111"}}',
                "arg_word": "999",
                "param": "expression"
            },
        ]

    def get_model_components(self):
        """Get embedding and layer modules."""
        if hasattr(self.model, 'model'):
            return self.model.model.embed_tokens, self.model.model.layers
        return self.model.embed_tokens, self.model.layers

    def extract_attention_patterns(self, text: str, layers: list = [4, 8, 12, 16]) -> dict:
        """Extract attention patterns for a full sequence."""
        embed_tokens, model_layers = self.get_model_components()

        tokens = self.tokenizer.encode(text)
        input_ids = mx.array([tokens])

        h = embed_tokens(input_ids)
        seq_len = h.shape[1]
        mask = mx.triu(mx.full((seq_len, seq_len), float('-inf'), dtype=h.dtype), k=1)

        attention_patterns = {}

        for i, layer_module in enumerate(model_layers):
            if i in layers:
                # Get attention weights from this layer
                if hasattr(layer_module, 'self_attn'):
                    attn_module = layer_module.self_attn

                    # GPT-OSS uses Grouped Query Attention
                    # Get architecture params
                    if hasattr(attn_module, 'num_attention_heads'):
                        n_q_heads = attn_module.num_attention_heads
                    else:
                        n_q_heads = 64  # GPT-OSS default

                    if hasattr(attn_module, 'num_key_value_heads'):
                        n_kv_heads = attn_module.num_key_value_heads
                    else:
                        n_kv_heads = 8  # GPT-OSS default

                    if hasattr(attn_module, 'head_dim'):
                        head_dim = attn_module.head_dim
                    else:
                        head_dim = 64  # GPT-OSS default

                    # Compute Q, K projections
                    if hasattr(attn_module, 'q_proj') and hasattr(attn_module, 'k_proj'):
                        q = attn_module.q_proj(h)
                        k = attn_module.k_proj(h)
                    else:
                        h = layer_module(h, mask=mask)
                        continue

                    # Reshape for GQA
                    batch_size = q.shape[0]
                    q = q.reshape(batch_size, seq_len, n_q_heads, head_dim).transpose(0, 2, 1, 3)
                    k = k.reshape(batch_size, seq_len, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

                    # For GQA, repeat K heads to match Q heads
                    n_rep = n_q_heads // n_kv_heads
                    k = mx.repeat(k, n_rep, axis=1)

                    # Compute attention scores
                    scale = head_dim ** -0.5
                    attn_scores = (q @ k.transpose(0, 1, 3, 2)) * scale

                    # Apply causal mask
                    causal_mask = mx.triu(mx.full((seq_len, seq_len), float('-inf')), k=1)
                    attn_scores = attn_scores + causal_mask

                    # Softmax to get attention weights
                    attn_weights = mx.softmax(attn_scores, axis=-1)
                    mx.eval(attn_weights)

                    # Store attention pattern (average over heads)
                    attn_avg = np.array(attn_weights[0].mean(axis=0).astype(mx.float32))
                    attention_patterns[i] = attn_avg

            # Forward through layer
            h = layer_module(h, mask=mask)

        return {
            "tokens": tokens,
            "token_strings": [self.tokenizer.decode([t]) for t in tokens],
            "attention": attention_patterns
        }

    def find_token_positions(self, token_strings: list, word: str) -> list:
        """Find positions where a word appears in token strings."""
        positions = []
        word_lower = word.lower()
        for i, tok in enumerate(token_strings):
            if word_lower in tok.lower():
                positions.append(i)
        return positions

    def analyze_argument_attention(self, layers: list = [4, 8, 12]):
        """Analyze attention from argument positions to query positions."""
        print("\n=== Argument-Query Attention Analysis ===")

        results = {"examples": [], "summary": {}}

        for example in self.binding_examples:
            # Build full sequence: query + JSON
            full_text = f"Query: {example['query']}\nResponse: {example['json']}"

            print(f"  Processing: {example['query'][:40]}...")

            attn_result = self.extract_attention_patterns(full_text, layers)

            if not attn_result["attention"]:
                print(f"    Skipping - no attention patterns extracted")
                continue

            tokens = attn_result["token_strings"]

            # Find query region (before "Response:")
            query_end = 0
            for i, tok in enumerate(tokens):
                if "Response" in tok or "response" in tok:
                    query_end = i
                    break

            if query_end == 0:
                query_end = len(tokens) // 2  # Fallback

            # Find argument word position in query
            arg_positions = self.find_token_positions(tokens[:query_end], example["arg_word"])

            # Find argument value position in JSON (after query)
            json_arg_positions = self.find_token_positions(tokens[query_end:], example["arg_word"])
            json_arg_positions = [p + query_end for p in json_arg_positions]

            if not arg_positions or not json_arg_positions:
                print(f"    Skipping - couldn't find arg positions")
                continue

            example_result = {
                "query": example["query"],
                "arg_word": example["arg_word"],
                "query_arg_positions": arg_positions,
                "json_arg_positions": json_arg_positions,
                "query_end": query_end,
                "layers": {}
            }

            for l in layers:
                if l not in attn_result["attention"]:
                    continue

                attn = attn_result["attention"][l]

                # For each JSON argument position, analyze attention to query
                layer_result = {
                    "arg_to_query_attention": [],
                    "arg_to_arg_word_attention": [],
                    "arg_to_other_query_attention": []
                }

                for json_pos in json_arg_positions:
                    if json_pos >= attn.shape[0]:
                        continue

                    # Attention from JSON arg position to all query positions
                    query_attention = attn[json_pos, :query_end]

                    # Total attention to query region
                    total_query_attn = float(query_attention.sum())
                    layer_result["arg_to_query_attention"].append(total_query_attn)

                    # Attention specifically to the argument word in query
                    arg_word_attn = sum(query_attention[p] for p in arg_positions if p < len(query_attention))
                    layer_result["arg_to_arg_word_attention"].append(float(arg_word_attn))

                    # Attention to other query positions (not the arg word)
                    other_attn = total_query_attn - arg_word_attn
                    layer_result["arg_to_other_query_attention"].append(float(other_attn))

                # Compute averages
                if layer_result["arg_to_query_attention"]:
                    layer_result["avg_query_attention"] = float(np.mean(layer_result["arg_to_query_attention"]))
                    layer_result["avg_arg_word_attention"] = float(np.mean(layer_result["arg_to_arg_word_attention"]))
                    layer_result["avg_other_attention"] = float(np.mean(layer_result["arg_to_other_query_attention"]))

                    # Compute selectivity: how much more attention to arg word vs others
                    if layer_result["avg_other_attention"] > 0:
                        layer_result["selectivity_ratio"] = (
                            layer_result["avg_arg_word_attention"] / layer_result["avg_other_attention"]
                        )
                    else:
                        layer_result["selectivity_ratio"] = float('inf')

                example_result["layers"][l] = layer_result

            results["examples"].append(example_result)

        # Aggregate summary
        for l in layers:
            selectivities = []
            arg_attentions = []
            for ex in results["examples"]:
                if l in ex["layers"] and "selectivity_ratio" in ex["layers"][l]:
                    if ex["layers"][l]["selectivity_ratio"] != float('inf'):
                        selectivities.append(ex["layers"][l]["selectivity_ratio"])
                    arg_attentions.append(ex["layers"][l]["avg_arg_word_attention"])

            if selectivities:
                results["summary"][f"layer_{l}_avg_selectivity"] = float(np.mean(selectivities))
                results["summary"][f"layer_{l}_avg_arg_attention"] = float(np.mean(arg_attentions))

        return results

    def analyze_syntax_vs_content_attention(self, layers: list = [4, 8, 12]):
        """Compare attention patterns for syntax tokens vs content tokens."""
        print("\n=== Syntax vs Content Attention Analysis ===")

        syntax_tokens = {'{', '}', '[', ']', ':', ',', '"'}

        results = {"layers": {}}

        all_syntax_attention = {l: [] for l in layers}
        all_content_attention = {l: [] for l in layers}

        for example in self.binding_examples[:4]:  # Subset for speed
            full_text = f"Query: {example['query']}\nResponse: {example['json']}"

            attn_result = self.extract_attention_patterns(full_text, layers)

            if not attn_result["attention"]:
                continue

            tokens = attn_result["token_strings"]

            # Find query end
            query_end = 0
            for i, tok in enumerate(tokens):
                if "Response" in tok:
                    query_end = i
                    break

            # Classify JSON tokens
            for i in range(query_end, len(tokens)):
                tok = tokens[i].strip()
                is_syntax = any(s in tok for s in syntax_tokens) and len(tok) <= 2

                for l in layers:
                    if l not in attn_result["attention"]:
                        continue

                    attn = attn_result["attention"][l]
                    if i >= attn.shape[0]:
                        continue

                    # Attention this token pays to query region
                    query_attn = float(attn[i, :query_end].sum())

                    if is_syntax:
                        all_syntax_attention[l].append(query_attn)
                    else:
                        all_content_attention[l].append(query_attn)

        # Summarize
        for l in layers:
            layer_result = {}

            if all_syntax_attention[l]:
                layer_result["syntax_to_query_avg"] = float(np.mean(all_syntax_attention[l]))
                layer_result["syntax_to_query_std"] = float(np.std(all_syntax_attention[l]))
                layer_result["syntax_n"] = len(all_syntax_attention[l])

            if all_content_attention[l]:
                layer_result["content_to_query_avg"] = float(np.mean(all_content_attention[l]))
                layer_result["content_to_query_std"] = float(np.std(all_content_attention[l]))
                layer_result["content_n"] = len(all_content_attention[l])

            if all_syntax_attention[l] and all_content_attention[l]:
                # Content should attend more to query than syntax does
                layer_result["content_vs_syntax_ratio"] = (
                    layer_result["content_to_query_avg"] /
                    (layer_result["syntax_to_query_avg"] + 1e-8)
                )

            results["layers"][l] = layer_result

        return results

    def analyze_attention_flow_over_json(self, layers: list = [4, 8, 12]):
        """Analyze how attention to query changes as we move through JSON structure."""
        print("\n=== Attention Flow Over JSON Structure ===")

        results = {"examples": [], "summary": {}}

        for example in self.binding_examples[:4]:
            full_text = f"Query: {example['query']}\nResponse: {example['json']}"

            attn_result = self.extract_attention_patterns(full_text, layers)

            if not attn_result["attention"]:
                continue

            tokens = attn_result["token_strings"]

            # Find query end
            query_end = 0
            for i, tok in enumerate(tokens):
                if "Response" in tok:
                    query_end = i
                    break

            json_length = len(tokens) - query_end

            example_result = {
                "query": example["query"][:30],
                "json_length": json_length,
                "layers": {}
            }

            for l in layers:
                if l not in attn_result["attention"]:
                    continue

                attn = attn_result["attention"][l]

                # Track attention to query at each JSON position
                attention_flow = []
                for i in range(query_end, min(len(tokens), attn.shape[0])):
                    query_attn = float(attn[i, :query_end].sum())
                    attention_flow.append(query_attn)

                if attention_flow:
                    # Divide into thirds: opening, middle, closing
                    n = len(attention_flow)
                    third = max(1, n // 3)

                    example_result["layers"][l] = {
                        "opening_avg": float(np.mean(attention_flow[:third])),
                        "middle_avg": float(np.mean(attention_flow[third:2*third])),
                        "closing_avg": float(np.mean(attention_flow[2*third:])),
                        "full_flow": attention_flow[:20]  # First 20 positions
                    }

            results["examples"].append(example_result)

        # Aggregate by layer
        for l in layers:
            openings = [e["layers"][l]["opening_avg"] for e in results["examples"] if l in e["layers"]]
            middles = [e["layers"][l]["middle_avg"] for e in results["examples"] if l in e["layers"]]
            closings = [e["layers"][l]["closing_avg"] for e in results["examples"] if l in e["layers"]]

            if openings:
                results["summary"][f"layer_{l}"] = {
                    "opening_avg": float(np.mean(openings)),
                    "middle_avg": float(np.mean(middles)),
                    "closing_avg": float(np.mean(closings))
                }

        return results

    def analyze_cross_layer_attention_patterns(self, layers: list = [4, 8, 12, 16]):
        """Compare attention patterns across layers."""
        print("\n=== Cross-Layer Attention Analysis ===")

        results = {"layer_progression": {}, "correlations": {}}

        all_patterns = {l: [] for l in layers}

        for example in self.binding_examples[:4]:
            full_text = f"Query: {example['query']}\nResponse: {example['json']}"

            attn_result = self.extract_attention_patterns(full_text, layers)

            if not attn_result["attention"]:
                continue

            tokens = attn_result["token_strings"]

            # Find query end and arg positions
            query_end = 0
            for i, tok in enumerate(tokens):
                if "Response" in tok:
                    query_end = i
                    break

            arg_positions = self.find_token_positions(tokens[:query_end], example["arg_word"])

            if not arg_positions:
                continue

            for l in layers:
                if l not in attn_result["attention"]:
                    continue

                attn = attn_result["attention"][l]

                # Get attention pattern from last JSON position to query
                if attn.shape[0] > query_end:
                    last_json_pos = min(attn.shape[0] - 1, len(tokens) - 1)
                    query_attn_pattern = attn[last_json_pos, :query_end]
                    all_patterns[l].append(query_attn_pattern)

        # Compute layer-wise statistics
        for l in layers:
            if all_patterns[l]:
                patterns = np.array([p[:min(len(p) for p in all_patterns[l])] for p in all_patterns[l]])
                mean_pattern = patterns.mean(axis=0)

                results["layer_progression"][l] = {
                    "mean_attention": float(mean_pattern.mean()),
                    "max_attention": float(mean_pattern.max()),
                    "entropy": float(-np.sum(mean_pattern * np.log(mean_pattern + 1e-10))),
                    "n_samples": len(patterns)
                }

        # Compute correlations between layers
        layer_list = [l for l in layers if all_patterns[l]]
        for i, l1 in enumerate(layer_list):
            for l2 in layer_list[i+1:]:
                # Align patterns
                min_len = min(
                    min(len(p) for p in all_patterns[l1]),
                    min(len(p) for p in all_patterns[l2])
                )

                p1 = np.array([p[:min_len] for p in all_patterns[l1]]).mean(axis=0)
                p2 = np.array([p[:min_len] for p in all_patterns[l2]]).mean(axis=0)

                corr = np.corrcoef(p1, p2)[0, 1]
                results["correlations"][f"layer_{l1}_vs_{l2}"] = float(corr)

        return results

    def run_full_analysis(self):
        """Run all attention binding analyses."""
        print("=" * 60)
        print("ATTENTION BINDING ANALYSIS FOR TOOL CALLING")
        print("=" * 60)

        results = {}

        # 1. Argument-query attention
        print("\n[1/4] Analyzing argument-query attention binding...")
        results["argument_binding"] = self.analyze_argument_attention()

        # 2. Syntax vs content attention
        print("\n[2/4] Analyzing syntax vs content attention...")
        results["syntax_vs_content"] = self.analyze_syntax_vs_content_attention()

        # 3. Attention flow over JSON
        print("\n[3/4] Analyzing attention flow over JSON structure...")
        results["attention_flow"] = self.analyze_attention_flow_over_json()

        # 4. Cross-layer patterns
        print("\n[4/4] Analyzing cross-layer attention patterns...")
        results["cross_layer"] = self.analyze_cross_layer_attention_patterns()

        # Generate summary
        results["summary"] = self._generate_summary(results)

        # Save
        output_path = RESULTS_DIR / "attention_binding_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=lambda x: None if x != x else x)  # Handle NaN
        print(f"\nResults saved to {output_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for finding in results["summary"]["findings"]:
            print(f"  - {finding}")

        return results

    def _generate_summary(self, results):
        """Generate summary of findings."""
        findings = []

        # Argument binding
        binding = results.get("argument_binding", {}).get("summary", {})
        for k, v in binding.items():
            if "selectivity" in k and v is not None:
                layer = k.split("_")[1]
                findings.append(f"Layer {layer}: {v:.2f}x selectivity for argument words")
            elif "arg_attention" in k and v is not None:
                layer = k.split("_")[1]
                findings.append(f"Layer {layer}: {v:.3f} avg attention to argument words")

        # Syntax vs content
        syntax = results.get("syntax_vs_content", {}).get("layers", {})
        for l, data in syntax.items():
            if "content_vs_syntax_ratio" in data:
                findings.append(f"Layer {l}: content tokens attend {data['content_vs_syntax_ratio']:.2f}x more to query than syntax")

        # Attention flow
        flow = results.get("attention_flow", {}).get("summary", {})
        for l, data in flow.items():
            if isinstance(data, dict):
                layer = l.split("_")[1]
                findings.append(f"Layer {layer}: attention flow opening={data['opening_avg']:.3f}, middle={data['middle_avg']:.3f}, closing={data['closing_avg']:.3f}")

        # Cross-layer correlations
        corrs = results.get("cross_layer", {}).get("correlations", {})
        if corrs:
            avg_corr = np.mean(list(corrs.values()))
            findings.append(f"Cross-layer attention pattern correlation: {avg_corr:.3f}")

        return {
            "findings": findings,
            "interpretation": self._interpret_findings(results)
        }

    def _interpret_findings(self, results):
        """Interpret the overall findings."""
        # Check selectivity
        selectivities = []
        binding = results.get("argument_binding", {}).get("summary", {})
        for k, v in binding.items():
            if "selectivity" in k and v is not None and v != float('inf'):
                selectivities.append(v)

        if selectivities:
            avg_sel = np.mean(selectivities)
            if avg_sel > 2.0:
                binding_finding = f"Strong selective attention to argument words ({avg_sel:.1f}x)"
            elif avg_sel > 1.0:
                binding_finding = f"Moderate selective attention ({avg_sel:.1f}x)"
            else:
                binding_finding = "Weak/no selective attention to argument words"
        else:
            binding_finding = "Could not measure argument selectivity"

        # Check syntax vs content
        syntax = results.get("syntax_vs_content", {}).get("layers", {})
        ratios = [d.get("content_vs_syntax_ratio", 1) for d in syntax.values() if isinstance(d, dict)]
        if ratios:
            avg_ratio = np.mean(ratios)
            if avg_ratio > 1.5:
                content_finding = "Content tokens attend more to query than syntax tokens"
            else:
                content_finding = "Similar attention patterns for syntax and content"
        else:
            content_finding = "Could not compare syntax vs content"

        return f"{binding_finding}. {content_finding}."


if __name__ == "__main__":
    analyzer = AttentionBindingAnalyzer()
    results = analyzer.run_full_analysis()
