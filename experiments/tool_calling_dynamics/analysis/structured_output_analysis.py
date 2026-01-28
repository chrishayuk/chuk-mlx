#!/usr/bin/env python3
"""
Structured Output Analysis for Tool Calling

Investigates HOW the model processes and represents structured tool calls:
1. JSON structure representation - how are different parts encoded?
2. Token-type patterns - syntax vs keys vs values
3. Schema awareness - does model understand structure at each position?
4. Argument binding - how query content maps to parameters
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class StructuredOutputAnalyzer:
    def __init__(self, model_path: str = "openai/gpt-oss-20b"):
        print(f"Loading model: {model_path}")
        self.model, self.tokenizer = load(model_path)
        self.model.eval()

        # Complete tool call examples with query + JSON response
        self.examples = [
            {
                "query": "What is 25 * 4?",
                "tool": "calculator",
                "json": '{"name": "calculator", "arguments": {"expression": "25 * 4"}}'
            },
            {
                "query": "What's the weather in Tokyo?",
                "tool": "get_weather",
                "json": '{"name": "get_weather", "arguments": {"location": "Tokyo"}}'
            },
            {
                "query": "Search for recent AI news",
                "tool": "search",
                "json": '{"name": "search", "arguments": {"query": "recent AI news"}}'
            },
            {
                "query": "Run print('hello') in Python",
                "tool": "code_exec",
                "json": '{"name": "code_exec", "arguments": {"code": "print(\'hello\')"}}'
            },
            {
                "query": "Calculate 100 divided by 5",
                "tool": "calculator",
                "json": '{"name": "calculator", "arguments": {"expression": "100 / 5"}}'
            },
            {
                "query": "What's the temperature in Paris?",
                "tool": "get_weather",
                "json": '{"name": "get_weather", "arguments": {"location": "Paris"}}'
            },
            {
                "query": "Find information about machine learning",
                "tool": "search",
                "json": '{"name": "search", "arguments": {"query": "machine learning"}}'
            },
            {
                "query": "Execute for i in range(5): print(i)",
                "tool": "code_exec",
                "json": '{"name": "code_exec", "arguments": {"code": "for i in range(5): print(i)"}}'
            }
        ]

    def get_model_components(self):
        """Get embedding and layer modules."""
        if hasattr(self.model, 'model'):
            return self.model.model.embed_tokens, self.model.model.layers
        return self.model.embed_tokens, self.model.layers

    def classify_json_token(self, token_str: str, context: str) -> str:
        """Classify token as syntax, key, or value based on context."""
        # Clean token
        t = token_str.strip()

        # Syntax: braces, colons, quotes, commas
        if t in ['{', '}', '[', ']', ':', ',', '"', "'"]:
            return "syntax"
        if t.startswith('{"') or t.endswith('"}') or t.endswith('":'):
            return "syntax"

        # Keys: known JSON keys for tool calls
        keys = ['name', 'arguments', 'expression', 'location', 'query', 'code', 'type']
        if any(k in t.lower() for k in keys) and '"' in context[-20:]:
            return "key"

        # Tool names
        tools = ['calculator', 'get_weather', 'search', 'code_exec']
        if any(tool in t.lower() for tool in tools):
            return "tool_name"

        return "value"

    def get_hidden_states_for_text(self, text: str, layers: list = [4, 8, 12]) -> dict:
        """Get hidden states for each token position in text."""
        embed_tokens, model_layers = self.get_model_components()

        tokens = self.tokenizer.encode(text)
        input_ids = mx.array([tokens])

        h = embed_tokens(input_ids)
        seq_len = h.shape[1]
        mask = mx.triu(mx.full((seq_len, seq_len), float('-inf'), dtype=h.dtype), k=1)

        layer_states = {}
        for i, layer_module in enumerate(model_layers):
            h = layer_module(h, mask=mask)
            if i in layers:
                mx.eval(h)
                layer_states[i] = np.array(h[0].astype(mx.float32))

        # Decode individual tokens
        token_strs = [self.tokenizer.decode([t]) for t in tokens]

        return {
            "tokens": tokens,
            "token_strings": token_strs,
            "hidden_states": layer_states
        }

    def analyze_json_structure_encoding(self, layers: list = [4, 8, 12]):
        """Analyze how different parts of JSON tool calls are encoded."""
        print("\n=== JSON Structure Encoding Analysis ===")

        # Collect hidden states by token type
        states_by_type = {l: {"syntax": [], "key": [], "tool_name": [], "value": []} for l in layers}

        for example in self.examples:
            # Full context: query + response
            full_text = f"Query: {example['query']}\nTool call: {example['json']}"

            result = self.get_hidden_states_for_text(full_text, layers)

            # Find where JSON starts (after "Tool call: ")
            json_start = full_text.find(example['json'])
            prefix_tokens = len(self.tokenizer.encode(full_text[:json_start]))

            # Classify each token in the JSON part
            json_tokens = result["tokens"][prefix_tokens:]
            json_strings = result["token_strings"][prefix_tokens:]

            context = ""
            for i, (tok, tok_str) in enumerate(zip(json_tokens, json_strings)):
                context += tok_str
                token_type = self.classify_json_token(tok_str, context)

                pos = prefix_tokens + i
                if pos < result["hidden_states"][layers[0]].shape[0]:
                    for l in layers:
                        states_by_type[l][token_type].append(result["hidden_states"][l][pos])

        # Analyze separability
        results = {"layers": {}}

        for l in layers:
            layer_result = {"token_counts": {}}

            for token_type in ["syntax", "key", "tool_name", "value"]:
                layer_result["token_counts"][token_type] = len(states_by_type[l][token_type])

            print(f"  Layer {l} token counts: {layer_result['token_counts']}")

            # Compute centroids
            centroids = {}
            for token_type, states in states_by_type[l].items():
                if len(states) >= 3:
                    centroids[token_type] = np.mean(states, axis=0)

            # Pairwise distances
            layer_result["centroid_distances"] = {}
            types = list(centroids.keys())
            for i, t1 in enumerate(types):
                for t2 in types[i+1:]:
                    dist = np.linalg.norm(centroids[t1] - centroids[t2])
                    layer_result["centroid_distances"][f"{t1}_vs_{t2}"] = float(dist)

            # Train classifier to distinguish token types
            X, y = [], []
            for token_type, states in states_by_type[l].items():
                for s in states:
                    X.append(s)
                    y.append(token_type)

            if len(set(y)) >= 2 and len(X) >= 20:
                X = np.array(X)
                y = np.array(y)

                clf = LogisticRegression(max_iter=1000, random_state=42)
                clf.fit(X, y)
                acc = accuracy_score(y, clf.predict(X))

                layer_result["token_type_accuracy"] = float(acc)

                # Per-class accuracy
                for token_type in set(y):
                    mask = y == token_type
                    if mask.sum() > 0:
                        type_acc = accuracy_score(y[mask], clf.predict(X[mask]))
                        layer_result[f"accuracy_{token_type}"] = float(type_acc)

            results["layers"][l] = layer_result

        return results

    def analyze_position_structure(self, layers: list = [4, 8, 12]):
        """Analyze if position within JSON predicts structure."""
        print("\n=== Position-Structure Analysis ===")

        # Different JSON positions: open_brace, name_key, name_value, args_key, inner_key, inner_value, close
        position_states = {l: {} for l in layers}

        for example in self.examples:
            json_str = example['json']
            result = self.get_hidden_states_for_text(f"Tool: {json_str}", layers)

            # Find key positions in the JSON
            # Tokenize the JSON and identify structural positions
            json_tokens = self.tokenizer.encode(json_str)
            json_text = self.tokenizer.decode(json_tokens)

            # Get prefix length
            prefix = "Tool: "
            prefix_len = len(self.tokenizer.encode(prefix))

            for i, tok in enumerate(json_tokens):
                tok_str = self.tokenizer.decode([tok])
                pos = prefix_len + i

                if pos >= result["hidden_states"][layers[0]].shape[0]:
                    continue

                # Classify structural position
                if '{' in tok_str and i < 3:
                    position = "json_open"
                elif '"name"' in tok_str.lower() or (i < 5 and 'name' in tok_str.lower()):
                    position = "name_key"
                elif '"arguments"' in tok_str.lower() or 'arguments' in tok_str.lower():
                    position = "args_key"
                elif '}' in tok_str:
                    position = "json_close"
                else:
                    continue  # Skip non-structural positions

                for l in layers:
                    if position not in position_states[l]:
                        position_states[l][position] = []
                    position_states[l][position].append(result["hidden_states"][l][pos])

        # Analyze
        results = {"layers": {}}

        for l in layers:
            layer_result = {}

            # Count per position
            counts = {p: len(s) for p, s in position_states[l].items()}
            layer_result["position_counts"] = counts
            print(f"  Layer {l} position counts: {counts}")

            # Distances between structural positions
            centroids = {}
            for pos, states in position_states[l].items():
                if len(states) >= 2:
                    centroids[pos] = np.mean(states, axis=0)

            if len(centroids) >= 2:
                layer_result["position_distances"] = {}
                positions = list(centroids.keys())
                for i, p1 in enumerate(positions):
                    for p2 in positions[i+1:]:
                        dist = np.linalg.norm(centroids[p1] - centroids[p2])
                        layer_result["position_distances"][f"{p1}_vs_{p2}"] = float(dist)

            results["layers"][l] = layer_result

        return results

    def analyze_query_argument_binding(self, layers: list = [8, 12]):
        """Analyze how query information is bound to argument positions."""
        print("\n=== Query-Argument Binding Analysis ===")

        binding_examples = [
            {"query": "weather in Tokyo", "arg": "Tokyo", "param": "location"},
            {"query": "weather in Paris", "arg": "Paris", "param": "location"},
            {"query": "calculate 25 * 4", "arg": "25 * 4", "param": "expression"},
            {"query": "calculate 100 / 5", "arg": "100 / 5", "param": "expression"},
            {"query": "search for AI news", "arg": "AI news", "param": "query"},
            {"query": "search for machine learning", "arg": "machine learning", "param": "query"},
        ]

        results = {"examples": [], "summary": {}}

        for ex in binding_examples:
            # Get hidden states for query
            query_result = self.get_hidden_states_for_text(f"Query: {ex['query']}", layers)

            # Get hidden states for the argument value in JSON context
            json_context = f'{{"arguments": {{"{ex["param"]}": "{ex["arg"]}"}}}}'
            json_result = self.get_hidden_states_for_text(json_context, layers)

            # Find position of the argument value in JSON
            arg_tokens = self.tokenizer.encode(ex['arg'])

            example_result = {
                "query": ex['query'],
                "argument": ex['arg'],
                "layers": {}
            }

            for l in layers:
                # Last position of query (summarizes query content)
                query_final = query_result["hidden_states"][l][-1]

                # Mean of JSON hidden states (represents structured context)
                json_mean = np.mean(json_result["hidden_states"][l], axis=0)

                # Cosine similarity between query representation and JSON representation
                cos_sim = np.dot(query_final, json_mean) / (
                    np.linalg.norm(query_final) * np.linalg.norm(json_mean) + 1e-8
                )

                example_result["layers"][l] = {
                    "query_json_cosine": float(cos_sim)
                }

            results["examples"].append(example_result)

        # Summary statistics
        for l in layers:
            cosines = [e["layers"][l]["query_json_cosine"] for e in results["examples"]]
            results["summary"][f"layer_{l}_avg_cosine"] = float(np.mean(cosines))
            results["summary"][f"layer_{l}_std_cosine"] = float(np.std(cosines))

        return results

    def analyze_schema_representation(self, layers: list = [4, 8, 12]):
        """Analyze how schemas are represented - does model understand structure?"""
        print("\n=== Schema Representation Analysis ===")

        # Different schemas
        schemas = [
            '{"name": "calculator", "arguments": {"expression": ""}}',
            '{"name": "get_weather", "arguments": {"location": ""}}',
            '{"name": "search", "arguments": {"query": ""}}',
            '{"name": "code_exec", "arguments": {"code": ""}}',
        ]

        # Get representations for each schema
        schema_states = {l: [] for l in layers}
        schema_names = []

        for schema in schemas:
            result = self.get_hidden_states_for_text(f"Schema: {schema}", layers)
            for l in layers:
                # Use mean pooling over all positions
                schema_states[l].append(np.mean(result["hidden_states"][l], axis=0))

            # Extract tool name
            import re
            match = re.search(r'"name":\s*"(\w+)"', schema)
            schema_names.append(match.group(1) if match else "unknown")

        results = {"layers": {}, "schema_names": schema_names}

        for l in layers:
            states = np.array(schema_states[l])

            # PCA to visualize schema space
            if len(states) >= 2:
                pca = PCA(n_components=min(2, len(states)))
                projected = pca.fit_transform(states)

                layer_result = {
                    "pca_coords": projected.tolist(),
                    "explained_variance": pca.explained_variance_ratio_.tolist()
                }

                # Pairwise distances
                distances = {}
                for i in range(len(schema_names)):
                    for j in range(i+1, len(schema_names)):
                        dist = np.linalg.norm(states[i] - states[j])
                        distances[f"{schema_names[i]}_vs_{schema_names[j]}"] = float(dist)

                layer_result["schema_distances"] = distances
                results["layers"][l] = layer_result

        return results

    def analyze_expert_routing_by_json_part(self, layers: list = [4, 8, 12]):
        """Analyze if different JSON parts route to different experts."""
        print("\n=== Expert Routing by JSON Part ===")

        embed_tokens, model_layers = self.get_model_components()

        # Track expert activations by JSON part
        expert_by_part = {l: {"syntax": [], "key": [], "value": []} for l in layers}

        for example in self.examples[:4]:  # Subset for speed
            full_text = f"Tool: {example['json']}"
            tokens = self.tokenizer.encode(full_text)
            input_ids = mx.array([tokens])

            h = embed_tokens(input_ids)
            seq_len = h.shape[1]
            mask = mx.triu(mx.full((seq_len, seq_len), float('-inf'), dtype=h.dtype), k=1)

            # Track experts at each position
            for i, layer_module in enumerate(model_layers):
                if i in layers and hasattr(layer_module, 'mlp') and hasattr(layer_module.mlp, 'router'):
                    router = layer_module.mlp.router
                    router_input = h
                    router_logits = router(router_input)
                    mx.eval(router_logits)

                    # Get top-k experts for each position
                    top_k = 4
                    expert_indices = mx.argpartition(router_logits, kth=-top_k, axis=-1)[..., -top_k:]
                    mx.eval(expert_indices)
                    expert_indices = np.array(expert_indices[0])

                    # Classify each position and record experts
                    prefix_len = len(self.tokenizer.encode("Tool: "))
                    json_str = example['json']
                    context = ""

                    for pos in range(prefix_len, len(tokens)):
                        if pos >= len(expert_indices):
                            continue
                        tok_str = self.tokenizer.decode([tokens[pos]])
                        context += tok_str
                        part = self.classify_json_token(tok_str, context)
                        if part in ["syntax", "key", "value"]:
                            expert_by_part[i][part].append(set(expert_indices[pos].tolist()))

                h = layer_module(h, mask=mask)

        # Analyze expert patterns
        results = {"layers": {}}

        for l in layers:
            layer_result = {}

            for part in ["syntax", "key", "value"]:
                activations = expert_by_part[l][part]
                if activations:
                    # Most common experts
                    from collections import Counter
                    all_experts = [e for s in activations for e in s]
                    counts = Counter(all_experts)

                    layer_result[part] = {
                        "n_tokens": len(activations),
                        "unique_experts": len(set(all_experts)),
                        "top_5": [{"expert": e, "count": c} for e, c in counts.most_common(5)]
                    }

            # Expert overlap between parts
            for p1, p2 in [("syntax", "key"), ("syntax", "value"), ("key", "value")]:
                if p1 in layer_result and p2 in layer_result:
                    e1 = set(x["expert"] for x in layer_result[p1]["top_5"])
                    e2 = set(x["expert"] for x in layer_result[p2]["top_5"])
                    if e1 and e2:
                        overlap = len(e1 & e2) / len(e1 | e2)
                        layer_result[f"overlap_{p1}_{p2}"] = float(overlap)

            results["layers"][l] = layer_result

        return results

    def run_full_analysis(self):
        """Run all structured output analyses."""
        print("=" * 60)
        print("STRUCTURED OUTPUT ANALYSIS FOR TOOL CALLING")
        print("=" * 60)

        results = {}

        # 1. JSON structure encoding
        print("\n[1/5] Analyzing JSON structure encoding...")
        results["json_structure"] = self.analyze_json_structure_encoding()

        # 2. Position-structure relationship
        print("\n[2/5] Analyzing position-structure relationships...")
        results["position_structure"] = self.analyze_position_structure()

        # 3. Query-argument binding
        print("\n[3/5] Analyzing query-argument binding...")
        results["argument_binding"] = self.analyze_query_argument_binding()

        # 4. Schema representation
        print("\n[4/5] Analyzing schema representation...")
        results["schema_representation"] = self.analyze_schema_representation()

        # 5. Expert routing by JSON part
        print("\n[5/5] Analyzing expert routing by JSON part...")
        results["expert_routing"] = self.analyze_expert_routing_by_json_part()

        # Generate summary
        results["summary"] = self._generate_summary(results)

        # Save
        output_path = RESULTS_DIR / "structured_output_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for finding in results["summary"]["findings"]:
            print(f"  • {finding}")

        return results

    def _generate_summary(self, results):
        """Generate summary of findings."""
        findings = []

        # JSON structure
        json_struct = results.get("json_structure", {}).get("layers", {})
        for l, data in json_struct.items():
            if "token_type_accuracy" in data:
                findings.append(f"Layer {l}: {data['token_type_accuracy']*100:.1f}% accuracy classifying JSON token types")

        # Schema distances
        schema = results.get("schema_representation", {}).get("layers", {})
        for l, data in schema.items():
            if "schema_distances" in data:
                dists = list(data["schema_distances"].values())
                if dists:
                    avg_dist = np.mean(dists)
                    findings.append(f"Layer {l}: avg schema distance = {avg_dist:.1f}")

        # Expert routing overlap
        routing = results.get("expert_routing", {}).get("layers", {})
        for l, data in routing.items():
            overlaps = [v for k, v in data.items() if k.startswith("overlap_")]
            if overlaps:
                findings.append(f"Layer {l}: {np.mean(overlaps)*100:.1f}% expert overlap between JSON parts")

        # Argument binding
        binding = results.get("argument_binding", {}).get("summary", {})
        for k, v in binding.items():
            if "avg_cosine" in k:
                layer = k.split("_")[1]
                findings.append(f"Layer {layer}: {v:.3f} query-argument cosine similarity")

        return {
            "findings": findings,
            "interpretation": self._interpret_findings(results)
        }

    def _interpret_findings(self, results):
        """Interpret the overall findings."""
        # Check if token types are well-separated
        json_acc = []
        for l, data in results.get("json_structure", {}).get("layers", {}).items():
            if "token_type_accuracy" in data:
                json_acc.append(data["token_type_accuracy"])

        if json_acc and np.mean(json_acc) > 0.8:
            structure_finding = "JSON structure is clearly encoded (syntax/key/value distinguished)"
        elif json_acc:
            structure_finding = "JSON structure is partially encoded"
        else:
            structure_finding = "Insufficient data for structure analysis"

        # Check expert routing
        overlaps = []
        for l, data in results.get("expert_routing", {}).get("layers", {}).items():
            for k, v in data.items():
                if k.startswith("overlap_"):
                    overlaps.append(v)

        if overlaps and np.mean(overlaps) > 0.5:
            routing_finding = "Shared experts handle different JSON parts (unified processing)"
        elif overlaps:
            routing_finding = "Some expert specialization for JSON parts"
        else:
            routing_finding = "Insufficient expert routing data"

        return f"{structure_finding}. {routing_finding}."


if __name__ == "__main__":
    analyzer = StructuredOutputAnalyzer()
    results = analyzer.run_full_analysis()
