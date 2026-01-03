"""Embedding analysis commands for introspection CLI.

Commands for analyzing what information is encoded at the embedding level
and in early layers.
"""

import json


def introspect_embedding(args):
    """Analyze what information is encoded at the embedding level vs after layers.

    This tests the RLVF backprop hypothesis: if RLVF gradients backprop to embeddings,
    we should find task-relevant information (like "is this arithmetic?") already
    encoded in the raw embeddings before any transformer layer computation.

    Tests:
    1. Task type detection (arithmetic vs language) from embeddings
    2. Operation type detection (mult vs add) from embeddings
    3. Answer correlation with embeddings vs after layers
    """
    import mlx.core as mx
    import numpy as np
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import cross_val_score

    from ....introspection import CaptureConfig, ModelHooks, PositionSelection
    from ....introspection.ablation import AblationStudy

    print(f"Loading model: {args.model}")
    study = AblationStudy.from_pretrained(args.model)
    model = study.adapter.model
    tokenizer = study.adapter.tokenizer
    config = study.adapter.config

    # Generate test prompts
    # Arithmetic prompts
    arith_prompts = []
    arith_answers = []
    if args.operation in ["*", "mult", "all", None]:
        for a in range(2, 8):
            for b in range(2, 8):
                arith_prompts.append(f"{a}*{b}=")
                arith_answers.append(a * b)
    if args.operation in ["+", "add", "all", None]:
        for a in range(2, 8):
            for b in range(2, 8):
                arith_prompts.append(f"{a}+{b}=")
                arith_answers.append(a + b)

    # Language prompts
    lang_prompts = [
        "The capital of France is",
        "Hello world",
        "Paris is a beautiful",
        "I went to the store",
        "The cat sat on the",
        "Once upon a time",
        "The quick brown fox",
        "It was a dark and",
    ]

    print(
        f"\nCollecting embeddings for {len(arith_prompts)} arithmetic + {len(lang_prompts)} language prompts..."
    )

    # Parse layers to analyze
    if args.layers:
        layers = [int(layer.strip()) for layer in args.layers.split(",")]
    else:
        layers = [0, 1, 2]  # Embedding and first few layers

    def get_embeddings_and_hidden(prompt, layers_to_capture):
        """Get embedding and hidden states at specified layers."""
        # Get raw embeddings (before any layer)
        input_ids = tokenizer.encode(prompt, return_tensors="np")
        input_ids_mx = mx.array(input_ids)

        # Access embedding layer directly
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            embed = model.model.embed_tokens(input_ids_mx)
        elif hasattr(model, "embed_tokens"):
            embed = model.embed_tokens(input_ids_mx)
        else:
            raise AttributeError("Cannot find embedding layer")

        embedding = np.array(embed[0, -1, :].astype(mx.float32), copy=False)

        # Get hidden states at specified layers
        hooks = ModelHooks(model, model_config=config)
        hooks.configure(
            CaptureConfig(
                layers=layers_to_capture,
                capture_hidden_states=True,
                positions=PositionSelection.LAST,
            )
        )
        hooks.forward(input_ids_mx)

        hidden_states = {}
        for layer in layers_to_capture:
            h = hooks.state.hidden_states[layer][0, 0, :]
            hidden_states[layer] = np.array(h.astype(mx.float32), copy=False)

        return embedding, hidden_states

    # Collect all data
    all_embeddings = []
    all_hidden = {layer: [] for layer in layers}
    all_task_labels = []  # 1 = arithmetic, 0 = language
    all_answers = []  # numerical answer for arithmetic, None for language

    # Arithmetic prompts
    for i, prompt in enumerate(arith_prompts):
        emb, hidden = get_embeddings_and_hidden(prompt, layers)
        all_embeddings.append(emb)
        for layer in layers:
            all_hidden[layer].append(hidden[layer])
        all_task_labels.append(1)
        all_answers.append(arith_answers[i])

    # Language prompts
    for prompt in lang_prompts:
        emb, hidden = get_embeddings_and_hidden(prompt, layers)
        all_embeddings.append(emb)
        for layer in layers:
            all_hidden[layer].append(hidden[layer])
        all_task_labels.append(0)
        all_answers.append(None)

    all_embeddings = np.array(all_embeddings)
    for layer in layers:
        all_hidden[layer] = np.array(all_hidden[layer])
    all_task_labels = np.array(all_task_labels)

    results = {}

    # Test 1: Task type detection from embeddings
    print(f"\n{'=' * 70}")
    print("TEST 1: TASK TYPE DETECTION")
    print(f"{'=' * 70}")

    X_emb = all_embeddings
    y_task = all_task_labels

    probe = LogisticRegression(max_iter=1000, random_state=42)
    try:
        scores = cross_val_score(probe, X_emb, y_task, cv=5)
        emb_task_acc = float(np.mean(scores))
    except ValueError:
        probe.fit(X_emb, y_task)
        emb_task_acc = float(probe.score(X_emb, y_task))

    print(f"Task type from embeddings: {emb_task_acc:.1%}")
    results["task_from_embedding"] = emb_task_acc

    # Check at each layer
    for layer in layers:
        X_layer = all_hidden[layer]
        probe = LogisticRegression(max_iter=1000, random_state=42)
        try:
            scores = cross_val_score(probe, X_layer, y_task, cv=5)
            layer_task_acc = float(np.mean(scores))
        except ValueError:
            probe.fit(X_layer, y_task)
            layer_task_acc = float(probe.score(X_layer, y_task))
        print(f"Task type after L{layer}: {layer_task_acc:.1%}")
        results[f"task_after_L{layer}"] = layer_task_acc

    # Test 2: Answer correlation (for arithmetic only)
    print(f"\n{'=' * 70}")
    print("TEST 2: ANSWER CORRELATION (arithmetic only)")
    print(f"{'=' * 70}")

    arith_mask = all_task_labels == 1
    X_arith_emb = all_embeddings[arith_mask]
    y_answers = np.array([a for a in all_answers if a is not None])

    reg = LinearRegression()
    reg.fit(X_arith_emb, y_answers)
    y_pred = reg.predict(X_arith_emb)
    ss_res = np.sum((y_answers - y_pred) ** 2)
    ss_tot = np.sum((y_answers - np.mean(y_answers)) ** 2)
    r2_emb = 1 - (ss_res / ss_tot)

    print(f"Answer R2 from embeddings: {r2_emb:.3f}")
    results["answer_r2_embedding"] = float(r2_emb)

    for layer in layers:
        X_arith_layer = all_hidden[layer][arith_mask]
        reg = LinearRegression()
        reg.fit(X_arith_layer, y_answers)
        y_pred = reg.predict(X_arith_layer)
        ss_res = np.sum((y_answers - y_pred) ** 2)
        ss_tot = np.sum((y_answers - np.mean(y_answers)) ** 2)
        r2_layer = 1 - (ss_res / ss_tot)
        print(f"Answer R2 after L{layer}: {r2_layer:.3f}")
        results[f"answer_r2_L{layer}"] = float(r2_layer)

    # Test 3: Embedding similarity analysis
    print(f"\n{'=' * 70}")
    print("TEST 3: EMBEDDING SIMILARITY ANALYSIS")
    print(f"{'=' * 70}")

    # Compare last-token embeddings across prompts
    arith_embeddings = all_embeddings[arith_mask]
    lang_embeddings = all_embeddings[~arith_mask]

    def mean_pairwise_cosine(embeddings):
        """Compute mean pairwise cosine similarity."""
        n = len(embeddings)
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                dot = np.dot(embeddings[i], embeddings[j])
                norm_i = np.linalg.norm(embeddings[i])
                norm_j = np.linalg.norm(embeddings[j])
                sims.append(dot / (norm_i * norm_j + 1e-8))
        return float(np.mean(sims)) if sims else 0.0

    within_arith = mean_pairwise_cosine(arith_embeddings)
    within_lang = mean_pairwise_cosine(lang_embeddings)

    # Cross-task similarity
    cross_sims = []
    for ae in arith_embeddings:
        for le in lang_embeddings:
            dot = np.dot(ae, le)
            norm_a = np.linalg.norm(ae)
            norm_l = np.linalg.norm(le)
            cross_sims.append(dot / (norm_a * norm_l + 1e-8))
    between_task = float(np.mean(cross_sims))

    print(f"Within arithmetic similarity: {within_arith:.4f}")
    print(f"Within language similarity: {within_lang:.4f}")
    print(f"Between task similarity: {between_task:.4f}")

    results["within_arith_sim"] = within_arith
    results["within_lang_sim"] = within_lang
    results["between_task_sim"] = between_task

    # Interpretation
    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print(f"{'=' * 70}")

    if results["task_from_embedding"] > 0.9:
        print("Task type is BAKED INTO embeddings (100% detection)")
        print("  -> Consistent with RLVF backprop hypothesis")
    else:
        print(f"Task type partially encoded ({results['task_from_embedding']:.0%})")
        print("  -> May need more layer computation to determine task")

    if results["answer_r2_embedding"] < 0.1:
        print("Answer NOT in embeddings (requires computation)")
    else:
        print(f"? Answer partially encoded in embeddings (R2={results['answer_r2_embedding']:.2f})")

    # Save results
    if args.output:
        output_data = {
            "model": args.model,
            "num_arith_prompts": len(arith_prompts),
            "num_lang_prompts": len(lang_prompts),
            "layers_analyzed": layers,
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


def introspect_early_layers(args):
    """Analyze what information is encoded in early layers.

    This command probes what the model has "computed" at each early layer by testing
    whether linear probes can extract:
    - Operation type (*, +, -)
    - Operand values (A and B)
    - The final answer

    Key insight: Even when hidden states look similar (high cosine similarity),
    information can be encoded in orthogonal subspaces. This command reveals when
    different pieces of information become linearly extractable.

    This is useful for understanding:
    - How quickly the model "understands" what computation to do
    - Whether computation happens in early layers (answer extractable early)
    - The difference between "representation similarity" and "information content"
    """
    import mlx.core as mx
    import numpy as np
    from sklearn.linear_model import LogisticRegression, Ridge

    from ....introspection import CaptureConfig, ModelHooks, PositionSelection
    from ....introspection.ablation import AblationStudy

    print(f"Loading model: {args.model}")
    study = AblationStudy.from_pretrained(args.model)
    model = study.adapter.model
    tokenizer = study.adapter.tokenizer
    config = study.adapter.config

    # Parse layers
    if args.layers:
        layers = [int(layer.strip()) for layer in args.layers.split(",")]
    else:
        # Default: first few layers
        num_layers = study.adapter.num_layers
        layers = [0, 1, 2, 4, min(8, num_layers - 1)]

    # Parse operations
    if args.operations:
        operations = args.operations.split(",")
    else:
        operations = ["*", "+", "-"]

    # Parse digit range
    if args.digits:
        digit_range = [int(d.strip()) for d in args.digits.split("-")]
        if len(digit_range) == 2:
            digits = list(range(digit_range[0], digit_range[1] + 1))
        else:
            digits = [int(d) for d in args.digits.split(",")]
    else:
        digits = list(range(2, 8))

    print(f"Analyzing layers: {layers}")
    print(f"Operations: {operations}")
    print(f"Digit range: {digits[0]}-{digits[-1]}")

    # Generate prompts
    prompts = []
    labels = {"op": [], "op_name": [], "a": [], "b": [], "answer": []}

    op_functions = {
        "*": lambda x, y: x * y,
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "/": lambda x, y: x / y if y != 0 else 0,
    }

    for a in digits:
        for b in digits:
            for op_idx, op in enumerate(operations):
                prompt = f"{a}{op}{b}="
                prompts.append(prompt)
                labels["op"].append(op_idx)
                labels["op_name"].append(op)
                labels["a"].append(a)
                labels["b"].append(b)
                labels["answer"].append(op_functions.get(op, lambda x, y: 0)(a, b))

    print(f"Generated {len(prompts)} prompts")

    def get_hidden(prompt, layer_idx):
        """Get last-token hidden state for a prompt at a given layer."""
        hooks = ModelHooks(model, model_config=config)
        hooks.configure(
            CaptureConfig(
                layers=[layer_idx],
                capture_hidden_states=True,
                positions=PositionSelection.LAST,
            )
        )
        input_ids = tokenizer.encode(prompt, return_tensors="np")
        hooks.forward(mx.array(input_ids))
        h = hooks.state.hidden_states[layer_idx][0, 0, :]
        return np.array(h.astype(mx.float32), copy=False)

    # Part 1: Cross-expression similarity at '=' position
    print(f"\n{'=' * 70}")
    print("PART 1: REPRESENTATION SIMILARITY")
    print(f"{'=' * 70}")
    print("How similar are different expressions at the '=' position?")

    # Pick representative expressions
    sample_exprs = []
    for op in operations[:3]:  # Up to 3 operations
        sample_exprs.append(f"{digits[0]}{op}{digits[1]}=")
    if len(sample_exprs) < 2:
        sample_exprs = [f"{digits[0]}*{digits[1]}=", f"{digits[0]}+{digits[1]}="]

    print(f"\nSample expressions: {sample_exprs}")
    print(f"\n{'Layer':<8}", end="")
    for i in range(len(sample_exprs)):
        for j in range(i + 1, len(sample_exprs)):
            print(f"{sample_exprs[i][:5]} vs {sample_exprs[j][:5]:<12}", end="")
    print()
    print("-" * (8 + 20 * (len(sample_exprs) * (len(sample_exprs) - 1) // 2)))

    similarity_results = {}
    for layer in layers:
        hiddens = [get_hidden(expr, layer) for expr in sample_exprs]
        sims = []
        for i in range(len(hiddens)):
            for j in range(i + 1, len(hiddens)):
                dot = np.dot(hiddens[i], hiddens[j])
                norm_i = np.linalg.norm(hiddens[i])
                norm_j = np.linalg.norm(hiddens[j])
                sim = float(dot / (norm_i * norm_j + 1e-8))
                sims.append(sim)

        similarity_results[layer] = sims
        print(f"L{layer:<7}", end="")
        for sim in sims:
            print(f"{sim:<20.4f}", end="")
        print()

    # Part 2: Linear probe analysis
    print(f"\n{'=' * 70}")
    print("PART 2: INFORMATION EXTRACTABILITY (Linear Probes)")
    print(f"{'=' * 70}")
    print("What can a linear probe extract at each layer?")
    print(f"\n{'Layer':<8} {'Op Acc':<12} {'A R2':<12} {'B R2':<12} {'Answer R2':<12}")
    print("-" * 56)

    probe_results = {}
    for layer in layers:
        # Collect hidden states
        X = np.array([get_hidden(p, layer) for p in prompts])

        results_layer = {}

        # Operation classification
        if len(operations) > 1:
            probe_op = LogisticRegression(max_iter=1000)
            probe_op.fit(X, labels["op"])
            op_acc = float(probe_op.score(X, labels["op"]))
        else:
            op_acc = 1.0  # Only one operation
        results_layer["op_accuracy"] = op_acc

        # Operand A regression
        probe_a = Ridge()
        probe_a.fit(X, labels["a"])
        pred_a = probe_a.predict(X)
        ss_res = np.sum((np.array(labels["a"]) - pred_a) ** 2)
        ss_tot = np.sum((np.array(labels["a"]) - np.mean(labels["a"])) ** 2)
        r2_a = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0
        results_layer["a_r2"] = r2_a

        # Operand B regression
        probe_b = Ridge()
        probe_b.fit(X, labels["b"])
        pred_b = probe_b.predict(X)
        ss_res = np.sum((np.array(labels["b"]) - pred_b) ** 2)
        ss_tot = np.sum((np.array(labels["b"]) - np.mean(labels["b"])) ** 2)
        r2_b = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0
        results_layer["b_r2"] = r2_b

        # Answer regression
        probe_ans = Ridge()
        probe_ans.fit(X, labels["answer"])
        pred_ans = probe_ans.predict(X)
        ss_res = np.sum((np.array(labels["answer"]) - pred_ans) ** 2)
        ss_tot = np.sum((np.array(labels["answer"]) - np.mean(labels["answer"])) ** 2)
        r2_ans = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0
        results_layer["answer_r2"] = r2_ans

        probe_results[layer] = results_layer

        # Print row
        print(f"L{layer:<7} {op_acc:<12.1%} {r2_a:<12.3f} {r2_b:<12.3f} {r2_ans:<12.3f}")

    # Part 3: Position-wise analysis (if requested)
    if args.analyze_positions:
        print(f"\n{'=' * 70}")
        print("PART 3: POSITION-WISE ANALYSIS")
        print(f"{'=' * 70}")
        print("How does each token position contribute?")

        sample_prompt = prompts[0]
        tokens = tokenizer.encode(sample_prompt)
        token_strs = [tokenizer.decode([t]) for t in tokens]

        print(f"\nSample: {sample_prompt!r} -> {token_strs}")

        for layer in layers[:3]:  # First 3 layers only
            print(f"\nLayer {layer} - position similarities:")

            hooks = ModelHooks(model, model_config=config)
            hooks.configure(
                CaptureConfig(
                    layers=[layer],
                    capture_hidden_states=True,
                    positions=PositionSelection.ALL,
                )
            )
            hooks.forward(mx.array(tokens)[None, :])
            h = np.array(hooks.state.hidden_states[layer].astype(mx.float32))[0]

            # Print similarity matrix
            print(f"{'':10}", end="")
            for t in token_strs:
                print(f"{t!r:>10}", end="")
            print()

            for i, ti in enumerate(token_strs):
                print(f"{ti!r:10}", end="")
                for j in range(len(token_strs)):
                    sim = np.dot(h[i], h[j]) / (np.linalg.norm(h[i]) * np.linalg.norm(h[j]) + 1e-8)
                    print(f"{sim:10.3f}", end="")
                print()

    # Summary
    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print(f"{'=' * 70}")

    # Find when answer becomes extractable
    answer_threshold = 0.95
    answer_layer = None
    for layer in layers:
        if probe_results[layer]["answer_r2"] >= answer_threshold:
            answer_layer = layer
            break

    if answer_layer is not None:
        print(f"Answer becomes extractable (R2 > {answer_threshold}) at layer {answer_layer}")
    else:
        best_layer = max(layers, key=lambda layer: probe_results[layer]["answer_r2"])
        print(
            f"Best answer extraction at layer {best_layer} (R2 = {probe_results[best_layer]['answer_r2']:.3f})"
        )

    # Check if early layers are "doing the work"
    if layers[0] in probe_results and probe_results[layers[0]]["answer_r2"] > 0.9:
        print(
            f"! Computation mostly complete by layer {layers[0]} (R2 = {probe_results[layers[0]]['answer_r2']:.3f})"
        )
        print("  -> Later layers may be formatting/output, not computation")

    # Check similarity vs extractability paradox
    first_layer = layers[0]
    if first_layer in similarity_results and first_layer in probe_results:
        avg_sim = np.mean(similarity_results[first_layer])
        ans_r2 = probe_results[first_layer]["answer_r2"]
        if avg_sim > 0.95 and ans_r2 > 0.9:
            print(f"\n! PARADOX at layer {first_layer}:")
            print(f"  - Representations look similar (avg cosine = {avg_sim:.3f})")
            print(f"  - But answer is extractable (R2 = {ans_r2:.3f})")
            print("  -> Information encoded in ORTHOGONAL subspaces")

    # Save results
    if args.output:
        output_data = {
            "model": args.model,
            "layers": layers,
            "operations": operations,
            "digits": digits,
            "num_prompts": len(prompts),
            "similarity_results": {str(k): v for k, v in similarity_results.items()},
            "probe_results": {str(k): v for k, v in probe_results.items()},
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


__all__ = [
    "introspect_embedding",
    "introspect_early_layers",
]
