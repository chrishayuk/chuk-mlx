"""Memory introspection commands for analyzing how facts are stored in model memory."""

import json
from collections import defaultdict


def introspect_memory(args):
    """Extract memory organization structure for facts.

    Analyzes how facts are stored in model memory by examining
    neighborhood activation patterns - what other facts co-activate
    when retrieving a specific fact.

    Reveals:
    - Memory organization (row vs column based, clusters)
    - Asymmetry (A->B vs B->A retrieval differences)
    - Attractor nodes (frequently co-activated facts)
    - Difficulty patterns (which facts are hardest)
    """
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    from ....inference.loader import DType, HFLoader
    from ....models_v2.families.registry import detect_model_family, get_family_info

    # Built-in fact generators
    def generate_multiplication_facts():
        """Generate single-digit multiplication facts."""
        facts = []
        for a in range(2, 10):
            for b in range(2, 10):
                facts.append(
                    {
                        "query": f"{a}*{b}=",
                        "answer": str(a * b),
                        "operand_a": a,
                        "operand_b": b,
                        "category": f"{a}x",  # Row category
                        "category_alt": f"x{b}",  # Column category
                    }
                )
        return facts

    def generate_addition_facts():
        """Generate single-digit addition facts."""
        facts = []
        for a in range(1, 10):
            for b in range(1, 10):
                facts.append(
                    {
                        "query": f"{a}+{b}=",
                        "answer": str(a + b),
                        "operand_a": a,
                        "operand_b": b,
                        "category": f"{a}+",
                        "category_alt": f"+{b}",
                    }
                )
        return facts

    def generate_capital_facts():
        """Generate country capital facts."""
        capitals = [
            ("France", "Paris"),
            ("Germany", "Berlin"),
            ("Italy", "Rome"),
            ("Spain", "Madrid"),
            ("UK", "London"),
            ("Japan", "Tokyo"),
            ("China", "Beijing"),
            ("India", "Delhi"),
            ("Brazil", "Brasilia"),
            ("Russia", "Moscow"),
            ("Canada", "Ottawa"),
            ("Australia", "Canberra"),
            ("Mexico", "Mexico City"),
            ("Egypt", "Cairo"),
            ("South Africa", "Pretoria"),
            ("Argentina", "Buenos Aires"),
            ("Poland", "Warsaw"),
            ("Netherlands", "Amsterdam"),
            ("Belgium", "Brussels"),
            ("Sweden", "Stockholm"),
            ("Norway", "Oslo"),
            ("Denmark", "Copenhagen"),
            ("Finland", "Helsinki"),
            ("Greece", "Athens"),
            ("Turkey", "Ankara"),
            ("Iran", "Tehran"),
            ("Iraq", "Baghdad"),
            ("Saudi Arabia", "Riyadh"),
            ("Israel", "Jerusalem"),
            ("Thailand", "Bangkok"),
        ]
        facts = []
        for country, capital in capitals:
            # Get continent/region for categorization
            region = (
                "Europe"
                if country
                in [
                    "France",
                    "Germany",
                    "Italy",
                    "Spain",
                    "UK",
                    "Poland",
                    "Netherlands",
                    "Belgium",
                    "Sweden",
                    "Norway",
                    "Denmark",
                    "Finland",
                    "Greece",
                ]
                else "Asia"
                if country
                in [
                    "Japan",
                    "China",
                    "India",
                    "Turkey",
                    "Iran",
                    "Iraq",
                    "Saudi Arabia",
                    "Israel",
                    "Thailand",
                ]
                else "Americas"
                if country in ["Brazil", "Canada", "Mexico", "Argentina"]
                else "Other"
            )
            facts.append(
                {
                    "query": f"The capital of {country} is",
                    "answer": capital,
                    "country": country,
                    "category": region,
                }
            )
        return facts

    def generate_element_facts():
        """Generate periodic table element facts."""
        elements = [
            (1, "H", "Hydrogen"),
            (2, "He", "Helium"),
            (3, "Li", "Lithium"),
            (4, "Be", "Beryllium"),
            (5, "B", "Boron"),
            (6, "C", "Carbon"),
            (7, "N", "Nitrogen"),
            (8, "O", "Oxygen"),
            (9, "F", "Fluorine"),
            (10, "Ne", "Neon"),
            (11, "Na", "Sodium"),
            (12, "Mg", "Magnesium"),
            (13, "Al", "Aluminum"),
            (14, "Si", "Silicon"),
            (15, "P", "Phosphorus"),
            (16, "S", "Sulfur"),
            (17, "Cl", "Chlorine"),
            (18, "Ar", "Argon"),
            (19, "K", "Potassium"),
            (20, "Ca", "Calcium"),
        ]
        facts = []
        for num, symbol, name in elements:
            period = 1 if num <= 2 else 2 if num <= 10 else 3
            facts.append(
                {
                    "query": f"Element {num} is",
                    "answer": name,
                    "number": num,
                    "symbol": symbol,
                    "category": f"Period {period}",
                }
            )
        return facts

    # Load facts
    fact_type = args.facts
    if fact_type.startswith("@"):
        # Load from file
        with open(fact_type[1:]) as f:
            facts = json.load(f)
    elif fact_type == "multiplication":
        facts = generate_multiplication_facts()
    elif fact_type == "addition":
        facts = generate_addition_facts()
    elif fact_type == "capitals":
        facts = generate_capital_facts()
    elif fact_type == "elements":
        facts = generate_element_facts()
    else:
        print(f"ERROR: Unknown fact type: {fact_type}")
        print("Use: multiplication, addition, capitals, elements, or @file.json")
        return

    print(f"Loading model: {args.model}")

    result = HFLoader.download(args.model)
    model_path = result.model_path

    config_path = model_path / "config.json"
    with open(config_path) as f:
        config_data = json.load(f)

    family_type = detect_model_family(config_data)
    if family_type is None:
        raise ValueError(f"Unsupported model: {args.model}")

    family_info = get_family_info(family_type)
    config = family_info.config_class.from_hf_config(config_data)
    model = family_info.model_class(config)

    HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
    tokenizer = HFLoader.load_tokenizer(model_path)

    num_layers = config.num_hidden_layers
    target_layer = args.layer if args.layer is not None else int(num_layers * 0.8)
    top_k = args.top_k

    print(f"  Layers: {num_layers}")
    print(f"  Target layer: {target_layer}")
    print(f"  Facts to analyze: {len(facts)}")
    print(f"  Top-k predictions: {top_k}")

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

    def get_predictions_at_layer(prompt: str, layer: int, k: int) -> list:
        """Get top-k predictions at specific layer using logit lens."""
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

        # Apply norm and get logits
        if norm is not None:
            h = norm(h)
        if lm_head is not None:
            outputs = lm_head(h)
            # Handle HeadOutput wrapper vs raw logits
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
        else:
            # Tied embeddings
            logits = h @ embed.weight.T

        # Get last position probabilities
        probs = mx.softmax(logits[0, -1, :], axis=-1)
        top_indices = mx.argsort(probs)[-k:][::-1]
        top_probs = probs[top_indices]

        predictions = []
        for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
            token = tokenizer.decode([idx])
            predictions.append(
                {
                    "token": token,
                    "token_id": idx,
                    "prob": prob,
                }
            )

        return predictions

    # Build answer vocabulary for categorization
    answer_vocab = {fact["answer"]: fact for fact in facts}

    print(f"\nAnalyzing {len(facts)} facts...")

    # Collect results
    results = []
    for i, fact in enumerate(facts):
        if (i + 1) % 10 == 0:
            print(f"  Processing {i + 1}/{len(facts)}...")

        query = fact["query"]
        correct_answer = fact["answer"]

        predictions = get_predictions_at_layer(query, target_layer, top_k)

        # Find correct answer rank
        correct_rank = None
        correct_prob = None
        for j, pred in enumerate(predictions):
            if pred["token"].strip() == correct_answer or correct_answer in pred["token"]:
                correct_rank = j + 1
                correct_prob = pred["prob"]
                break

        # Categorize predictions
        neighborhood = {
            "correct_rank": correct_rank,
            "correct_prob": correct_prob,
            "same_category": [],
            "same_category_alt": [],
            "other_answers": [],
            "non_answers": [],
        }

        for pred in predictions:
            token = pred["token"].strip()
            if token == correct_answer:
                continue

            # Check if this is a known answer
            if token in answer_vocab:
                other_fact = answer_vocab[token]
                # Check category match
                if "category" in fact and "category" in other_fact:
                    if fact["category"] == other_fact["category"]:
                        neighborhood["same_category"].append(
                            {
                                "answer": token,
                                "prob": pred["prob"],
                                "from_query": other_fact["query"],
                            }
                        )
                    elif "category_alt" in fact and fact.get("category_alt") == other_fact.get(
                        "category_alt"
                    ):
                        neighborhood["same_category_alt"].append(
                            {
                                "answer": token,
                                "prob": pred["prob"],
                                "from_query": other_fact["query"],
                            }
                        )
                    else:
                        neighborhood["other_answers"].append(
                            {
                                "answer": token,
                                "prob": pred["prob"],
                                "from_query": other_fact["query"],
                            }
                        )
                else:
                    neighborhood["other_answers"].append(
                        {
                            "answer": token,
                            "prob": pred["prob"],
                        }
                    )
            else:
                # Not a known answer
                neighborhood["non_answers"].append(
                    {
                        "token": token,
                        "prob": pred["prob"],
                    }
                )

        results.append(
            {
                **fact,
                "predictions": predictions[:10],  # Save top 10 for reference
                "neighborhood": neighborhood,
            }
        )

    # Aggregate analysis
    print(f"\n{'=' * 70}")
    print(f"MEMORY STRUCTURE ANALYSIS: {fact_type}")
    print(f"{'=' * 70}")

    # 1. Overall accuracy
    correct_top1 = sum(1 for r in results if r["neighborhood"]["correct_rank"] == 1)
    correct_top5 = sum(
        1
        for r in results
        if r["neighborhood"]["correct_rank"] and r["neighborhood"]["correct_rank"] <= 5
    )
    not_found = sum(1 for r in results if r["neighborhood"]["correct_rank"] is None)

    print("\n1. RETRIEVAL ACCURACY")
    print(f"   Top-1: {correct_top1}/{len(results)} ({100 * correct_top1 / len(results):.1f}%)")
    print(f"   Top-5: {correct_top5}/{len(results)} ({100 * correct_top5 / len(results):.1f}%)")
    print(
        f"   Not in top-{top_k}: {not_found}/{len(results)} ({100 * not_found / len(results):.1f}%)"
    )

    # 2. Category analysis (if applicable)
    if "category" in facts[0]:
        print("\n2. ACCURACY BY CATEGORY")
        categories = list({f["category"] for f in facts})
        for cat in sorted(categories):
            cat_facts = [r for r in results if r["category"] == cat]
            cat_top1 = sum(1 for r in cat_facts if r["neighborhood"]["correct_rank"] == 1)
            cat_avg_prob = np.mean([r["neighborhood"]["correct_prob"] or 0 for r in cat_facts])
            print(f"   {cat}: {cat_top1}/{len(cat_facts)} top-1, avg_prob={cat_avg_prob:.3f}")

    # 3. Neighborhood composition
    print("\n3. NEIGHBORHOOD COMPOSITION")
    total_same_cat = sum(len(r["neighborhood"]["same_category"]) for r in results)
    total_same_cat_alt = sum(len(r["neighborhood"]["same_category_alt"]) for r in results)
    total_other = sum(len(r["neighborhood"]["other_answers"]) for r in results)
    total_non = sum(len(r["neighborhood"]["non_answers"]) for r in results)

    print(f"   Same category (primary): {total_same_cat}")
    if total_same_cat_alt > 0:
        print(f"   Same category (alt): {total_same_cat_alt}")
    print(f"   Other known answers: {total_other}")
    print(f"   Non-answer tokens: {total_non}")

    # 4. Attractor analysis
    print("\n4. ATTRACTOR NODES (most frequently co-activated)")
    answer_counts = defaultdict(int)
    answer_probs = defaultdict(list)
    for r in results:
        for cat in ["same_category", "same_category_alt", "other_answers"]:
            for item in r["neighborhood"][cat]:
                answer_counts[item["answer"]] += 1
                answer_probs[item["answer"]].append(item["prob"])

    top_attractors = sorted(answer_counts.items(), key=lambda x: -x[1])[:10]
    for answer, count in top_attractors:
        avg_prob = np.mean(answer_probs[answer])
        print(f"   '{answer}': appears {count} times, avg_prob={avg_prob:.4f}")

    # 5. Hardest facts
    print("\n5. HARDEST FACTS (lowest retrieval rank)")
    sorted_by_difficulty = sorted(results, key=lambda x: x["neighborhood"]["correct_rank"] or 999)
    for r in sorted_by_difficulty[-10:]:
        rank = r["neighborhood"]["correct_rank"] or f">{top_k}"
        prob = r["neighborhood"]["correct_prob"] or 0
        print(f"   {r['query'][:30]:<30} -> {r['answer']}: rank={rank}, prob={prob:.4f}")

    # 6. Asymmetry analysis (for facts with operand_a and operand_b)
    if "operand_a" in facts[0] and "operand_b" in facts[0]:
        print("\n6. ASYMMETRY ANALYSIS (A op B vs B op A)")
        asymmetries = []
        for r in results:
            a, b = r["operand_a"], r["operand_b"]
            if a >= b:
                continue
            # Find reverse
            reverse = next(
                (x for x in results if x["operand_a"] == b and x["operand_b"] == a), None
            )
            if reverse:
                rank_ab = r["neighborhood"]["correct_rank"] or 999
                rank_ba = reverse["neighborhood"]["correct_rank"] or 999
                prob_ab = r["neighborhood"]["correct_prob"] or 0
                prob_ba = reverse["neighborhood"]["correct_prob"] or 0
                if abs(rank_ab - rank_ba) > 2 or abs(prob_ab - prob_ba) > 0.05:
                    asymmetries.append(
                        {
                            "a": a,
                            "b": b,
                            "rank_ab": rank_ab,
                            "rank_ba": rank_ba,
                            "prob_ab": prob_ab,
                            "prob_ba": prob_ba,
                        }
                    )

        if asymmetries:
            asymmetries.sort(key=lambda x: abs(x["rank_ab"] - x["rank_ba"]), reverse=True)
            for asym in asymmetries[:10]:
                a, b = asym["a"], asym["b"]
                print(f"   {a}*{b}: rank={asym['rank_ab']}, prob={asym['prob_ab']:.3f}")
                print(f"   {b}*{a}: rank={asym['rank_ba']}, prob={asym['prob_ba']:.3f}")
                print(f"      Delta rank={asym['rank_ab'] - asym['rank_ba']:+d}")
                print()
        else:
            print("   No significant asymmetries found")

    # 7. Row vs Column bias (for operand-based facts)
    if "category" in facts[0] and "category_alt" in facts[0]:
        print("\n7. ORGANIZATION BIAS (primary vs alt category)")
        row_bias = 0
        col_bias = 0
        neutral = 0
        for r in results:
            n_primary = len(r["neighborhood"]["same_category"])
            n_alt = len(r["neighborhood"]["same_category_alt"])
            if n_primary > n_alt:
                row_bias += 1
            elif n_alt > n_primary:
                col_bias += 1
            else:
                neutral += 1
        print(f"   Primary category bias: {row_bias}")
        print(f"   Alt category bias: {col_bias}")
        print(f"   Neutral: {neutral}")

    # 8. Memorization classification (if --classify flag)
    if getattr(args, "classify", False):
        print("\n8. MEMORIZATION CLASSIFICATION")
        print("-" * 50)

        memorized = []  # rank 1, prob > 0.1
        partial = []  # rank 2-5, prob > 0.01
        weak = []  # rank 6-15, prob > 0.001
        not_memorized = []  # rank > 15 or prob < 0.001

        for r in results:
            query = r["query"]
            answer = r["answer"]
            rank = r["neighborhood"]["correct_rank"]
            prob = r["neighborhood"]["correct_prob"] or 0

            if rank == 1 and prob > 0.1:
                memorized.append((query, answer, rank, prob))
            elif rank and rank <= 5 and prob > 0.01:
                partial.append((query, answer, rank, prob))
            elif rank and rank <= 15 and prob > 0.001:
                weak.append((query, answer, rank, prob))
            else:
                not_memorized.append((query, answer, rank, prob))

        print(f"\n   MEMORIZED ({len(memorized)} facts) - rank 1, prob > 10%")
        for q, a, r, p in sorted(memorized, key=lambda x: -x[3])[:5]:
            print(f"      {q:<20} = {a:<6} prob={p:.1%}")

        print(f"\n   PARTIALLY MEMORIZED ({len(partial)} facts) - rank 2-5, prob > 1%")
        for q, a, r, p in sorted(partial, key=lambda x: -x[3])[:5]:
            print(f"      {q:<20} = {a:<6} rank={r}, prob={p:.1%}")

        print(f"\n   WEAK ({len(weak)} facts) - rank 6-15, prob > 0.1%")
        for q, a, r, p in sorted(weak, key=lambda x: x[2] if x[2] else 999)[:5]:
            print(f"      {q:<20} = {a:<6} rank={r}, prob={p:.2%}")

        print(f"\n   NOT MEMORIZED ({len(not_memorized)} facts) - rank > 15 or prob < 0.1%")
        for q, a, r, p in sorted(not_memorized, key=lambda x: x[2] if x[2] else 999)[:5]:
            rank_str = str(r) if r else f">{top_k}"
            print(f"      {q:<20} = {a:<6} rank={rank_str}, prob={p:.3%}")

        # Summary bar
        print("\n   Summary: ", end="")
        print(
            f"[{'#' * len(memorized)}{'~' * len(partial)}{'?' * len(weak)}{'.' * len(not_memorized)}]"
        )
        print("            # memorized  ~ partial  ? weak  . not memorized")

    # Save results
    if args.output:
        output_data = {
            "model_id": args.model,
            "fact_type": fact_type,
            "layer": target_layer,
            "num_facts": len(facts),
            "accuracy": {
                "top1": correct_top1,
                "top5": correct_top5,
                "not_found": not_found,
            },
            "attractors": [
                {"answer": a, "count": c, "avg_prob": float(np.mean(answer_probs[a]))}
                for a, c in top_attractors
            ],
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")

    # Save plot
    if getattr(args, "save_plot", None):
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # Plot 1: Accuracy by category
            if "category" in facts[0]:
                ax = axes[0, 0]
                categories = sorted({f["category"] for f in facts})
                cat_accuracy = []
                for cat in categories:
                    cat_facts = [r for r in results if r["category"] == cat]
                    cat_top1 = sum(1 for r in cat_facts if r["neighborhood"]["correct_rank"] == 1)
                    cat_accuracy.append(100 * cat_top1 / len(cat_facts))
                ax.bar(categories, cat_accuracy)
                ax.set_ylabel("Top-1 Accuracy (%)")
                ax.set_title("Accuracy by Category")
                ax.tick_params(axis="x", rotation=45)

            # Plot 2: Rank distribution
            ax = axes[0, 1]
            ranks = [r["neighborhood"]["correct_rank"] or top_k + 1 for r in results]
            ax.hist(ranks, bins=range(1, top_k + 3), edgecolor="black")
            ax.set_xlabel("Correct Answer Rank")
            ax.set_ylabel("Count")
            ax.set_title("Rank Distribution")

            # Plot 3: Top attractors
            ax = axes[1, 0]
            if top_attractors:
                answers = [a for a, _ in top_attractors[:10]]
                counts = [c for _, c in top_attractors[:10]]
                ax.barh(answers, counts)
                ax.set_xlabel("Co-activation Count")
                ax.set_title("Top Attractor Nodes")

            # Plot 4: Probability vs rank
            ax = axes[1, 1]
            probs = [r["neighborhood"]["correct_prob"] or 0 for r in results]
            ranks_plot = [r["neighborhood"]["correct_rank"] or top_k + 1 for r in results]
            ax.scatter(ranks_plot, probs, alpha=0.5)
            ax.set_xlabel("Rank")
            ax.set_ylabel("Probability")
            ax.set_title("Probability vs Rank")

            plt.suptitle(f"Memory Structure: {fact_type} @ Layer {target_layer}\n{args.model}")
            plt.tight_layout()
            plt.savefig(args.save_plot, dpi=150)
            print(f"Plot saved to: {args.save_plot}")
            plt.close()

        except ImportError:
            print("WARNING: matplotlib not available for plotting")


def introspect_memory_inject(args):
    """External memory injection for fact retrieval.

    Builds an external memory store from known facts and uses it to
    inject correct answers at inference time. This can rescue queries
    that the model would otherwise get wrong.

    Examples:
        # Build memory from multiplication table and test
        lazarus introspect memory-inject -m openai/gpt-oss-20b
            --facts multiplication --query "7*8="

        # Test rescue on non-standard format
        lazarus introspect memory-inject -m openai/gpt-oss-20b
            --facts multiplication --query "seven times eight equals"

        # Load custom facts and save memory store
        lazarus introspect memory-inject -m openai/gpt-oss-20b
            --facts @my_facts.json --save-store memory.npz
    """
    from ....introspection.external_memory import ExternalMemory, MemoryConfig

    # Configure memory layers
    query_layer = getattr(args, "query_layer", None)
    inject_layer = getattr(args, "inject_layer", None)
    blend = getattr(args, "blend", 1.0)
    threshold = getattr(args, "threshold", 0.7)

    memory_config = None
    if query_layer is not None or inject_layer is not None:
        memory_config = MemoryConfig(
            query_layer=query_layer or 22,
            inject_layer=inject_layer or 21,
            value_layer=query_layer or 22,
            blend=blend,
            similarity_threshold=threshold,
        )

    # Create memory system
    memory = ExternalMemory.from_pretrained(args.model, memory_config)

    # Load facts
    fact_type = args.facts
    if fact_type.startswith("@"):
        # Load from file
        with open(fact_type[1:]) as f:
            facts = json.load(f)
        memory.add_facts(facts)
    elif fact_type == "multiplication":
        memory.add_multiplication_table(2, 9)
    elif fact_type == "addition":
        facts = []
        for a in range(1, 10):
            for b in range(1, 10):
                facts.append({"query": f"{a}+{b}=", "answer": str(a + b)})
        memory.add_facts(facts)
    else:
        print(f"ERROR: Unknown fact type: {fact_type}")
        print("Use: multiplication, addition, or @file.json")
        return

    # Save store if requested
    save_store = getattr(args, "save_store", None)
    if save_store:
        memory.save(save_store)

    # Load store if provided
    load_store = getattr(args, "load_store", None)
    if load_store:
        memory.load(load_store)

    # Process queries
    queries = []
    if hasattr(args, "query") and args.query:
        queries = [args.query]
    elif hasattr(args, "queries") and args.queries:
        queries = args.queries.split("|")

    if not queries:
        print("\nNo queries provided. Use --query or --queries")
        print(f"Memory store has {memory.num_entries} entries")
        return

    print(f"\n{'=' * 70}")
    print("EXTERNAL MEMORY INJECTION")
    print(f"{'=' * 70}")

    force = getattr(args, "force", False)

    for query in queries:
        result = memory.query(query, use_injection=True, force_injection=force)

        print(f"\nQuery: '{query}'")
        print(f"  Baseline: '{result.baseline_answer}' ({result.baseline_confidence:.1%})")

        if result.used_injection:
            print(f"  Injected: '{result.injected_answer}' ({result.injected_confidence:.1%})")
            if result.matched_entry:
                print(
                    f"  Matched:  '{result.matched_entry.query}' -> {result.matched_entry.answer}"
                )
                print(f"  Similarity: {result.similarity:.3f}")

            # Show if it was rescued
            if result.baseline_answer.strip() != result.injected_answer.strip():
                print("  Status: MODIFIED")
        else:
            if result.matched_entry:
                print(f"  Matched:  '{result.matched_entry.query}' (sim={result.similarity:.3f})")
                print(f"  Status: Below threshold ({threshold}), no injection")
            else:
                print("  Status: No match found")

    # Evaluate mode
    if getattr(args, "evaluate", False):
        print(f"\n{'=' * 70}")
        print("EVALUATION")
        print(f"{'=' * 70}")

        # Build test set from the facts
        if fact_type == "multiplication":
            test_facts = [
                {"query": f"{a}*{b}=", "answer": str(a * b)}
                for a in range(2, 10)
                for b in range(2, 10)
            ]
        elif fact_type == "addition":
            test_facts = [
                {"query": f"{a}+{b}=", "answer": str(a + b)}
                for a in range(1, 10)
                for b in range(1, 10)
            ]
        else:
            test_facts = facts

        metrics = memory.evaluate(test_facts, verbose=False)
        print(f"\nBaseline accuracy: {metrics['baseline_accuracy']:.1%}")
        print(f"Injected accuracy: {metrics['injected_accuracy']:.1%}")
        print(f"Rescued: {metrics['rescued']}")
        print(f"Broken: {metrics['broken']}")


__all__ = [
    "introspect_memory",
    "introspect_memory_inject",
]
