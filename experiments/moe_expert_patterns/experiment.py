"""
Expert Analysis Experiment.

Analyzes MoE (Mixture of Experts) expert specialization patterns.

This experiment investigates:
1. Pattern specialization - which experts handle specific token patterns
2. Semantic patterns - expert specialization on semantic relationships
3. Sequence patterns - n-gram and positional specialization
4. Combined analysis - multi-aspect expert behavior

Requires an MoE model (e.g., Mixtral, GPT-MoE variants).
"""

import asyncio
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

from chuk_lazarus.experiments import ExperimentBase, ExperimentConfig

logger = logging.getLogger(__name__)


class ExpertAnalysisExperiment(ExperimentBase):
    """
    MoE expert specialization analysis experiment.

    Analyzes expert routing patterns to understand:
    - Token type specialization
    - Semantic relationship handling
    - Positional patterns
    - Layer-wise evolution
    """

    def setup(self) -> None:
        """Initialize model and load prompts."""
        self.log("Setting up expert analysis...")

        # Get analysis parameters
        self.params = self.config.parameters
        self.num_prompts = self.params.get("num_prompts", 100)
        self.min_activity = self.params.get("min_activity", 5)
        self.specialist_threshold = self.params.get("specialist_threshold", 0.15)

        # We'll load the model lazily when needed
        self.router = None

    def _classify_token(self, token: str) -> str:
        """Classify token by type."""
        clean = token.strip()
        lower = clean.lower()

        if not clean:
            return "WS"

        if re.match(r"^-?\d+\.?\d*$", clean):
            return "NUM"

        code_keywords = {
            "def",
            "class",
            "import",
            "return",
            "if",
            "else",
            "for",
            "while",
            "function",
            "const",
            "let",
            "var",
            "async",
            "await",
            "SELECT",
            "FROM",
            "WHERE",
            "INSERT",
            "CREATE",
            "fn",
            "mut",
            "impl",
            "struct",
            "enum",
        }
        if clean in code_keywords or lower in code_keywords:
            return "KW"

        if clean in "()[]{}":
            return "BR"

        if clean in "+-*/=<>!&|^~" or clean in ["==", "!=", "<=", ">=", "+=", "-=", "->", "=>"]:
            return "OP"

        if re.match(r"^[^\w\s]+$", clean):
            return "PN"

        func_words = {
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "by",
            "of",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "this",
            "that",
        }
        if lower in func_words:
            return "FW"

        if clean and clean[0].isupper():
            return "CAP"

        if len(clean) == 1 and clean.isalpha():
            return "VAR"

        return "CW"

    def _get_semantic_type(self, token: str) -> str:
        """Classify token by semantic type."""
        clean = token.strip().lower()

        if not clean:
            return "WS"

        if re.match(r"^-?\d+\.?\d*$", clean):
            return "NUM"

        if clean in ["+", "-", "*", "/", "=", "<", ">", "==", "!=", "+=", "-="]:
            return "OP"

        if clean in ["same", "similar", "like", "equals", "means"]:
            return "SYN"
        if clean in ["opposite", "contrary", "versus", "against", "opposed"]:
            return "ANT"

        if clean == "as":
            return "AS"
        if clean == "to":
            return "TO"

        if clean in ["the", "a", "an", "of", "is", "are", "was", "were"]:
            return "FUNC"

        adjectives = {
            "happy",
            "sad",
            "hot",
            "cold",
            "big",
            "small",
            "fast",
            "slow",
            "good",
            "bad",
            "old",
            "new",
            "light",
            "dark",
            "high",
            "low",
        }
        if clean in adjectives:
            return "ADJ"

        nouns = {
            "dog",
            "cat",
            "car",
            "tree",
            "book",
            "house",
            "person",
            "animal",
            "king",
            "queen",
            "man",
            "woman",
            "doctor",
            "teacher",
            "student",
        }
        if clean in nouns:
            return "NOUN"

        return "OTHER"

    async def _analyze_pattern_specialists(self, router) -> dict:
        """Find the most specialized experts by pattern."""
        from chuk_lazarus.introspection.moe import get_prompts_flat

        all_prompts = get_prompts_flat()[: self.num_prompts]
        self.log(f"Analyzing {len(all_prompts)} prompts for pattern specialists...")

        expert_trigrams: dict[tuple, Counter] = defaultdict(Counter)
        expert_examples: dict[tuple, list] = defaultdict(list)

        for cat, prompt in all_prompts:
            weights = await router.capture_router_weights(prompt)

            for layer_weights in weights:
                layer_idx = layer_weights.layer_idx
                positions = layer_weights.positions
                types = [self._classify_token(p.token) for p in positions]

                for i, pos in enumerate(positions):
                    prev_t = types[i - 1] if i > 0 else "^"
                    curr_t = types[i]
                    next_t = types[i + 1] if i < len(types) - 1 else "$"
                    trigram = f"{prev_t}→{curr_t}→{next_t}"

                    for exp in pos.expert_indices:
                        key = (layer_idx, exp)
                        expert_trigrams[key][trigram] += 1
                        if len(expert_examples[key]) < 8:
                            expert_examples[key].append((trigram, pos.token, prompt[:30]))

        # Find specialists
        specialists = []
        for (layer, exp), counts in expert_trigrams.items():
            total = sum(counts.values())
            if total < self.min_activity:
                continue

            top_pattern, top_count = counts.most_common(1)[0]
            concentration = top_count / total
            top_3 = counts.most_common(3)

            specialists.append(
                {
                    "layer": layer,
                    "expert": exp,
                    "top_pattern": top_pattern,
                    "concentration": concentration,
                    "top_3": [(p, c) for p, c in top_3],
                    "total": total,
                    "examples": [e[1] for e in expert_examples[(layer, exp)][:5]],
                }
            )

        specialists.sort(key=lambda x: (-x["concentration"], x["layer"]))
        return {"specialists": specialists[:50]}

    async def _analyze_semantic_patterns(self, router) -> dict:
        """Analyze expert specialization on semantic patterns."""
        semantic_prompts = self.config.parameters.get("semantic_prompts", {})

        all_prompts = []
        for category, prompts in semantic_prompts.items():
            for prompt in prompts:
                all_prompts.append((category, prompt))

        if not all_prompts:
            self.log("No semantic prompts configured, skipping...")
            return {}

        self.log(f"Analyzing {len(all_prompts)} semantic prompts...")

        expert_semantic_trigrams: dict[tuple, Counter] = defaultdict(Counter)

        for cat, prompt in all_prompts:
            weights = await router.capture_router_weights(prompt)

            for layer_weights in weights:
                layer = layer_weights.layer_idx
                positions = layer_weights.positions
                sem_types = [self._get_semantic_type(p.token) for p in positions]

                for i, pos in enumerate(positions):
                    prev_t = sem_types[i - 1] if i > 0 else "^"
                    curr_t = sem_types[i]
                    next_t = sem_types[i + 1] if i < len(sem_types) - 1 else "$"
                    trigram = f"{prev_t}→{curr_t}→{next_t}"

                    for exp in pos.expert_indices:
                        key = (layer, exp)
                        expert_semantic_trigrams[key][trigram] += 1

        # Find interesting semantic patterns
        patterns_of_interest = {
            "synonym": "ADJ→SYN",
            "antonym": "ADJ→ANT",
            "arithmetic": "NUM→OP",
            "analogy_as": "→AS→",
            "analogy_to": "→TO→",
            "hypernym": "NOUN→FUNC",
        }

        results = {}
        for name, pattern in patterns_of_interest.items():
            experts = []
            for (layer, exp), counts in expert_semantic_trigrams.items():
                for trigram, count in counts.items():
                    if pattern in trigram:
                        experts.append(
                            {
                                "layer": layer,
                                "expert": exp,
                                "trigram": trigram,
                                "count": count,
                            }
                        )
            experts.sort(key=lambda x: -x["count"])
            results[name] = experts[:10]

        return {"semantic_patterns": results}

    def run(self) -> dict:
        """Run all configured analyses."""
        # Run async analyses
        return asyncio.run(self._run_async())

    async def _run_async(self) -> dict:
        """Async implementation of run."""
        from chuk_lazarus.introspection.moe import ExpertRouter

        self.log(f"Loading model: {self.config.model}")

        async with await ExpertRouter.from_pretrained(self.config.model) as router:
            info = router.info
            self.log(f"Model loaded: {info.num_experts} experts, {len(info.moe_layers)} MoE layers")

            results = {
                "model": self.config.model,
                "num_experts": info.num_experts,
                "num_moe_layers": len(info.moe_layers),
            }

            analyses = self.config.parameters.get("analyses", ["pattern_summary"])

            if "pattern_summary" in analyses:
                self.log("Running pattern specialist analysis...")
                results["pattern_specialists"] = await self._analyze_pattern_specialists(router)

            if "semantic_patterns" in analyses or "combined_analysis" in analyses:
                self.log("Running semantic pattern analysis...")
                results["semantic_patterns"] = await self._analyze_semantic_patterns(router)

            return results

    def evaluate(self) -> dict:
        """Summarize analysis results."""
        latest = self.load_latest_results("results")
        if not latest:
            return {"error": "No results to evaluate"}

        summary = {
            "model": latest.get("model"),
            "num_experts": latest.get("num_experts"),
            "num_moe_layers": latest.get("num_moe_layers"),
        }

        # Summarize pattern specialists
        if "pattern_specialists" in latest:
            specialists = latest["pattern_specialists"].get("specialists", [])
            summary["num_specialists_found"] = len(specialists)
            if specialists:
                summary["top_specialist"] = {
                    "layer": specialists[0]["layer"],
                    "expert": specialists[0]["expert"],
                    "pattern": specialists[0]["top_pattern"],
                    "concentration": specialists[0]["concentration"],
                }

        # Summarize semantic patterns
        if "semantic_patterns" in latest:
            sem = latest["semantic_patterns"].get("semantic_patterns", {})
            summary["semantic_pattern_counts"] = {
                name: len(experts) for name, experts in sem.items()
            }

        return summary

    def cleanup(self) -> None:
        """Release resources."""
        self.log("Cleaning up...")
        self.router = None


# For backwards compatibility
if __name__ == "__main__":
    import sys

    import yaml

    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    # Allow model override from command line
    if len(sys.argv) > 1:
        config_data["model"] = sys.argv[1]

    config = ExperimentConfig(
        experiment_dir=Path(__file__).parent,
        **config_data,
    )

    experiment = ExpertAnalysisExperiment(config)
    experiment.setup()
    results = experiment.run()
    eval_results = experiment.evaluate()
    experiment.cleanup()

    print(f"\nSummary: {json.dumps(eval_results, indent=2)}")
