#!/usr/bin/env python3
"""
Spec-based Generator for Training Data.

Generates technical specifications that can be expanded to natural language by an LLM.
Designed to be expert-agnostic (works for arithmetic, time, weather, etc.).

Flow:
    1. Schema → Technical Spec (structured, machine-readable)
    2. Spec + Domain → LLM → Natural Language Question
    3. Verify answer matches formula

Usage:
    from spec_generator import SpecGenerator

    gen = SpecGenerator()
    spec = gen.generate_spec("combined_rate")
    print(spec)
    # {
    #   "schema": "combined_rate",
    #   "domain": "kitchen",
    #   "agent1": "Sarah",
    #   "agent2": "Mike",
    #   "item": "cookies",
    #   "rate1": 12,
    #   "rate2": 8,
    #   "time": 3,
    #   "time_unit": "hour",
    #   "formula": "(rate1 + rate2) * time",
    #   "answer": 60,
    #   "trace": [...]
    # }
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


class SpecGenerator:
    """Generates technical specifications from schemas + domains."""

    def __init__(self):
        self._schemas = self._load_schemas()
        self._domains = self._load_domains()
        self._shared_vocab = self._load_shared_vocab()

    def _load_schemas(self) -> dict[str, dict]:
        """Load schemas from the schemas directory."""
        schemas = {}
        candidates = [
            Path("/Users/christopherhay/chris-source/chuk-ai/virtual-experts/packages/chuk-virtual-expert-arithmetic/src/chuk_virtual_expert_arithmetic/schemas"),
            Path(__file__).parent / "schemas",
        ]

        for schema_dir in candidates:
            if schema_dir.exists():
                for subdir in schema_dir.iterdir():
                    if subdir.is_dir():
                        for schema_file in subdir.glob("*.json"):
                            with open(schema_file) as f:
                                schema = json.load(f)
                                name = schema.get("name", schema_file.stem)
                                schemas[name] = schema
                break

        return schemas

    def _load_domains(self) -> dict[str, dict]:
        """Load domains from the domains directory."""
        domains = {}
        candidates = [
            Path("/Users/christopherhay/chris-source/chuk-ai/virtual-experts/packages/chuk-virtual-expert-arithmetic/src/chuk_virtual_expert_arithmetic/vocab/domains"),
            Path(__file__).parent / "domains",
        ]

        for domains_dir in candidates:
            if domains_dir.exists():
                for domain_file in domains_dir.glob("*.json"):
                    with open(domain_file) as f:
                        domain = json.load(f)
                        name = domain.get("name", domain_file.stem)
                        domains[name] = domain
                break

        return domains

    def _load_shared_vocab(self) -> dict[str, Any]:
        """Load shared vocabulary (names, items, etc.)."""
        vocab = {}
        candidates = [
            Path("/Users/christopherhay/chris-source/chuk-ai/virtual-experts/packages/chuk-virtual-expert-arithmetic/src/chuk_virtual_expert_arithmetic/vocab"),
            Path(__file__).parent / "vocab",
        ]

        for vocab_dir in candidates:
            if vocab_dir.exists():
                # Load top-level JSON files
                for json_file in vocab_dir.glob("*.json"):
                    key = json_file.stem
                    with open(json_file) as f:
                        vocab[key] = json.load(f)
                break

        return vocab

    @property
    def schema_names(self) -> list[str]:
        return list(self._schemas.keys())

    @property
    def domain_names(self) -> list[str]:
        return list(self._domains.keys())

    def _resolve_vocab_path(self, path: str) -> Any:
        """Resolve a dot-separated vocab path like 'names.people'."""
        parts = path.split(".")
        data: Any = self._shared_vocab
        for part in parts:
            if isinstance(data, dict):
                data = data.get(part)
            else:
                return None
            if data is None:
                return None
        return data

    def _generate_agent_name(self, template: dict) -> str:
        """Generate an agent name from a template spec."""
        # If template has a source, sample from shared vocab
        if "source" in template:
            vocab_list = self._resolve_vocab_path(template["source"])
            if vocab_list and isinstance(vocab_list, list):
                name = random.choice(vocab_list)
                pattern = template.get("pattern", "${name}")
                return pattern.replace("${name}", name)
            return "Agent"

        # Otherwise use pattern with letters/numbers
        pattern = template.get("pattern", "Agent ${letter}")

        if "letters" in template:
            letter = random.choice(template["letters"])
            return pattern.replace("${letter}", letter)
        elif "numbers" in template:
            number = random.choice(template["numbers"])
            return pattern.replace("${number}", str(number))

        return pattern

    def _generate_agent_pair(self, domain: dict) -> tuple[str, str]:
        """Generate a pair of distinct agent names from domain templates."""
        templates = domain.get("agent_templates", {})
        agent_types = domain.get("agent_types", list(templates.keys()))

        if not templates or not agent_types:
            return ("Agent A", "Agent B")

        # Pick an agent type for consistency (both agents same type)
        agent_type = random.choice(agent_types)
        template = templates.get(agent_type, templates.get(list(templates.keys())[0], {}))

        # Generate two distinct agents
        if "source" in template:
            # Sample two different names from vocab
            vocab_list = self._resolve_vocab_path(template["source"])
            if vocab_list and isinstance(vocab_list, list) and len(vocab_list) >= 2:
                names = random.sample(vocab_list, 2)
                pattern = template.get("pattern", "${name}")
                return (
                    pattern.replace("${name}", names[0]),
                    pattern.replace("${name}", names[1])
                )
        else:
            # Use pattern with different letters/numbers
            pattern = template.get("pattern", "Agent ${letter}")
            if "letters" in template and len(template["letters"]) >= 2:
                letters = random.sample(template["letters"], 2)
                return (
                    pattern.replace("${letter}", letters[0]),
                    pattern.replace("${letter}", letters[1])
                )
            elif "numbers" in template and len(template["numbers"]) >= 2:
                numbers = random.sample(template["numbers"], 2)
                return (
                    pattern.replace("${number}", str(numbers[0])),
                    pattern.replace("${number}", str(numbers[1]))
                )

        return ("Agent A", "Agent B")

    def generate_spec(
        self,
        schema_name: str,
        domain_name: str | None = None,
    ) -> dict[str, Any]:
        """Generate a technical specification.

        Args:
            schema_name: Name of the schema to use
            domain_name: Domain for vocab (None = random)

        Returns:
            Technical specification dict
        """
        schema = self._schemas.get(schema_name)
        if not schema:
            raise ValueError(f"Unknown schema: {schema_name}")

        # Pick domain
        if domain_name is None:
            domain_name = random.choice(list(self._domains.keys()))
        domain = self._domains.get(domain_name, {})

        # Generate variables (only numeric ones from schema) with constraints
        variables = self._generate_variables(
            schema.get("variables", {}),
            schema.get("constraints", {})
        )

        # Compute derived variables if any
        derived = self._compute_derived(schema.get("derived", {}), variables)

        # Generate agent pair from domain templates
        agent1, agent2 = self._generate_agent_pair(domain)

        # Sample other domain vocab
        item = random.choice(domain.get("items", ["items"]))
        time_unit = random.choice(domain.get("time_units", [{"singular": "hour", "plural": "hours"}]))
        verbs = domain.get("verbs", {"singular": "produces", "plural": "produce"})

        # Compute answer (using both variables and derived)
        answer = self._compute_answer(schema.get("answer", "0"), variables, derived)

        # Build spec - order matters!
        # 1. Base numeric variables and derived go first
        # 2. Domain vocab goes AFTER to override any schema-derived conflicts
        spec = {
            "schema": schema_name,
            "domain": domain_name,
            "description": schema.get("description", ""),

            # Numeric values first
            **variables,

            # Derived values (computed from variables) - may include legacy 'item' etc.
            **{k: v for k, v in derived.items() if k not in ['item', 'time_unit', 'time_units']},

            # Domain-coherent vocab OVERRIDES any schema-derived conflicts
            "agent1": agent1,
            "agent2": agent2,
            "item": item,
            "time_unit": time_unit["singular"],
            "time_units": time_unit["plural"],
            "verb": verbs.get("plural", "produce"),
            "verbs": verbs.get("singular", "produces"),

            # Formula and answer
            "formula": schema.get("answer", ""),
            "answer": answer,

            # Trace for verification
            "trace": schema.get("trace", []),
        }

        return spec

    def generate_batch(
        self,
        schema_names: list[str] | None = None,
        n: int = 10,
        balance_domains: bool = True,
    ) -> list[dict[str, Any]]:
        """Generate multiple specs.

        Args:
            schema_names: Schemas to use (None = all)
            n: Number of specs
            balance_domains: If True, distribute evenly across domains
        """
        if schema_names is None:
            schema_names = self.schema_names

        specs = []
        domain_list = list(self._domains.keys())

        for i in range(n):
            schema_name = random.choice(schema_names)

            if balance_domains:
                domain_name = domain_list[i % len(domain_list)]
            else:
                domain_name = None

            specs.append(self.generate_spec(schema_name, domain_name))

        return specs

    def _generate_variables(
        self, var_specs: dict, constraints: dict = None, max_attempts: int = 100
    ) -> dict[str, Any]:
        """Generate random variable values that satisfy constraints."""
        for _ in range(max_attempts):
            variables = {}
            for name, spec in var_specs.items():
                var_type = spec.get("type", "int")
                if var_type == "int":
                    variables[name] = random.randint(
                        spec.get("min", 1),
                        spec.get("max", 100)
                    )
                elif var_type == "float":
                    variables[name] = round(
                        random.uniform(spec.get("min", 0), spec.get("max", 10)),
                        spec.get("precision", 2)
                    )
                elif var_type == "choice":
                    options = spec.get("options") or spec.get("values", [1])
                    variables[name] = random.choice(options)

            # Check constraints
            if constraints and not self._check_constraints(variables, constraints):
                continue

            return variables

        # If we couldn't satisfy constraints, return last attempt with warning
        return variables

    def _check_constraints(self, variables: dict, constraints: dict) -> bool:
        """Check if variables satisfy all constraints."""
        for expr, constraint in constraints.items():
            try:
                value = eval(expr, {"__builtins__": {}}, variables)
                min_val = constraint.get("min")
                max_val = constraint.get("max")
                if min_val is not None and value < min_val:
                    return False
                if max_val is not None and value > max_val:
                    return False
            except:
                return False
        return True

    def _compute_derived(self, derived_specs: dict, variables: dict) -> dict:
        """Compute derived variables from formulas."""
        derived = {}
        context = dict(variables)
        for name, expr in derived_specs.items():
            try:
                value = eval(expr, {"__builtins__": {}}, context)
                derived[name] = value
                context[name] = value
            except:
                derived[name] = 0
        return derived

    def _compute_answer(self, expr: str, variables: dict, derived: dict = None) -> float:
        """Compute answer from formula."""
        try:
            context = dict(variables)
            if derived:
                context.update(derived)
            return float(eval(expr, {"__builtins__": {}}, context))
        except:
            return 0.0

    def spec_to_prompt(self, spec: dict) -> str:
        """Convert spec to LLM expansion prompt."""
        return f'''Convert this technical specification into a natural, GSM-8K style word problem:

SPECIFICATION:
- Domain: {spec["domain"]}
- Agent 1: {spec["agent1"]}
- Agent 2: {spec["agent2"]}
- Item: {spec["item"]}
- Rate 1: {spec.get("rate1", "?")} {spec["item"]} per {spec["time_unit"]}
- Rate 2: {spec.get("rate2", "?")} {spec["item"]} per {spec["time_unit"]}
- Duration: {spec.get("time", "?")} {spec["time_units"]}
- Operation: {spec["formula"]}
- Answer: {spec["answer"]}

REQUIREMENTS:
1. Write a natural word problem that a human would write
2. Include context and narrative (who, why, where)
3. The answer must be exactly {spec["answer"]}
4. Use the exact agents and items from the spec
5. Make it sound like a real-world scenario

OUTPUT: Just the word problem, nothing else.'''


def demo():
    """Demo the spec generator."""
    gen = SpecGenerator()

    print("=" * 70)
    print("AVAILABLE SCHEMAS:", gen.schema_names[:10], "...")
    print("AVAILABLE DOMAINS:", gen.domain_names)
    print("=" * 70)

    # Generate specs for various domains
    for domain in gen.domain_names[:5]:
        print(f"\n### Domain: {domain} ###")
        spec = gen.generate_spec("combined_rate", domain)
        print(f"  Agents: {spec['agent1']} + {spec['agent2']}")
        print(f"  Rates: {spec.get('rate1', '?')} + {spec.get('rate2', '?')} {spec['item']}/{spec['time_unit']}")
        print(f"  Duration: {spec.get('time', '?')} {spec['time_units']}")
        print(f"  Answer: {spec['answer']}")


if __name__ == "__main__":
    demo()
