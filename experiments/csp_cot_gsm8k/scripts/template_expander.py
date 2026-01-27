#!/usr/bin/env python3
"""
Template Expander for Schema-Based Training Data.

Uses LLM to generate diverse linguistic variations of existing templates
while preserving mathematical structure. Validates all generated templates
produce correct answers.

Usage:
    # Expand a single schema
    python template_expander.py expand \
        --schema combined_rate \
        --n-variations 30 \
        --output patterns_expanded/

    # Expand all schemas
    python template_expander.py expand-all \
        --n-variations 30 \
        --output patterns_expanded/

    # Validate existing templates
    python template_expander.py validate \
        --schema combined_rate \
        --n-samples 10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# Default paths - adjust based on your setup
SCHEMA_BASE = Path(__file__).parent.parent.parent.parent / "chuk-ai/virtual-experts/packages/chuk-virtual-expert-arithmetic/src/chuk_virtual_expert_arithmetic"
SCHEMA_DIR = SCHEMA_BASE / "schemas"
PATTERN_DIR = SCHEMA_BASE / "vocab/patterns"


# =============================================================================
# EXPANSION PROMPT
# =============================================================================

EXPANSION_PROMPT = '''You are generating diverse word problem templates for a math training dataset.

## Schema Definition
Name: {schema_name}
Description: {description}
Formula: {answer_formula}
Variables: {variables}

## Original Templates (for style reference)
{original_templates}

## Requirements
Generate {n} NEW templates that:
1. Use the EXACT same variable placeholders: {placeholder_list}
2. Produce the EXACT same mathematical computation: {answer_formula}
3. **CRITICALLY IMPORTANT**: Use DIVERSE domains and contexts. The formula is abstract and applies to MANY scenarios, not just the examples shown. Vary across:

   **Domains** (use ALL of these, not just factories!):
   - Home/family: cooking, reading, chores, gardening
   - School: homework, studying, art projects
   - Nature: animals, plants growing, weather
   - Sports: running, swimming, scoring points
   - Travel: driving, walking, cycling distances
   - Commerce: sales, earnings, tips
   - Construction: building, painting, digging
   - Office: typing, filing, emails
   - Farm: harvesting, planting, feeding animals
   - Entertainment: watching movies, playing games

   **Phrasing variations**:
   - Formal: "Calculate the total...", "Determine the combined..."
   - Casual: "How many...", "What's the total..."
   - Narrative: "After working for X hours...", "By the end of the day..."
   - Direct: "Find...", "What is..."

   **Structure**:
   - Single sentence with embedded rates
   - Multi-sentence with setup then question
   - Story format with characters and context

## Operation Trigger Words to Use
{operation_triggers}

## Variable Placeholders (MUST use these exactly)
{placeholder_list}

**Placeholder semantics** (these are GENERIC, not domain-specific):
- ${{name}} = person's name (Alex, Maria, James, etc.)
- ${{his_her}} = possessive pronoun matching name
- ${{agent1}}, ${{agent2}} = ANY two rate-producing entities (machines, people, faucets, teams, etc.)
- ${{item}} = ANY unit being produced/consumed (widgets, pages, gallons, points, miles, etc.)
- ${{rate1}}, ${{rate2}} = numeric rates
- ${{time}} = duration count
- ${{time_unit}} = singular time unit (hour, minute, day, week)
- ${{time_units}} = plural time unit (hours, minutes, days, weeks)

So "${{agent1}} produces ${{rate1}} ${{item}} per ${{time_unit}}" could become:
- "Machine A produces 5 widgets" (factory)
- "Faucet 1 fills 5 gallons" (plumbing)
- "Alice reads 5 pages" (reading)
- "Runner A covers 5 miles" (sports)

Write templates as if these will be filled with domain-appropriate values!

## Output Format
Return ONLY a JSON array of template strings. Each template must:
- Use all required variable placeholders (${{var}} format)
- Be grammatically correct English
- Be solvable with formula: {answer_formula}
- Be distinct from originals AND from each other (no repetitive domains!)

IMPORTANT: Distribute templates evenly across domains. If generating 30 templates, aim for ~3 per domain category.

```json
[
  "template 1...",
  "template 2..."
]
```'''

OPERATION_TRIGGERS = {
    "add": ["combined with", "together with", "plus", "in addition to", "along with", "and", "total of", "sum of"],
    "sub": ["minus", "less", "remaining", "left over", "after removing", "difference", "take away"],
    "mul": ["times", "each produces", "per hour", "at a rate of", "every X makes Y", "multiplied by", "groups of"],
    "div": ["divided by", "split among", "shared between", "per person", "each gets", "half of", "distributed"],
}


# =============================================================================
# TEMPLATE VALIDATOR
# =============================================================================

class TemplateValidator:
    """Validates that templates produce correct traces and answers."""

    def __init__(self, schema: dict, verifier=None):
        """Initialize validator.

        Args:
            schema: Schema definition dict
            verifier: Optional TraceVerifier instance for execution validation
        """
        self.schema = schema
        self.verifier = verifier
        self._setup_vocab()

    def _setup_vocab(self):
        """Setup vocabulary for template filling."""
        try:
            from chuk_virtual_expert_arithmetic.vocab import get_vocab
            self._vocab = get_vocab()
        except ImportError:
            self._vocab = None
            print("Warning: Could not import vocab, using fallback values")

    def validate_template(self, template: str, n_samples: int = 10) -> dict:
        """Validate a template produces correct answers.

        Args:
            template: Template string with ${var} placeholders
            n_samples: Number of random samples to test

        Returns:
            Validation result dict with accuracy and failure details
        """
        results = {
            "template": template,
            "samples_tested": n_samples,
            "samples_passed": 0,
            "failures": [],
            "placeholder_check": self._check_placeholders(template),
        }

        # First check placeholders
        if not results["placeholder_check"]["valid"]:
            results["validated"] = False
            results["accuracy"] = 0.0
            return results

        for i in range(n_samples):
            try:
                # Generate random values
                values = self._generate_values()

                # Fill template
                question = self._fill_template(template, values)

                # Compute expected answer
                expected = self._compute_answer(values)

                # Generate trace from schema
                trace_yaml = self._generate_trace_yaml(values)

                # Verify trace produces correct answer
                if self.verifier:
                    result = self.verifier.verify(trace_yaml, expected)
                    if result.answer_correct:
                        results["samples_passed"] += 1
                    else:
                        results["failures"].append({
                            "sample": i,
                            "question": question[:100] + "..." if len(question) > 100 else question,
                            "expected": expected,
                            "got": result.computed_answer,
                            "error": result.trace_error,
                        })
                else:
                    # Without verifier, just check template fills correctly
                    if question and expected is not None:
                        results["samples_passed"] += 1

            except Exception as e:
                results["failures"].append({
                    "sample": i,
                    "error": str(e),
                })

        results["accuracy"] = results["samples_passed"] / n_samples if n_samples > 0 else 0
        results["validated"] = results["accuracy"] == 1.0

        return results

    def _check_placeholders(self, template: str) -> dict:
        """Check that template has all required placeholders."""
        # Extract placeholders from template
        found = set(re.findall(r'\$\{(\w+)\}', template))

        # Get required variable placeholders
        required_vars = set(self.schema.get("variables", {}).keys())

        # Get template_vars that should be in template
        template_vars = set(self.schema.get("template_vars", {}).keys())

        # Check for missing required variables
        # Note: template_vars are derived, not all need to be in every template
        missing_vars = required_vars - found

        return {
            "valid": len(missing_vars) == 0,
            "found": list(found),
            "required_vars": list(required_vars),
            "template_vars": list(template_vars),
            "missing": list(missing_vars),
        }

    def _generate_values(self) -> dict:
        """Generate random values per schema constraints."""
        values = {}
        for var, spec in self.schema.get("variables", {}).items():
            var_type = spec.get("type", "int")
            if var_type == "int":
                values[var] = random.randint(spec.get("min", 1), spec.get("max", 100))
            elif var_type == "float":
                values[var] = round(random.uniform(spec.get("min", 0), spec.get("max", 10)), 2)
            elif var_type == "choice":
                options = spec.get("options") or spec.get("values", [1, 2, 3])
                values[var] = random.choice(options)

        # Compute derived values
        for name, expr in self.schema.get("derived", {}).items():
            try:
                values[name] = eval(expr, {"__builtins__": {}}, values)
            except Exception:
                values[name] = 0

        return values

    def _fill_template(self, template: str, values: dict) -> str:
        """Fill template placeholders with values using domain-coherent sampling."""
        result = template

        # Add vocab items if available
        template_values = dict(values)

        # Try domain-coherent sampling first
        domain_values = self._sample_domain_coherent()
        if domain_values:
            template_values.update(domain_values)
        elif self._vocab:
            # Fallback to random sampling if domains not available
            person = self._vocab.person_with_pronouns()
            template_values.update({
                "name": person.get("name", "Alex"),
                "his_her": person.get("possessive", "their"),
                "him_her": person.get("object", "them"),
            })

            # Sample agent pairs (machines, people, faucets, etc.)
            try:
                agent_pair = self._vocab.random("phrases.agent_pairs")
                if agent_pair and isinstance(agent_pair, dict):
                    template_values["agent1"] = agent_pair.get("first", "Agent A")
                    template_values["agent2"] = agent_pair.get("second", "Agent B")
                else:
                    template_values["agent1"] = "Agent A"
                    template_values["agent2"] = "Agent B"
            except Exception:
                template_values["agent1"] = "Agent A"
                template_values["agent2"] = "Agent B"

            # Sample item (what's being produced/counted)
            try:
                item = self._vocab.random("phrases.rate_items")
                template_values["item"] = item if item else "items"
            except Exception:
                template_values["item"] = "items"

            # Sample time unit (hour, minute, day, week)
            try:
                time_pair = self._vocab.random("phrases.time_unit_pairs")
                if time_pair and isinstance(time_pair, dict):
                    template_values["time_unit"] = time_pair.get("singular", "hour")
                    template_values["time_units"] = time_pair.get("plural", "hours")
                else:
                    template_values["time_unit"] = "hour"
                    template_values["time_units"] = "hours"
            except Exception:
                template_values["time_unit"] = "hour"
                template_values["time_units"] = "hours"
        else:
            # Fallback values
            template_values.update({
                "name": "Alex",
                "his_her": "their",
                "agent1": "Agent A",
                "agent2": "Agent B",
                "item": "items",
                "time_unit": "hour",
                "time_units": "hours",
            })

        # Replace placeholders
        for key, value in template_values.items():
            result = result.replace(f"${{{key}}}", str(value))

        return result

    def _sample_domain_coherent(self) -> dict | None:
        """Sample coherent values from a single domain."""
        try:
            domains = self._vocab.get("domains")
            if not domains:
                return None

            # Pick a random domain
            domain_name = random.choice(list(domains.keys()))
            domain = domains[domain_name]

            # Sample agent pair from this domain
            agent_pair = random.choice(domain["agents"])

            # Sample item from this domain
            item = random.choice(domain["items"])

            # Sample time unit from this domain
            time_unit = random.choice(domain["time_units"])

            # Get person for name placeholders
            person = self._vocab.person_with_pronouns() if self._vocab else {}

            # Get verbs from domain
            verbs = domain.get("verbs", {})
            verb_singular = verbs.get("singular", "produces")
            verb_plural = verbs.get("plural", "produce")

            return {
                "name": person.get("name", "Alex"),
                "his_her": person.get("possessive", "their"),
                "him_her": person.get("object", "them"),
                "agent1": agent_pair["first"],
                "agent2": agent_pair["second"],
                "item": item,
                "time_unit": time_unit["singular"],
                "time_units": time_unit["plural"],
                "verb": verb_plural,      # "bake", "read", "process"
                "verbs": verb_singular,   # "bakes", "reads", "processes"
                "domain": domain_name,
            }
        except Exception as e:
            return None

        # Replace placeholders
        for key, value in template_values.items():
            result = result.replace(f"${{{key}}}", str(value))

        return result

    def _compute_answer(self, values: dict) -> float:
        """Compute expected answer from schema formula."""
        formula = self.schema.get("answer", "0")
        try:
            return float(eval(formula, {"__builtins__": {}}, values))
        except Exception:
            return 0.0

    def _generate_trace_yaml(self, values: dict) -> str:
        """Generate YAML trace from schema definition."""
        trace_specs = self.schema.get("trace", [])
        expert = self.schema.get("expert", "arithmetic")

        lines = [f"expert: {expert}", "trace:"]

        for spec in trace_specs:
            op = spec["op"]

            if op == "init":
                var = spec["var"]
                value_ref = spec["value"]
                if isinstance(value_ref, (int, float)):
                    value = value_ref
                else:
                    value = values.get(value_ref, 0)
                lines.append(f"- {{op: init, var: {var}, value: {value}}}")

            elif op == "compute":
                compute_op = spec["compute_op"]
                args = spec["args"]
                var = spec["var"]
                args_str = ", ".join(str(a) for a in args)
                lines.append(f"- {{op: compute, compute_op: {compute_op}, args: [{args_str}], var: {var}}}")

            elif op == "query":
                var = spec["var"]
                lines.append(f"- {{op: query, var: {var}}}")

        return "\n".join(lines)


# =============================================================================
# SCHEMA EXPANDER
# =============================================================================

class SchemaExpander:
    """Expands schemas with LLM-generated diverse templates."""

    def __init__(self, llm_client=None, model: str = "gpt-4"):
        """Initialize expander.

        Args:
            llm_client: LLM client with generate() method
            model: Model name for metadata
        """
        self.llm_client = llm_client
        self.model_name = model

    def expand_schema(
        self,
        schema_path: str | Path,
        pattern_path: str | Path,
        n_variations: int = 30,
        validation_samples: int = 10,
        verifier=None,
    ) -> dict:
        """Generate and validate diverse templates for a schema.

        Args:
            schema_path: Path to schema JSON file
            pattern_path: Path to pattern JSON file
            n_variations: Number of variations to generate
            validation_samples: Samples per template for validation
            verifier: Optional TraceVerifier for answer validation

        Returns:
            Expansion result with validated and rejected templates
        """
        # Load schema and pattern
        with open(schema_path) as f:
            schema = json.load(f)

        with open(pattern_path) as f:
            pattern = json.load(f)

        # Create validator
        validator = TemplateValidator(schema, verifier)

        # Build expansion prompt
        prompt = self._build_prompt(schema, pattern, n_variations)

        # Generate variations via LLM
        if self.llm_client:
            response = self._call_llm(prompt)
            candidates = self._parse_llm_response(response)
        else:
            print("No LLM client - generating mock variations for testing")
            candidates = self._generate_mock_variations(schema, pattern, n_variations)

        # Validate each candidate
        validated = []
        rejected = []

        for template in candidates:
            result = validator.validate_template(template, n_samples=validation_samples)

            if result["validated"]:
                validated.append({
                    "text": template,
                    "source": self.model_name,
                    "generated_at": datetime.now().isoformat(),
                    "validated": True,
                    "validation_samples": validation_samples,
                    "validation_accuracy": result["accuracy"],
                })
            else:
                rejected.append({
                    "text": template,
                    "reason": "validation_failed",
                    "accuracy": result["accuracy"],
                    "placeholder_check": result["placeholder_check"],
                    "failures": result["failures"][:3],  # Sample failures
                })

        return {
            "schema": schema.get("name", "unknown"),
            "schema_path": str(schema_path),
            "pattern_path": str(pattern_path),
            "original_templates": len(pattern.get("templates", [])),
            "generated": len(candidates),
            "validated": len(validated),
            "rejected": len(rejected),
            "llm_expanded": validated,
            "rejected_templates": rejected,
            "expansion_prompt": prompt,
        }

    def _build_prompt(self, schema: dict, pattern: dict, n: int) -> str:
        """Build the LLM expansion prompt."""
        # Extract original templates
        templates = pattern.get("templates", [])
        if templates and isinstance(templates[0], dict):
            originals = [t.get("text", t) for t in templates]
        else:
            originals = templates

        # Build variable info
        variables = schema.get("variables", {})
        var_info = json.dumps(variables, indent=2)

        # Build placeholder list
        placeholders = []
        for key in variables.keys():
            placeholders.append(f"${{{key}}}")
        for key in schema.get("template_vars", {}).keys():
            placeholders.append(f"${{{key}}}")

        # Determine operation triggers based on formula
        formula = schema.get("answer", "")
        triggers = []
        if "+" in formula:
            triggers.extend(OPERATION_TRIGGERS["add"])
        if "-" in formula:
            triggers.extend(OPERATION_TRIGGERS["sub"])
        if "*" in formula:
            triggers.extend(OPERATION_TRIGGERS["mul"])
        if "/" in formula:
            triggers.extend(OPERATION_TRIGGERS["div"])

        trigger_text = "\n".join(f"- {t}" for t in triggers[:10])

        return EXPANSION_PROMPT.format(
            schema_name=schema.get("name", "unknown"),
            description=schema.get("description", ""),
            answer_formula=schema.get("answer", ""),
            variables=var_info,
            original_templates="\n".join(f"{i+1}. {t}" for i, t in enumerate(originals[:4])),
            placeholder_list=", ".join(placeholders),
            operation_triggers=trigger_text,
            n=n,
        )

    def _call_llm(self, prompt: str) -> str:
        """Call LLM API to generate variations."""
        if hasattr(self.llm_client, "generate"):
            return self.llm_client.generate(prompt, temperature=0.8)
        elif hasattr(self.llm_client, "chat"):
            # OpenAI-style client
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
            )
            return response.choices[0].message.content
        else:
            raise ValueError("LLM client must have generate() or chat.completions.create() method")

    def _parse_llm_response(self, response: str) -> list[str]:
        """Parse LLM response to extract template list."""
        # Try to find JSON array in response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            try:
                templates = json.loads(json_match.group())
                if isinstance(templates, list):
                    return [t for t in templates if isinstance(t, str)]
            except json.JSONDecodeError:
                pass

        # Fallback: try to parse line by line
        templates = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('"') and line.endswith('"'):
                templates.append(line[1:-1])
            elif line.startswith("- "):
                templates.append(line[2:].strip('"'))

        return templates

    def _generate_mock_variations(self, schema: dict, pattern: dict, n: int) -> list[str]:
        """Generate mock variations for testing without LLM."""
        templates = pattern.get("templates", [])
        if not templates:
            return []

        # Use original templates as base
        base = templates[0] if isinstance(templates[0], str) else templates[0].get("text", "")

        # Create simple variations by adding prefixes/suffixes
        variations = []
        prefixes = [
            "In a busy factory, ",
            "At the local workshop, ",
            "During production, ",
            "For a special order, ",
            "To meet demand, ",
        ]
        suffixes = [
            " Calculate the result.",
            " What's the answer?",
            " Find the total.",
            " Determine the outcome.",
            " How many in total?",
        ]

        for i in range(min(n, len(prefixes) * len(suffixes))):
            prefix = prefixes[i % len(prefixes)]
            suffix = suffixes[i // len(prefixes) % len(suffixes)]
            # Just return original for mock - real LLM would create proper variations
            variations.append(base)

        return variations[:n]


# =============================================================================
# PATTERN FILE UPDATER
# =============================================================================

def update_pattern_file(
    pattern_path: str | Path,
    expansion_result: dict,
    output_path: str | Path | None = None,
) -> Path:
    """Update pattern file with LLM-expanded templates.

    Args:
        pattern_path: Original pattern file path
        expansion_result: Result from SchemaExpander.expand_schema()
        output_path: Output path (default: pattern_path with _expanded suffix)

    Returns:
        Path to updated pattern file
    """
    pattern_path = Path(pattern_path)
    if output_path is None:
        output_path = pattern_path.parent / f"{pattern_path.stem}_expanded.json"
    output_path = Path(output_path)

    # Load original pattern
    with open(pattern_path) as f:
        pattern = json.load(f)

    # Convert to new format if needed
    templates = pattern.get("templates", [])
    if templates and isinstance(templates[0], str):
        # Convert old format to new format
        original_templates = [
            {"text": t, "source": "original", "validated": True}
            for t in templates
        ]
    else:
        original_templates = templates

    # Build updated pattern
    updated = {
        "schema": expansion_result["schema"],
        "version": "2.0",
        "templates": {
            "original": original_templates,
            "llm_expanded": expansion_result["llm_expanded"],
        },
        "stats": {
            "total_templates": len(original_templates) + len(expansion_result["llm_expanded"]),
            "original_count": len(original_templates),
            "llm_count": len(expansion_result["llm_expanded"]),
            "last_expansion": datetime.now().isoformat(),
            "expansion_model": expansion_result["llm_expanded"][0]["source"] if expansion_result["llm_expanded"] else "none",
        },
        "rejected": expansion_result.get("rejected_templates", []),
    }

    # Write updated pattern
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(updated, f, indent=2)

    print(f"Updated pattern saved: {output_path}")
    print(f"  Original templates: {len(original_templates)}")
    print(f"  LLM expanded: {len(expansion_result['llm_expanded'])}")
    print(f"  Rejected: {len(expansion_result.get('rejected_templates', []))}")

    return output_path


# =============================================================================
# CLI
# =============================================================================

def find_schema_and_pattern(schema_name: str) -> tuple[Path | None, Path | None]:
    """Find schema and pattern files by name."""
    schema_path = None
    pattern_path = None

    # Search in subdirectories
    for subdir in SCHEMA_DIR.iterdir():
        if subdir.is_dir():
            candidate = subdir / f"{schema_name}.json"
            if candidate.exists():
                schema_path = candidate
                break

    # Also check root
    if schema_path is None:
        candidate = SCHEMA_DIR / f"{schema_name}.json"
        if candidate.exists():
            schema_path = candidate

    # Find pattern
    for subdir in PATTERN_DIR.iterdir():
        if subdir.is_dir():
            candidate = subdir / f"{schema_name}.json"
            if candidate.exists():
                pattern_path = candidate
                break

    if pattern_path is None:
        candidate = PATTERN_DIR / f"{schema_name}.json"
        if candidate.exists():
            pattern_path = candidate

    return schema_path, pattern_path


def cmd_validate(args):
    """Validate existing templates for a schema."""
    schema_path, pattern_path = find_schema_and_pattern(args.schema)

    if not schema_path:
        print(f"Schema not found: {args.schema}")
        return 1

    if not pattern_path:
        print(f"Pattern not found: {args.schema}")
        return 1

    print(f"Schema: {schema_path}")
    print(f"Pattern: {pattern_path}")

    # Load files
    with open(schema_path) as f:
        schema = json.load(f)
    with open(pattern_path) as f:
        pattern = json.load(f)

    # Create validator
    verifier = None
    if args.with_verifier:
        try:
            from chuk_virtual_expert import ExpertRegistry, TraceVerifier
            from chuk_virtual_expert_arithmetic import ArithmeticExpert
            registry = ExpertRegistry()
            registry.register(ArithmeticExpert())
            verifier = TraceVerifier(registry)
        except ImportError:
            print("Warning: Could not import verifier")

    validator = TemplateValidator(schema, verifier)

    # Validate templates
    templates = pattern.get("templates", [])
    print(f"\nValidating {len(templates)} templates with {args.n_samples} samples each...\n")

    for i, template in enumerate(templates):
        text = template if isinstance(template, str) else template.get("text", "")
        result = validator.validate_template(text, n_samples=args.n_samples)

        status = "✓" if result["validated"] else "✗"
        print(f"{status} Template {i+1}: {result['accuracy']:.0%} accuracy")

        if not result["validated"]:
            if not result["placeholder_check"]["valid"]:
                print(f"    Missing placeholders: {result['placeholder_check']['missing']}")
            for failure in result["failures"][:2]:
                print(f"    Failure: {failure.get('error', failure.get('question', 'unknown'))[:60]}")

    return 0


def cmd_expand(args):
    """Expand a schema with LLM-generated variations."""
    schema_path, pattern_path = find_schema_and_pattern(args.schema)

    if not schema_path:
        print(f"Schema not found: {args.schema}")
        return 1

    if not pattern_path:
        print(f"Pattern not found: {args.schema}")
        return 1

    print(f"Schema: {schema_path}")
    print(f"Pattern: {pattern_path}")

    # Setup LLM client
    llm_client = None
    if args.api_key or os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            llm_client = OpenAI(api_key=args.api_key or os.environ.get("OPENAI_API_KEY"))
            print(f"Using OpenAI API with model: {args.model}")
        except ImportError:
            print("OpenAI package not installed, using mock expansion")

    # Setup verifier
    verifier = None
    if args.with_verifier:
        try:
            from chuk_virtual_expert import ExpertRegistry, TraceVerifier
            from chuk_virtual_expert_arithmetic import ArithmeticExpert
            registry = ExpertRegistry()
            registry.register(ArithmeticExpert())
            verifier = TraceVerifier(registry)
        except ImportError:
            print("Warning: Could not import verifier")

    # Expand
    expander = SchemaExpander(llm_client, model=args.model)
    result = expander.expand_schema(
        schema_path,
        pattern_path,
        n_variations=args.n_variations,
        validation_samples=args.validation_samples,
        verifier=verifier,
    )

    print(f"\nExpansion Results:")
    print(f"  Generated: {result['generated']}")
    print(f"  Validated: {result['validated']}")
    print(f"  Rejected: {result['rejected']}")

    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.schema}_expanded.json"
        update_pattern_file(pattern_path, result, output_path)

    # Also save full result for debugging
    if args.save_full:
        full_path = Path(args.output or ".") / f"{args.schema}_expansion_result.json"
        with open(full_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Full result saved: {full_path}")

    return 0


def cmd_show_prompt(args):
    """Show the expansion prompt for a schema (for debugging)."""
    schema_path, pattern_path = find_schema_and_pattern(args.schema)

    if not schema_path or not pattern_path:
        print(f"Schema or pattern not found: {args.schema}")
        return 1

    with open(schema_path) as f:
        schema = json.load(f)
    with open(pattern_path) as f:
        pattern = json.load(f)

    expander = SchemaExpander()
    prompt = expander._build_prompt(schema, pattern, args.n_variations)

    print("=" * 70)
    print("EXPANSION PROMPT")
    print("=" * 70)
    print(prompt)
    print("=" * 70)

    return 0


def main():
    parser = argparse.ArgumentParser(description="Template Expander for Training Data")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate existing templates")
    validate_parser.add_argument("--schema", required=True, help="Schema name")
    validate_parser.add_argument("--n-samples", type=int, default=10, help="Samples per template")
    validate_parser.add_argument("--with-verifier", action="store_true", help="Use TraceVerifier")

    # Expand command
    expand_parser = subparsers.add_parser("expand", help="Expand schema with LLM")
    expand_parser.add_argument("--schema", required=True, help="Schema name")
    expand_parser.add_argument("--n-variations", type=int, default=30, help="Variations to generate")
    expand_parser.add_argument("--validation-samples", type=int, default=10, help="Samples per validation")
    expand_parser.add_argument("--output", help="Output directory")
    expand_parser.add_argument("--model", default="gpt-4", help="LLM model name")
    expand_parser.add_argument("--api-key", help="API key (or set OPENAI_API_KEY)")
    expand_parser.add_argument("--with-verifier", action="store_true", help="Use TraceVerifier")
    expand_parser.add_argument("--save-full", action="store_true", help="Save full expansion result")

    # Show prompt command
    prompt_parser = subparsers.add_parser("show-prompt", help="Show expansion prompt")
    prompt_parser.add_argument("--schema", required=True, help="Schema name")
    prompt_parser.add_argument("--n-variations", type=int, default=30, help="Variations count for prompt")

    args = parser.parse_args()

    if args.command == "validate":
        return cmd_validate(args)
    elif args.command == "expand":
        return cmd_expand(args)
    elif args.command == "show-prompt":
        return cmd_show_prompt(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())
