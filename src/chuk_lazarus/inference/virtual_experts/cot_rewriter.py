"""
Chain-of-Thought (CoT) Rewriter for Virtual Experts.

Rewrites user queries into normalized VirtualExpertAction format
for consistent routing and expert invocation.

Flow:
    User Query (any phrasing)
        ↓
    CoT Rewrite (LLM)
        ↓
    VirtualExpertAction JSON
        ↓
    Calibration Router (trained on action JSONs)
        ↓
    Expert.execute_action(action)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx


@dataclass
class VirtualExpertAction:
    """
    Normalized action format for virtual expert invocation.

    CoT rewriting produces this format from any query phrasing.
    The calibration router is trained on the JSON representation.
    """

    expert: str
    """Name of the virtual expert to invoke, or 'none' for passthrough."""

    operation: str
    """Specific operation to perform."""

    parameters: dict[str, Any] = field(default_factory=dict)
    """Parameters for the operation."""

    confidence: float = 1.0
    """Model's confidence in this action (0-1)."""

    reasoning: str = ""
    """CoT reasoning that led to this action."""

    def to_json(self) -> str:
        """Convert to JSON string for routing."""
        return json.dumps({
            "expert": self.expert,
            "operation": self.operation,
            "parameters": self.parameters,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        })

    @classmethod
    def from_json(cls, json_str: str) -> "VirtualExpertAction":
        """Parse from JSON string."""
        data = json.loads(json_str)
        return cls(
            expert=data.get("expert", "none"),
            operation=data.get("operation", "passthrough"),
            parameters=data.get("parameters", {}),
            confidence=data.get("confidence", 1.0),
            reasoning=data.get("reasoning", ""),
        )

    @classmethod
    def none_action(cls, reasoning: str = "") -> "VirtualExpertAction":
        """Create a passthrough action."""
        return cls(
            expert="none",
            operation="passthrough",
            confidence=1.0,
            reasoning=reasoning,
        )

    def is_passthrough(self) -> bool:
        """Check if this action should pass to base model."""
        return self.expert == "none" or self.operation == "passthrough"


class CoTRewriter(ABC):
    """
    Abstract base class for CoT rewriters.

    Rewriters convert natural language queries into normalized
    VirtualExpertAction format for consistent routing.
    """

    @abstractmethod
    def rewrite(self, query: str, available_experts: list[str]) -> VirtualExpertAction:
        """
        Rewrite a query into a VirtualExpertAction.

        Args:
            query: User's natural language query
            available_experts: List of registered expert names

        Returns:
            VirtualExpertAction with expert, operation, and parameters
        """
        ...


class FewShotCoTRewriter(CoTRewriter):
    """
    CoT rewriter using few-shot prompting.

    Uses examples from each expert's cot_examples.json to build
    a few-shot prompt that the model uses to extract actions.
    """

    SYSTEM_PROMPT = """You extract structured actions from user queries.

## Available Experts
{expert_descriptions}

## Output Format
Respond with ONLY a JSON object:
{{"expert": "<name or none>", "operation": "<op>", "parameters": {{...}}, "confidence": <0-1>, "reasoning": "<brief>"}}

## Examples
{few_shot_examples}

## Query
Query: "{query}"
Action:"""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        expert_examples: dict[str, list[dict]] | None = None,
        max_examples_per_expert: int = 3,
    ):
        """
        Initialize the rewriter.

        Args:
            model: The language model for generation
            tokenizer: The tokenizer
            expert_examples: Dict mapping expert name to list of examples
            max_examples_per_expert: Max examples to include per expert
        """
        self.model = model
        self.tokenizer = tokenizer
        self.expert_examples = expert_examples or {}
        self.max_examples_per_expert = max_examples_per_expert
        self._expert_descriptions: dict[str, str] = {}

    def set_expert_info(
        self,
        expert_name: str,
        description: str,
        examples: list[dict],
    ) -> None:
        """
        Add information about an expert.

        Args:
            expert_name: Name of the expert
            description: What the expert does
            examples: List of {"query": str, "action": dict} examples
        """
        self._expert_descriptions[expert_name] = description
        self.expert_examples[expert_name] = examples

    def _build_prompt(self, query: str, available_experts: list[str]) -> str:
        """Build the few-shot prompt."""
        # Expert descriptions
        descriptions = []
        for name in available_experts:
            desc = self._expert_descriptions.get(name, f"Expert: {name}")
            descriptions.append(f"- **{name}**: {desc}")

        # Few-shot examples
        examples = []
        for name in available_experts:
            expert_examples = self.expert_examples.get(name, [])
            for ex in expert_examples[:self.max_examples_per_expert]:
                action_json = json.dumps(ex["action"])
                examples.append(f'Query: "{ex["query"]}"\nAction: {action_json}')

        # Add negative examples (passthrough)
        examples.append('Query: "Tell me a joke"\nAction: {"expert": "none", "operation": "passthrough", "parameters": {}, "confidence": 1.0, "reasoning": "Not related to any expert"}')

        return self.SYSTEM_PROMPT.format(
            expert_descriptions="\n".join(descriptions),
            few_shot_examples="\n\n".join(examples),
            query=query,
        )

    def rewrite(self, query: str, available_experts: list[str]) -> VirtualExpertAction:
        """Rewrite query to VirtualExpertAction using few-shot prompting."""
        import mlx.core as mx

        prompt = self._build_prompt(query, available_experts)

        # Tokenize and generate
        input_ids = mx.array(self.tokenizer.encode(prompt))[None, :]
        generated = []

        for _ in range(200):  # Max tokens for response
            outputs = self.model(input_ids)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            next_token = int(mx.argmax(logits[0, -1, :]))

            if hasattr(self.tokenizer, "eos_token_id"):
                if next_token == self.tokenizer.eos_token_id:
                    break

            generated.append(next_token)
            input_ids = mx.concatenate([input_ids, mx.array([[next_token]])], axis=1)

            # Check for end of JSON
            text = self.tokenizer.decode(generated)
            if text.count("{") > 0 and text.count("{") == text.count("}"):
                break

        response = self.tokenizer.decode(generated).strip()
        return self._parse_response(response)

    def _parse_response(self, response: str) -> VirtualExpertAction:
        """Parse LLM response into VirtualExpertAction."""
        try:
            # Find the first complete JSON object
            start = response.find("{")
            if start == -1:
                return VirtualExpertAction.none_action("No JSON found")

            # Find matching closing brace
            depth = 0
            end = start
            in_string = False
            escape_next = False

            for i, char in enumerate(response[start:], start):
                if escape_next:
                    escape_next = False
                    continue
                if char == "\\":
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            json_str = response[start:end]
            return VirtualExpertAction.from_json(json_str)
        except (json.JSONDecodeError, KeyError) as e:
            return VirtualExpertAction.none_action(f"Parse error: {e}")


class DirectCoTRewriter(CoTRewriter):
    """
    Direct rewriter that doesn't use LLM.

    Uses pattern matching to extract actions. Useful for testing
    or when you want deterministic behavior.
    """

    def __init__(self, patterns: dict[str, dict] | None = None):
        """
        Initialize with optional patterns.

        Args:
            patterns: Dict mapping regex patterns to action templates
        """
        self.patterns = patterns or {}

    def add_pattern(
        self,
        pattern: str,
        expert: str,
        operation: str,
        param_groups: dict[str, int] | None = None,
    ) -> None:
        """
        Add a pattern for matching.

        Args:
            pattern: Regex pattern
            expert: Expert name
            operation: Operation name
            param_groups: Dict mapping param names to regex group indices
        """
        self.patterns[pattern] = {
            "expert": expert,
            "operation": operation,
            "param_groups": param_groups or {},
        }

    def rewrite(self, query: str, available_experts: list[str]) -> VirtualExpertAction:
        """Rewrite using pattern matching."""
        import re

        for pattern, config in self.patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match and config["expert"] in available_experts:
                params = {}
                for param_name, group_idx in config.get("param_groups", {}).items():
                    if group_idx <= len(match.groups()):
                        params[param_name] = match.group(group_idx)

                return VirtualExpertAction(
                    expert=config["expert"],
                    operation=config["operation"],
                    parameters=params,
                    reasoning=f"Matched pattern: {pattern}",
                )

        return VirtualExpertAction.none_action("No pattern matched")
