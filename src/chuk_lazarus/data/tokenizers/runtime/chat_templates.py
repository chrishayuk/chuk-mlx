"""
Chat template registry, validation, and patching.

This module provides:
- Registry of known chat templates by model family
- Validation of Jinja2 template syntax
- Detection of template format/type
- Patching utilities for tokenizers with missing/malformed templates

Usage:
    from chuk_lazarus.data.tokenizers.runtime.chat_templates import (
        ChatTemplateRegistry,
        validate_chat_template,
        detect_template_format,
        patch_chat_template,
    )

    # Detect and validate
    format = detect_template_format(tokenizer.chat_template)
    issues = validate_chat_template(tokenizer)

    # Patch if needed
    if not tokenizer.chat_template:
        patch_chat_template(tokenizer, "llama")
"""

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..types import TokenizerProtocol


class TemplateFormat(str, Enum):
    """Known chat template formats."""

    CHATML = "chatml"  # <|im_start|>...<|im_end|>
    LLAMA = "llama"  # [INST]...[/INST]
    PHI = "phi"  # <|user|>...<|end|>
    GEMMA = "gemma"  # <start_of_turn>...<end_of_turn>
    ZEPHYR = "zephyr"  # <|system|>...<|endoftext|>
    VICUNA = "vicuna"  # USER: ... ASSISTANT: ...
    ALPACA = "alpaca"  # ### Instruction: ... ### Response: ...
    UNKNOWN = "unknown"


class TemplateCapability(str, Enum):
    """Template feature capabilities."""

    SYSTEM_MESSAGE = "system_message"
    MULTI_TURN = "multi_turn"
    TOOL_CALLS = "tool_calls"
    GENERATION_PROMPT = "generation_prompt"


class TemplateIssue(BaseModel):
    """An issue detected in a chat template."""

    severity: str = Field(description="error, warning, or info")
    message: str = Field(description="Human-readable issue description")
    suggestion: str | None = Field(default=None, description="How to fix")


class TemplateValidationResult(BaseModel):
    """Result of validating a chat template."""

    is_valid: bool = Field(description="Whether template is syntactically valid")
    format: TemplateFormat = Field(description="Detected template format")
    capabilities: list[TemplateCapability] = Field(
        default_factory=list, description="Detected capabilities"
    )
    issues: list[TemplateIssue] = Field(default_factory=list, description="Issues found")
    test_outputs: dict[str, str] = Field(default_factory=dict, description="Test scenario outputs")


class TemplateDefinition(BaseModel):
    """A chat template definition."""

    format: TemplateFormat = Field(description="Template format identifier")
    template: str = Field(description="Jinja2 template string")
    description: str = Field(description="Human-readable description")
    supports_system: bool = Field(default=True, description="Supports system messages")
    supports_tools: bool = Field(default=False, description="Supports tool calls")
    model_families: list[str] = Field(
        default_factory=list, description="Model families using this template"
    )


# =============================================================================
# Known Templates Registry
# =============================================================================

# ChatML template (Qwen, OpenHermes, etc.)
CHATML_TEMPLATE = """{% for message in messages %}{{'<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>' + '
'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}"""

# Llama 2/3 style template
LLAMA_TEMPLATE = """{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"""

# Llama 3 style (simplified)
LLAMA3_TEMPLATE = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>

'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>

' }}{% endif %}"""

# Phi-style template
PHI_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'system' %}{{'<|system|>
' + message['content'] + '<|end|>
'}}{% elif message['role'] == 'user' %}{{'<|user|>
' + message['content'] + '<|end|>
'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>
' + message['content'] + '<|end|>
'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>
' }}{% endif %}"""

# Gemma-style template (no system messages)
GEMMA_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'user' %}{{'<start_of_turn>user
' + message['content'] + '<end_of_turn>
'}}{% elif message['role'] == 'model' or message['role'] == 'assistant' %}{{'<start_of_turn>model
' + message['content'] + '<end_of_turn>
'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model
' }}{% endif %}"""

# Zephyr-style template
ZEPHYR_TEMPLATE = """{% for message in messages %}
{% if message['role'] == 'user' %}
{{ '<|user|>
' + message['content'] + '<|endoftext|>' }}
{% elif message['role'] == 'system' %}
{{ '<|system|>
' + message['content'] + '<|endoftext|>' }}
{% elif message['role'] == 'assistant' %}
{{ '<|assistant|>
' + message['content'] + '<|endoftext|>' }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
{{ '<|assistant|>' }}
{% endif %}"""

# Vicuna style (simple text-based)
VICUNA_TEMPLATE = """{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + '

' }}{% set loop_messages = messages[1:] %}{% else %}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + '
' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '
' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT: ' }}{% endif %}"""

# Alpaca style (instruction format)
ALPACA_TEMPLATE = """{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + '

' }}{% set loop_messages = messages[1:] %}{% else %}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ '### Instruction:
' + message['content'] + '

' }}{% elif message['role'] == 'assistant' %}{{ '### Response:
' + message['content'] + '

' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '### Response:
' }}{% endif %}"""


class ChatTemplateRegistry:
    """
    Registry of known chat templates with detection and patching support.

    Usage:
        registry = ChatTemplateRegistry()

        # Get available templates
        templates = registry.list_templates()

        # Get a specific template
        template = registry.get_template("chatml")

        # Detect template format from string
        format = registry.detect_format(template_str)

        # Get template for a model family
        template = registry.get_for_model_family("llama")
    """

    def __init__(self):
        self._templates: dict[TemplateFormat, TemplateDefinition] = {}
        self._model_family_map: dict[str, TemplateFormat] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register all default templates."""
        self.register(
            TemplateDefinition(
                format=TemplateFormat.CHATML,
                template=CHATML_TEMPLATE,
                description="ChatML format used by Qwen, OpenHermes, etc.",
                supports_system=True,
                supports_tools=False,
                model_families=["qwen", "openhermes", "yi", "deepseek"],
            )
        )

        self.register(
            TemplateDefinition(
                format=TemplateFormat.LLAMA,
                template=LLAMA_TEMPLATE,
                description="Llama 2 style with [INST] markers",
                supports_system=True,
                supports_tools=False,
                model_families=["llama", "llama2", "mistral", "codellama"],
            )
        )

        self.register(
            TemplateDefinition(
                format=TemplateFormat.PHI,
                template=PHI_TEMPLATE,
                description="Phi-style with <|user|> markers",
                supports_system=True,
                supports_tools=False,
                model_families=["phi", "phi2", "phi3", "tinyllama"],
            )
        )

        self.register(
            TemplateDefinition(
                format=TemplateFormat.GEMMA,
                template=GEMMA_TEMPLATE,
                description="Gemma-style with <start_of_turn> markers",
                supports_system=False,
                supports_tools=False,
                model_families=["gemma", "gemma2"],
            )
        )

        self.register(
            TemplateDefinition(
                format=TemplateFormat.ZEPHYR,
                template=ZEPHYR_TEMPLATE,
                description="Zephyr-style with <|user|> and <|endoftext|>",
                supports_system=True,
                supports_tools=False,
                model_families=["zephyr"],
            )
        )

        self.register(
            TemplateDefinition(
                format=TemplateFormat.VICUNA,
                template=VICUNA_TEMPLATE,
                description="Vicuna style with USER:/ASSISTANT: prefixes",
                supports_system=True,
                supports_tools=False,
                model_families=["vicuna", "wizard"],
            )
        )

        self.register(
            TemplateDefinition(
                format=TemplateFormat.ALPACA,
                template=ALPACA_TEMPLATE,
                description="Alpaca instruction format",
                supports_system=True,
                supports_tools=False,
                model_families=["alpaca"],
            )
        )

    def register(self, definition: TemplateDefinition) -> None:
        """Register a template definition."""
        self._templates[definition.format] = definition
        for family in definition.model_families:
            self._model_family_map[family.lower()] = definition.format

    def get_template(self, format_name: str | TemplateFormat) -> TemplateDefinition | None:
        """Get a template by format name."""
        if isinstance(format_name, str):
            try:
                format_name = TemplateFormat(format_name.lower())
            except ValueError:
                return None
        return self._templates.get(format_name)

    def list_templates(self) -> list[TemplateDefinition]:
        """List all registered templates."""
        return list(self._templates.values())

    def get_for_model_family(self, model_name: str) -> TemplateDefinition | None:
        """
        Get the best template for a model name/family.

        Tries to match based on common patterns in the model name.
        """
        model_lower = model_name.lower()

        # Direct family match
        for family, format in self._model_family_map.items():
            if family in model_lower:
                return self._templates[format]

        # Pattern-based detection
        if "qwen" in model_lower or "yi" in model_lower:
            return self._templates.get(TemplateFormat.CHATML)
        if "llama" in model_lower and "tiny" not in model_lower:
            return self._templates.get(TemplateFormat.LLAMA)
        if "mistral" in model_lower:
            return self._templates.get(TemplateFormat.LLAMA)
        if "phi" in model_lower or "tiny" in model_lower:
            return self._templates.get(TemplateFormat.PHI)
        if "gemma" in model_lower:
            return self._templates.get(TemplateFormat.GEMMA)
        if "zephyr" in model_lower:
            return self._templates.get(TemplateFormat.ZEPHYR)

        return None

    def detect_format(self, template_str: str) -> TemplateFormat:
        """Detect the template format from a template string."""
        if not template_str:
            return TemplateFormat.UNKNOWN

        # Check for distinctive markers (order matters - more specific first)
        if "<|im_start|>" in template_str or "<|im_end|>" in template_str:
            return TemplateFormat.CHATML
        if "[INST]" in template_str or "[/INST]" in template_str:
            return TemplateFormat.LLAMA
        if "<start_of_turn>" in template_str:
            return TemplateFormat.GEMMA
        if "<|endoftext|>" in template_str and "<|user|>" in template_str:
            return TemplateFormat.ZEPHYR
        # Phi-style: <|user|> and <|assistant|> markers (with various endings)
        if "<|user|>" in template_str and "<|assistant|>" in template_str:
            return TemplateFormat.PHI
        if "USER:" in template_str and "ASSISTANT:" in template_str:
            return TemplateFormat.VICUNA
        if "### Instruction:" in template_str:
            return TemplateFormat.ALPACA

        return TemplateFormat.UNKNOWN


# =============================================================================
# Validation Functions
# =============================================================================


def validate_jinja2_syntax(template_str: str) -> tuple[bool, str | None]:
    """
    Validate Jinja2 template syntax.

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        from jinja2 import Environment, TemplateSyntaxError

        env = Environment()
        env.parse(template_str)
        return True, None
    except TemplateSyntaxError as e:
        return False, f"Jinja2 syntax error at line {e.lineno}: {e.message}"
    except Exception as e:
        return False, f"Template parsing error: {e}"


def validate_chat_template(
    tokenizer: "TokenizerProtocol",
    test_system: bool = True,
    test_multi_turn: bool = True,
) -> TemplateValidationResult:
    """
    Validate a tokenizer's chat template.

    Args:
        tokenizer: The tokenizer to validate
        test_system: Whether to test system message support
        test_multi_turn: Whether to test multi-turn conversations

    Returns:
        TemplateValidationResult with all findings
    """
    issues: list[TemplateIssue] = []
    capabilities: list[TemplateCapability] = []
    test_outputs: dict[str, str] = {}

    # Check if template exists
    template_str = getattr(tokenizer, "chat_template", None)
    if not template_str:
        return TemplateValidationResult(
            is_valid=False,
            format=TemplateFormat.UNKNOWN,
            capabilities=[],
            issues=[
                TemplateIssue(
                    severity="error",
                    message="No chat template defined",
                    suggestion="Set tokenizer.chat_template or use patch_chat_template()",
                )
            ],
            test_outputs={},
        )

    # Detect format
    registry = ChatTemplateRegistry()
    format = registry.detect_format(template_str)

    # Validate Jinja2 syntax
    is_valid, syntax_error = validate_jinja2_syntax(template_str)
    if not is_valid:
        issues.append(TemplateIssue(severity="error", message=syntax_error or "Invalid syntax"))
        return TemplateValidationResult(
            is_valid=False,
            format=format,
            capabilities=[],
            issues=issues,
            test_outputs={},
        )

    # Test basic rendering
    try:
        basic_messages = [{"role": "user", "content": "Hello"}]
        result = tokenizer.apply_chat_template(
            basic_messages, add_generation_prompt=True, tokenize=False
        )
        test_outputs["basic"] = str(result)[:200]
        capabilities.append(TemplateCapability.GENERATION_PROMPT)
    except Exception as e:
        issues.append(
            TemplateIssue(
                severity="error",
                message=f"Failed basic render: {e}",
                suggestion="Check template has 'messages' variable and proper loop",
            )
        )

    # Test system message support
    if test_system:
        try:
            system_messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ]
            result = tokenizer.apply_chat_template(
                system_messages, add_generation_prompt=True, tokenize=False
            )
            if "You are helpful" in str(result):
                capabilities.append(TemplateCapability.SYSTEM_MESSAGE)
                test_outputs["system"] = str(result)[:200]
            else:
                issues.append(
                    TemplateIssue(
                        severity="warning",
                        message="System message not rendered in output",
                        suggestion="Template may not support system role",
                    )
                )
        except Exception as e:
            issues.append(
                TemplateIssue(
                    severity="warning",
                    message=f"System message test failed: {e}",
                )
            )

    # Test multi-turn
    if test_multi_turn:
        try:
            multi_messages = [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
            ]
            result = tokenizer.apply_chat_template(
                multi_messages, add_generation_prompt=True, tokenize=False
            )
            if "Hi" in str(result) and "Hello!" in str(result) and "How are you?" in str(result):
                capabilities.append(TemplateCapability.MULTI_TURN)
                test_outputs["multi_turn"] = str(result)[:300]
        except Exception as e:
            issues.append(
                TemplateIssue(
                    severity="warning",
                    message=f"Multi-turn test failed: {e}",
                )
            )

    # Check for generation prompt marker
    if "add_generation_prompt" not in template_str:
        issues.append(
            TemplateIssue(
                severity="info",
                message="Template doesn't check add_generation_prompt",
                suggestion="Consider adding {% if add_generation_prompt %} block",
            )
        )

    return TemplateValidationResult(
        is_valid=len([i for i in issues if i.severity == "error"]) == 0,
        format=format,
        capabilities=capabilities,
        issues=issues,
        test_outputs=test_outputs,
    )


# =============================================================================
# Patching Functions
# =============================================================================


def detect_template_format(template_str: str | None) -> TemplateFormat:
    """Detect the format of a chat template string."""
    registry = ChatTemplateRegistry()
    return registry.detect_format(template_str or "")


def suggest_template_for_model(model_name: str) -> TemplateDefinition | None:
    """Suggest a chat template for a model based on its name."""
    registry = ChatTemplateRegistry()
    return registry.get_for_model_family(model_name)


def patch_chat_template(
    tokenizer: "TokenizerProtocol",
    template_format: str | TemplateFormat | None = None,
    custom_template: str | None = None,
) -> bool:
    """
    Patch a tokenizer with a chat template.

    Args:
        tokenizer: The tokenizer to patch
        template_format: Format name (chatml, llama, phi, etc.) or auto-detect
        custom_template: Custom Jinja2 template string (overrides format)

    Returns:
        True if successfully patched, False otherwise

    Example:
        # Auto-detect based on model name
        patch_chat_template(tokenizer)

        # Specify format
        patch_chat_template(tokenizer, "chatml")

        # Custom template
        patch_chat_template(tokenizer, custom_template="{% for m in messages %}...")
    """
    if custom_template:
        # Validate custom template
        is_valid, error = validate_jinja2_syntax(custom_template)
        if not is_valid:
            raise ValueError(f"Invalid custom template: {error}")
        tokenizer.chat_template = custom_template
        return True

    registry = ChatTemplateRegistry()

    if template_format:
        # Use specified format
        if isinstance(template_format, str):
            try:
                template_format = TemplateFormat(template_format.lower())
            except ValueError as err:
                raise ValueError(
                    f"Unknown template format: {template_format}. "
                    f"Available: {[f.value for f in TemplateFormat if f != TemplateFormat.UNKNOWN]}"
                ) from err
        definition = registry.get_template(template_format)
    else:
        # Auto-detect based on model name
        model_name = getattr(tokenizer, "name_or_path", None) or ""
        definition = registry.get_for_model_family(model_name)

        if not definition:
            # Try to detect from existing template hints
            existing = getattr(tokenizer, "chat_template", None)
            if existing:
                format = registry.detect_format(existing)
                definition = registry.get_template(format)

    if not definition:
        return False

    tokenizer.chat_template = definition.template
    return True


def get_template_diff(
    tokenizer: "TokenizerProtocol",
    new_template: str,
) -> dict:
    """
    Compare current template with a new one.

    Returns:
        Dict with 'current_format', 'new_format', 'changes' info
    """
    registry = ChatTemplateRegistry()

    current = getattr(tokenizer, "chat_template", None) or ""
    current_format = registry.detect_format(current)
    new_format = registry.detect_format(new_template)

    return {
        "current_format": current_format.value,
        "new_format": new_format.value,
        "current_length": len(current),
        "new_length": len(new_template),
        "format_changed": current_format != new_format,
    }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Enums
    "TemplateFormat",
    "TemplateCapability",
    # Models
    "TemplateIssue",
    "TemplateValidationResult",
    "TemplateDefinition",
    # Registry
    "ChatTemplateRegistry",
    # Constants
    "CHATML_TEMPLATE",
    "LLAMA_TEMPLATE",
    "LLAMA3_TEMPLATE",
    "PHI_TEMPLATE",
    "GEMMA_TEMPLATE",
    "ZEPHYR_TEMPLATE",
    "VICUNA_TEMPLATE",
    "ALPACA_TEMPLATE",
    # Functions
    "validate_jinja2_syntax",
    "validate_chat_template",
    "detect_template_format",
    "suggest_template_for_model",
    "patch_chat_template",
    "get_template_diff",
]
