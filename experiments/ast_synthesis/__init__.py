"""
AST-Based IR Synthesis

This experiment tests compositional generalization through structural abstraction.
Instead of classifying into program IDs (which fails on held-out programs),
we classify into template IDs (structural equivalence classes).

Key hypothesis: sum_even and collatz share the same template (LOOP_CONDITIONAL_ACCUMULATE).
If trained on sum_even, the model should generalize to collatz.

Modules:
  - ast_nodes: AST dataclass definitions
  - templates: Template definitions (structural equivalence classes)
  - slot_filler: Rule-based slot filling for templates
  - linearize: AST → IR opcode compiler
  - data_generator: Training data generator with template labels
  - template_classifier: Template classifier training
  - evaluate: End-to-end evaluation pipeline
"""

from .templates import TemplateID, NUM_TEMPLATES, template_name, PROGRAM_TO_TEMPLATE
from .ast_nodes import Program, Loop, If, SetLocal, BinOp, BinOpType, Const, Local, Slot
from .slot_filler import fill_slots, detect_program_hint
from .linearize import linearize, compile_and_execute

__all__ = [
    "TemplateID",
    "NUM_TEMPLATES",
    "template_name",
    "PROGRAM_TO_TEMPLATE",
    "Program",
    "Loop",
    "If",
    "SetLocal",
    "BinOp",
    "BinOpType",
    "Const",
    "Local",
    "Slot",
    "fill_slots",
    "detect_program_hint",
    "linearize",
    "compile_and_execute",
]
