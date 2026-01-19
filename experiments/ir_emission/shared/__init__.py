"""
Shared utilities for IR emission experiments.

Core components:
- IROpcode: Enumeration of IR opcodes
- OPCODE_TO_WASM: Mapping from opcodes to WASM bytecode
- encode_i32_const: LEB128 encoding for i32 constants
- WASMRuntime: WASM execution engine
- ExecutionResult: Result dataclass from execution
"""

from .codebook import (
    IROpcode,
    OPCODE_TO_WASM,
    encode_i32_const,
    CodebookConfig,
    IRCodebook,
    IRSequenceDecoder,
)
from .wasm_runtime import (
    WASMRuntime,
    ExecutionResult,
    execute_ir,
    WASMTIME_AVAILABLE,
)

__all__ = [
    # Codebook
    "IROpcode",
    "OPCODE_TO_WASM",
    "encode_i32_const",
    "CodebookConfig",
    "IRCodebook",
    "IRSequenceDecoder",
    # Runtime
    "WASMRuntime",
    "ExecutionResult",
    "execute_ir",
    "WASMTIME_AVAILABLE",
]
