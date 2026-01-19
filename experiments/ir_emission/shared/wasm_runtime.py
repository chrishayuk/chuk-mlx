"""
WASM Runtime

Wrapper around wasmtime for executing IR programs.
Provides validation, compilation, and execution of generated WASM.
"""

import struct
from dataclasses import dataclass

# Try to import wasmtime, fall back to pure Python interpreter
try:
    import wasmtime

    WASMTIME_AVAILABLE = True
except ImportError:
    WASMTIME_AVAILABLE = False


@dataclass
class ExecutionResult:
    """Result of WASM execution."""

    success: bool
    result: int | None = None
    error: str | None = None
    execution_time_us: float | None = None


class WASMRuntime:
    """
    WASM execution runtime.

    Wraps wasmtime for native execution, with fallback to
    a simple stack-based interpreter for testing.
    """

    def __init__(self, use_native: bool = True):
        """
        Initialize the runtime.

        Args:
            use_native: Use wasmtime if available, else pure Python
        """
        self.use_native = use_native and WASMTIME_AVAILABLE

        if self.use_native:
            self.engine = wasmtime.Engine()
            self.store = wasmtime.Store(self.engine)
        else:
            self.engine = None
            self.store = None

    def build_module(
        self,
        body_bytes: bytes,
        num_locals: int = 2,
    ) -> bytes:
        """
        Build a complete WASM module from function body bytes.

        Creates a minimal module with:
        - One function that returns i32
        - Specified number of i32 locals
        - The provided body bytecode

        Args:
            body_bytes: Function body bytecode
            num_locals: Number of i32 local variables

        Returns:
            Complete WASM module bytes
        """
        # WASM module structure:
        # magic + version + sections

        module = bytearray()

        # Magic number and version
        module.extend(b"\x00asm")  # Magic
        module.extend(struct.pack("<I", 1))  # Version 1

        # Type section (section 1)
        # Defines function signature: () -> i32
        type_section = bytearray()
        type_section.append(1)  # 1 type
        type_section.append(0x60)  # func type
        type_section.append(0)  # 0 params
        type_section.append(1)  # 1 result
        type_section.append(0x7F)  # i32

        module.append(1)  # Section ID
        module.append(len(type_section))  # Section size
        module.extend(type_section)

        # Function section (section 3)
        # Maps function to type
        func_section = bytearray()
        func_section.append(1)  # 1 function
        func_section.append(0)  # Type index 0

        module.append(3)  # Section ID
        module.append(len(func_section))
        module.extend(func_section)

        # Export section (section 7)
        # Export function as "compute"
        export_section = bytearray()
        export_section.append(1)  # 1 export
        export_section.append(7)  # Name length
        export_section.extend(b"compute")  # Name
        export_section.append(0)  # Export kind: func
        export_section.append(0)  # Function index

        module.append(7)  # Section ID
        module.append(len(export_section))
        module.extend(export_section)

        # Code section (section 10)
        # Contains function bodies
        code_section = bytearray()
        code_section.append(1)  # 1 function

        # Function body
        func_body = bytearray()

        # Locals declaration
        if num_locals > 0:
            func_body.append(1)  # 1 local declaration
            func_body.append(num_locals)  # Count
            func_body.append(0x7F)  # Type: i32
        else:
            func_body.append(0)  # No locals

        # Body bytecode
        func_body.extend(body_bytes)

        # End opcode
        func_body.append(0x0B)

        # Function body size
        code_section.append(len(func_body))
        code_section.extend(func_body)

        module.append(10)  # Section ID
        module.append(len(code_section))
        module.extend(code_section)

        return bytes(module)

    def validate(self, wasm_bytes: bytes) -> tuple[bool, str | None]:
        """
        Validate WASM module.

        Args:
            wasm_bytes: Complete WASM module

        Returns:
            (is_valid, error_message)
        """
        if self.use_native:
            try:
                wasmtime.Module.validate(self.engine, wasm_bytes)
                return True, None
            except wasmtime.WasmtimeError as e:
                return False, str(e)
        else:
            # Basic validation for interpreter
            if not wasm_bytes.startswith(b"\x00asm"):
                return False, "Invalid magic number"
            return True, None

    def execute(
        self,
        body_bytes: bytes,
        num_locals: int = 2,
        timeout_ms: int = 1000,
    ) -> ExecutionResult:
        """
        Execute IR and return result.

        Args:
            body_bytes: Function body bytecode
            num_locals: Number of local variables
            timeout_ms: Execution timeout (native only)

        Returns:
            ExecutionResult with success status and result/error
        """
        import time

        start = time.perf_counter()

        try:
            # Build complete module
            wasm_bytes = self.build_module(body_bytes, num_locals)

            # Validate
            valid, error = self.validate(wasm_bytes)
            if not valid:
                return ExecutionResult(success=False, error=f"Validation: {error}")

            if self.use_native:
                result = self._execute_native(wasm_bytes, timeout_ms)
            else:
                result = self._execute_interpreted(body_bytes)

            elapsed = (time.perf_counter() - start) * 1_000_000
            result.execution_time_us = elapsed
            return result

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1_000_000
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time_us=elapsed,
            )

    def _execute_native(
        self,
        wasm_bytes: bytes,
        timeout_ms: int,
    ) -> ExecutionResult:
        """Execute with wasmtime."""
        # Create new store for isolation
        store = wasmtime.Store(self.engine)

        # Compile module
        module = wasmtime.Module(self.engine, wasm_bytes)

        # Instantiate
        instance = wasmtime.Instance(store, module, [])

        # Get exported function
        compute = instance.exports(store)["compute"]

        # Execute
        result = compute(store)

        return ExecutionResult(success=True, result=int(result))

    def _execute_interpreted(self, body_bytes: bytes) -> ExecutionResult:
        """
        Simple stack-based interpreter for testing without wasmtime.

        Only supports basic arithmetic - enough for Phase 1.
        """
        stack: list[int] = []
        locals_: dict[int, int] = {}
        pc = 0

        while pc < len(body_bytes):
            opcode = body_bytes[pc]
            pc += 1

            # i32.const
            if opcode == 0x41:
                # Decode LEB128
                value = 0
                shift = 0
                while True:
                    byte = body_bytes[pc]
                    pc += 1
                    value |= (byte & 0x7F) << shift
                    shift += 7
                    if (byte & 0x80) == 0:
                        break
                # Sign extend
                if shift < 32 and (byte & 0x40):
                    value |= ~0 << shift
                stack.append(value & 0xFFFFFFFF)
                if stack[-1] >= 0x80000000:
                    stack[-1] -= 0x100000000

            # i32.add
            elif opcode == 0x6A:
                b = stack.pop()
                a = stack.pop()
                stack.append((a + b) & 0xFFFFFFFF)
                if stack[-1] >= 0x80000000:
                    stack[-1] -= 0x100000000

            # i32.sub
            elif opcode == 0x6B:
                b = stack.pop()
                a = stack.pop()
                stack.append((a - b) & 0xFFFFFFFF)
                if stack[-1] >= 0x80000000:
                    stack[-1] -= 0x100000000

            # i32.mul
            elif opcode == 0x6C:
                b = stack.pop()
                a = stack.pop()
                stack.append((a * b) & 0xFFFFFFFF)
                if stack[-1] >= 0x80000000:
                    stack[-1] -= 0x100000000

            # i32.div_s
            elif opcode == 0x6D:
                b = stack.pop()
                a = stack.pop()
                if b == 0:
                    return ExecutionResult(success=False, error="Division by zero")
                # Signed division
                result = int(a / b)  # Python's // rounds toward -inf, we want toward 0
                stack.append(result)

            # i32.rem_s
            elif opcode == 0x6F:
                b = stack.pop()
                a = stack.pop()
                if b == 0:
                    return ExecutionResult(success=False, error="Division by zero")
                stack.append(a % b)

            # local.get
            elif opcode == 0x20:
                idx = body_bytes[pc]
                pc += 1
                stack.append(locals_.get(idx, 0))

            # local.set
            elif opcode == 0x21:
                idx = body_bytes[pc]
                pc += 1
                locals_[idx] = stack.pop()

            # local.tee
            elif opcode == 0x22:
                idx = body_bytes[pc]
                pc += 1
                locals_[idx] = stack[-1]  # Don't pop

            # drop
            elif opcode == 0x1A:
                stack.pop()

            # end
            elif opcode == 0x0B:
                break

            else:
                return ExecutionResult(
                    success=False,
                    error=f"Unknown opcode: 0x{opcode:02x}",
                )

        if len(stack) == 0:
            return ExecutionResult(success=False, error="Stack underflow")

        return ExecutionResult(success=True, result=stack[-1])


# Convenience function for quick execution
def execute_ir(
    indices: list[int],
    operands: list[int],
    codebook: "IRCodebook",  # type: ignore
) -> ExecutionResult:
    """
    Execute IR program from codebook indices.

    Args:
        indices: Sequence of IROpcode values
        operands: Numbers extracted from input
        codebook: IRCodebook for conversion

    Returns:
        ExecutionResult
    """
    runtime = WASMRuntime(use_native=WASMTIME_AVAILABLE)
    body_bytes = codebook.indices_to_wasm(indices, operands)
    return runtime.execute(body_bytes)


if __name__ == "__main__":
    # Quick test
    print(f"wasmtime available: {WASMTIME_AVAILABLE}")

    runtime = WASMRuntime()

    # Test: 3 + 4
    from .codebook import CodebookConfig, IRCodebook, IROpcode

    config = CodebookConfig(hidden_dim=128)
    codebook = IRCodebook(config)

    # Manual IR: push 3, push 4, add
    indices = [IROpcode.SLOT_0, IROpcode.SLOT_1, IROpcode.I32_ADD]
    operands = [3, 4]

    body = codebook.indices_to_wasm(indices, operands)
    print(f"Body bytes: {body.hex()}")

    result = runtime.execute(body)
    print(f"3 + 4 = {result.result} (success={result.success})")

    # Test: 16 - 3 - 4 then * 7 (Janet's eggs)
    indices = [
        IROpcode.SLOT_0,  # 16
        IROpcode.SLOT_1,  # 3
        IROpcode.I32_SUB,
        IROpcode.SLOT_2,  # 4
        IROpcode.I32_SUB,
        IROpcode.SLOT_3,  # 7
        IROpcode.I32_MUL,
    ]
    operands = [16, 3, 4, 7]

    body = codebook.indices_to_wasm(indices, operands)
    result = runtime.execute(body)
    print(f"(16 - 3 - 4) * 7 = {result.result} (success={result.success})")
