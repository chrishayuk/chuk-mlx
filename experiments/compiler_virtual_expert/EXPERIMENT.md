# Compiler as Virtual Expert

## Thesis

The model generates code. The compiler expert **verifies and executes** it.
This creates a tight feedback loop where:
- Model does what it's good at: semantic understanding, code generation
- Compiler does what it's good at: syntax validation, type checking, execution
- Errors feed back immediately, enabling self-correction within generation

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GENERATION FLOW                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User: "Write a function to sort a list and test it"            │
│         │                                                       │
│         ▼                                                       │
│  Model generates: ```python                                     │
│                   def sort(lst):                                │
│                       return sorted(lst)                        │
│                   ```                                           │
│         │                                                       │
│         ▼  [Router detects code block end]                      │
│                                                                 │
│  ┌─────────────────────────────────────────┐                   │
│  │  COMPILER EXPERT ACTIVATES              │                   │
│  │                                         │                   │
│  │  1. Extract code from ``` block         │                   │
│  │  2. Parse → Check syntax                │                   │
│  │  3. Execute in sandbox                  │                   │
│  │  4. Run test cases (if provided)        │                   │
│  │  5. Return: status + output/error       │                   │
│  └─────────────────────────────────────────┘                   │
│         │                                                       │
│         ▼                                                       │
│  Inject result: "✓ Executed successfully"                       │
│                 "Output: [1, 2, 3, 5, 8]"                       │
│         │                                                       │
│         ▼                                                       │
│  Model continues: "The function works! Let me also..."          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Error Correction Flow

```
Model generates buggy code:
  ```python
  def sort(lst):
      return lst.sort()  # Bug: .sort() returns None
  ```
         │
         ▼
Compiler Expert:
  - Executes: sort([3,1,2])
  - Returns: None (not a list!)
  - Injects: "⚠ Function returned None instead of list"
         │
         ▼
Model self-corrects:
  "I see the issue - .sort() modifies in place and returns None.
   Let me fix that:
   ```python
   def sort(lst):
       return sorted(lst)
   ```"
         │
         ▼
Compiler Expert:
  - Executes: sort([3,1,2])
  - Returns: [1, 2, 3] ✓
  - Injects: "✓ Returns [1, 2, 3]"
```

## Implementation

### Phase 1: Detection & Routing

The router learns to detect "code generation complete" states:

**Trigger signals:**
- End of ``` code block
- Function definition complete (dedent after def)
- "Let me test this" / "Running:" patterns
- High confidence "code_execution_intent" in hidden states

**Calibration prompts:**
```python
positive = [
    "```python\ndef foo():\n    return 1\n```",
    "Here's the code:\n```\nprint('hello')\n```",
    "def test():\n    pass\n\nLet me run this",
]
negative = [
    "The syntax for Python is",
    "In programming, a function",
    "```python  # Just starting",
    "What does this code do?",
]
```

### Phase 2: Compiler Expert Plugin

```python
class CompilerExpertPlugin(VirtualExpertPlugin):
    """
    Verifies and executes model-generated code.
    """
    name = "compiler"
    description = "Executes code and returns results/errors"
    priority = 8

    def can_handle(self, prompt: str) -> bool:
        """Detect completed code blocks."""
        # Look for closed code blocks
        if "```" in prompt:
            blocks = re.findall(r"```(\w*)\n(.*?)```", prompt, re.DOTALL)
            return len(blocks) > 0
        return False

    def execute(self, prompt: str) -> str | None:
        """Extract code, run it, return results."""
        code = self._extract_code(prompt)
        if not code:
            return None

        # 1. Syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            return f"SyntaxError: {e.msg} (line {e.lineno})"

        # 2. Execute in sandbox
        result = self._execute_sandboxed(code)

        # 3. Format result
        if result.success:
            return f"✓ Executed successfully\nOutput: {result.output}"
        else:
            return f"✗ Error: {result.error}"

    def _execute_sandboxed(self, code: str) -> ExecutionResult:
        """Run code in restricted environment."""
        # Use RestrictedPython or subprocess sandbox
        # Timeout after 5 seconds
        # Capture stdout/stderr
        ...
```

### Phase 3: Injection Mechanism

Adapting from moe_bypass.py:

```python
class GeneratorWithCompilerExpert:
    """Generator that invokes compiler expert on code blocks."""

    def generate(self, prompt: str, max_tokens: int = 500):
        # ... normal generation ...

        # Detect code block end
        if self._is_code_block_end(generated_text):
            # Get routing confidence
            confidence = self._get_compiler_routing_score(hidden_state)

            if confidence > self.threshold:
                # Extract and execute code
                result = self.compiler_expert.execute(full_context)

                if result:
                    # Inject result into generation
                    result_tokens = self.tokenizer.encode(f"\n{result}\n")
                    generated.extend(result_tokens)

                    # Update context for continued generation
                    input_ids = mx.array([tokens + generated])

        # Continue generation...
```

### Phase 4: Test Injection (Advanced)

The expert can also inject test cases:

```python
def _generate_test_cases(self, code: str, func_name: str) -> list[str]:
    """Generate basic test cases for a function."""
    # Parse function signature
    # Generate edge cases: empty input, single element, typical case
    # Return test code

    return [
        f"assert {func_name}([]) == []",
        f"assert {func_name}([1]) == [1]",
        f"assert {func_name}([3,1,2]) == [1,2,3]",
    ]

def execute_with_tests(self, code: str) -> str:
    """Execute code and run generated tests."""
    # Compile function
    namespace = {}
    exec(code, namespace)

    # Find function name
    func_name = self._extract_function_name(code)

    # Generate and run tests
    tests = self._generate_test_cases(code, func_name)

    results = []
    for test in tests:
        try:
            exec(test, namespace)
            results.append(f"✓ {test}")
        except AssertionError:
            results.append(f"✗ {test} FAILED")
        except Exception as e:
            results.append(f"✗ {test} ERROR: {e}")

    return "\n".join(results)
```

## Routing Signal: Where Does This Live?

From your MoE dynamics work, attention encodes intent. The question:
**Which layer/position encodes "code execution intent"?**

Hypothesis: Similar to arithmetic at "=" position, code execution intent
concentrates at the "```" closing position (end of code block).

**Experiment:**
1. Collect hidden states at ``` positions for code blocks
2. Train classifier: "should execute" vs "just explaining code"
3. Find optimal layer (likely L8-L12 based on arithmetic findings)

## Benchmark

**Tasks:**
1. Simple function generation + execution
2. Bug detection and self-correction
3. Multi-step: generate, test, iterate

**Metrics:**
- Syntax error rate (model-only vs with expert)
- Semantic correctness (does output match expected?)
- Self-correction rate (errors caught and fixed)
- Generation coherence (does model continue sensibly after feedback?)

**Baseline comparison:**
| Approach | Syntax Valid | Semantically Correct | Self-Corrects |
|----------|--------------|---------------------|---------------|
| Model only | ~85% | ~60% | N/A |
| With compiler expert | 100%* | ~90% | Yes |

*Syntax errors are caught and fed back for correction

## Connection to Existing Work

This builds directly on:
- **VirtualExpertPlugin** (base.py): Same interface
- **moe_bypass.py**: Same injection pattern
- **MathExpertPlugin**: Similar calibration approach
- **Hidden state routing**: Same learned direction mechanism

The compiler expert is just another plugin that:
1. Has different trigger conditions (code blocks vs "=")
2. Has different execution backend (Python sandbox vs WASM)
3. Returns different output format (execution results vs numbers)

## Why This Matters

**Current LLM code generation:**
- Generate code → hope it works → user runs it → finds bugs → asks for fix

**With compiler expert:**
- Generate code → compiler validates → bugs caught immediately → self-correct → verified output

This is the "neural frontend + symbolic backend" applied to code:
- Neural: understands intent, generates code structure
- Symbolic: verifies syntax, executes, provides ground truth feedback

## Files

```
experiments/compiler_virtual_expert/
├── EXPERIMENT.md           # This document
├── compiler_plugin.py      # CompilerExpertPlugin implementation
├── sandbox.py              # Safe code execution environment
├── code_detector.py        # Detect code blocks and completion
├── test_generator.py       # Auto-generate test cases
├── routing_analysis.py     # Find optimal routing layer/signal
├── generator.py            # GeneratorWithCompilerExpert
├── benchmark.py            # Evaluation suite
└── results/                # Experiment outputs
```

## Next Steps

1. [ ] Implement basic CompilerExpertPlugin with syntax check only
2. [ ] Add sandboxed execution (RestrictedPython or subprocess)
3. [ ] Train routing classifier on code block hidden states
4. [ ] Integrate with VirtualMoEWrapper
5. [ ] Benchmark against model-only baseline
6. [ ] Add test case generation
7. [ ] Implement self-correction loop
