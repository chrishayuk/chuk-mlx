# Tool Calling Dynamics in GPT-OSS

## Overview

This experiment investigates how GPT-OSS (20B MoE) internally represents and generates tool calls. We study the mechanistic process from user query to tool invocation, examining where tool intent forms, how tool selection happens, and which MoE experts participate.

## Research Questions

### 1. Tool Intent Detection

**Question**: At which layer can we detect that the model will call a tool vs answer directly?

**Method**:
- Collect prompts that require tool use vs direct answers
- Extract hidden states at each layer
- Train linear probes to classify "will call tool" vs "will answer directly"
- Identify the earliest layer with >95% accuracy

**Hypothesis**: Tool intent should be detectable by L4-L8 (similar to task classification in format_gate_gptoss).

**Metrics**:
- Probe accuracy per layer
- Layer of first reliable detection (>95%)
- Comparison with task classification layers

### 2. Tool Selection Routing

**Question**: Once tool intent is detected, how is the specific tool selected? Do specific MoE experts activate for tool selection?

**Method**:
- For prompts requiring different tools (calculator, search, code_exec), track expert activations
- Identify experts with high activation variance across tool types
- Test if expert activation patterns predict tool selection

**Expected Patterns**:
```
Calculator prompts:  High activation in experts E3, E17, E28
Search prompts:      High activation in experts E5, E12, E31
Code prompts:        High activation in experts E8, E22, E35
```

### 3. Tool Token Expert Patterns

**Question**: Do specific experts specialize in tool-calling syntax tokens?

**Tokens of Interest**:
- Function call tokens: `(`, `)`, `,`, `:`
- JSON tokens: `{`, `}`, `[`, `]`, `"`
- Tool markers: `<tool>`, `</tool>`, `function`, `call`
- Parameter tokens: `name`, `arguments`, `value`

**Method**:
- Track expert activations for each token type
- Compute specialization score (concentration of activation)
- Identify "tool syntax experts" vs "tool content experts"

### 4. Cross-Layer Tool Circuits

**Question**: Do specific expert combinations across layers form "tool circuits"?

**Method**:
- Track expert co-occurrence across layers for tool-calling sequences
- Identify stable circuits (e.g., L4-E17 → L8-E35 → L12-E3)
- Test if circuits are tool-specific or general

**Expected**:
```
Tool Intent Circuit:    L4-E5  → L8-E12 → L12-E3   (for all tool calls)
Calculator Circuit:     L4-E17 → L8-E35 → L12-E28  (math-specific)
Search Circuit:         L4-E5  → L8-E22 → L12-E31  (retrieval-specific)
```

### 5. Tool Name Vocab Alignment

**Question**: Are tool names represented in vocabulary space at intermediate layers?

**Background**: format_gate_gptoss found "synonyms" aligned to vocab at L13. Do tool names show similar alignment?

**Method**:
- For prompts like "What is 45 * 37?", extract L13 hidden state
- Project to vocabulary space (with layer norm)
- Check if "calculator", "multiply", "compute" rank highly

**Hypothesis**: Tool names may show partial vocab alignment (similar to task tokens).

### 6. Format vs Intent Separation

**Question**: Can we separately detect tool format (JSON schema) from tool intent (which tool)?

**Method**:
- Train two probes:
  1. Format probe: Will output be JSON/function call format?
  2. Intent probe: Which tool will be called?
- Test if these are independent directions in activation space
- Measure correlation between format and intent directions

### 7. Tool Call Generation Dynamics

**Question**: How does expert routing change during tool call generation?

**Method**:
- Generate tool calls token by token
- Track expert activations at each position
- Analyze phase transitions:
  - Intent phase (deciding to call tool)
  - Selection phase (choosing which tool)
  - Format phase (generating JSON/function syntax)
  - Argument phase (filling in parameters)

## Experimental Setup

### Model
- Primary: `openai/gpt-oss-20b` (32 experts, top-k=4)
- Comparison: `allenai/OLMoE-1B-7B-0924` (64 experts, top-k=8)

### Tools
We define four canonical tools for the experiment:

1. **calculator** - Arithmetic operations
2. **search** - Web/knowledge retrieval
3. **code_exec** - Python code execution
4. **get_weather** - Structured API call

### Prompt Categories

**Tool-Required Prompts** (should trigger tool call):
```
Calculator: "What is 127 * 89?", "Calculate 15% of 340"
Search: "What is the current price of Bitcoin?", "Who won the 2024 Super Bowl?"
Code: "Run this Python: print(sum(range(100)))"
Weather: "What's the weather in Tokyo?"
```

**Direct-Answer Prompts** (should NOT trigger tool call):
```
"What is the capital of France?"
"Explain photosynthesis"
"Write a haiku about mountains"
"What is 2 + 2?" (trivial math - might not need tool)
```

### Tool Call Format

We use a simple function-call format:
```json
{
  "tool": "calculator",
  "arguments": {
    "expression": "127 * 89"
  }
}
```

## Running the Experiment

### Full Experiment
```bash
python experiments/tool_calling_dynamics/experiment.py
```

### Individual Analyses
```bash
# Tool intent probing
python experiments/tool_calling_dynamics/probes/tool_intent_probe.py

# Tool selection probing
python experiments/tool_calling_dynamics/probes/tool_selection_probe.py

# Expert pattern analysis
python experiments/tool_calling_dynamics/analysis/expert_patterns.py

# Cross-layer circuit analysis
python experiments/tool_calling_dynamics/analysis/circuit_analysis.py

# Vocab alignment test
python experiments/tool_calling_dynamics/analysis/vocab_alignment.py
```

### Via CLI (when integrated)
```bash
lazarus introspect tool-dynamics intent -m openai/gpt-oss-20b
lazarus introspect tool-dynamics selection -m openai/gpt-oss-20b
lazarus introspect tool-dynamics experts -m openai/gpt-oss-20b
lazarus introspect tool-dynamics circuits -m openai/gpt-oss-20b
```

## Expected Results

| Analysis | Hypothesis | Success Criteria |
|----------|------------|------------------|
| Tool Intent | Detectable by L8 | >95% probe accuracy |
| Tool Selection | Expert specialization | >0.15 concentration score |
| Token Patterns | Syntax experts exist | Distinct expert clusters |
| Circuits | Tool-specific circuits | >80% circuit consistency |
| Vocab Alignment | Partial alignment | Tool names in top-50 tokens |
| Format vs Intent | Separable directions | <0.3 correlation |
| Generation Dynamics | Phase transitions | Distinct expert patterns per phase |

## Implications

### For Tool-Augmented LLMs
Understanding where tool intent forms allows:
- Early routing to tool-specialized pathways
- Efficient tool selection without full generation
- Detection of tool-calling errors before execution

### For MoE Architecture
Understanding expert specialization for tools informs:
- Expert pruning strategies (keep tool experts)
- Expert merging constraints (don't merge tool experts)
- Sparse activation optimization

### For Virtual Expert Integration
If tool intent is detectable early, we can:
- Route to virtual experts at intermediate layers
- Bypass language generation for deterministic tools
- Reduce latency by parallel tool preparation

## References

- `experiments/moe_dynamics/` - Expert behavior analysis methodology
- `experiments/format_gate_gptoss/` - Vocab alignment testing
- `experiments/csp_virtual_expert/` - Virtual expert integration
- `src/chuk_lazarus/env/mcp_client.py` - Tool calling interface
