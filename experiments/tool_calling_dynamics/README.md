# Tool Calling Dynamics

Understanding how GPT-OSS internally represents and generates tool calls.

## Quick Start

```bash
# Run full experiment
python experiments/tool_calling_dynamics/experiment.py

# Run specific analysis
python experiments/tool_calling_dynamics/experiment.py --analysis intent
python experiments/tool_calling_dynamics/experiment.py --analysis selection
python experiments/tool_calling_dynamics/experiment.py --analysis experts
python experiments/tool_calling_dynamics/experiment.py --analysis circuits
python experiments/tool_calling_dynamics/experiment.py --analysis vocab
```

## Research Questions

1. **Tool Intent** - When does the model decide to call a tool?
2. **Tool Selection** - How does it choose which tool?
3. **Expert Patterns** - Which MoE experts handle tool syntax?
4. **Circuits** - Do cross-layer expert circuits form for tools?
5. **Vocab Alignment** - Are tool names in vocabulary space?

## Key Files

- `EXPERIMENT.md` - Full methodology and research questions
- `config.yaml` - Configuration (model, tools, prompts)
- `experiment.py` - Main orchestration script
- `probes/` - Linear probe training for intent/selection
- `analysis/` - Expert patterns, circuits, vocab alignment
- `results/` - Output artifacts

## Expected Findings

- Tool intent detectable by layer 8 (similar to task classification)
- Specific experts specialize in tool-calling syntax
- Cross-layer circuits form for different tool types
- Tool names partially vocab-aligned at L13

See `EXPERIMENT.md` for detailed methodology.
