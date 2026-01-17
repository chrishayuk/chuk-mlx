"""
IR Emission Experiments.

Neural Compiler: NL -> WASM IR -> Execute

This package contains experiments demonstrating that:
1. Transformers can serve as semantic frontends (NL -> canonical)
2. Logit lens classification extracts operation intent (canonical -> IR)
3. Deterministic runtimes handle computation (WASM execution)
4. The combination achieves Turing completeness via loops

Core thesis: "Chain-of-Thought is format normalization, not reasoning."
"""
