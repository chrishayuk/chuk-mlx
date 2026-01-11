"""
Built-in virtual expert plugins.

This package contains the built-in plugin implementations:
- math: Arithmetic computation using Python
- (future) code: Code execution
- (future) search: Web search
- (future) database: Database queries
"""

from .math import MathExpertPlugin, SafeMathEvaluator

__all__ = [
    "MathExpertPlugin",
    "SafeMathEvaluator",
]
