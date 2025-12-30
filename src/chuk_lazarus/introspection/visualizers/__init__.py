"""
Visualization tools for introspection results.

Provides HTML and image output for attention patterns and logit lens.
"""

from .attention_heatmap import render_attention_heatmap
from .logit_evolution import render_logit_evolution

__all__ = [
    "render_attention_heatmap",
    "render_logit_evolution",
]
