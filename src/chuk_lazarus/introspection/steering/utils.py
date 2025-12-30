"""
Utility functions for activation steering.

This module contains convenience functions for common steering operations.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..circuit.directions import DirectionBundle


def steer_model(
    model_id: str,
    prompt: str,
    directions: DirectionBundle,
    layers: list[int] | None = None,
    coefficient: float = 1.0,
) -> str:
    """
    Convenience function to apply steering and generate.

    Args:
        model_id: Model to load
        prompt: Input prompt
        directions: Direction bundle to use
        layers: Layers to steer (default: all in bundle)
        coefficient: Steering strength

    Returns:
        Generated text
    """
    from .config import SteeringConfig
    from .core import ActivationSteering

    steerer = ActivationSteering.from_pretrained(model_id)
    steerer.add_directions(directions)

    config = SteeringConfig(
        layers=layers or list(directions.directions.keys()),
        coefficient=coefficient,
    )

    return steerer.generate(prompt, config)


def compare_steering_effects(
    model_id: str,
    prompt: str,
    directions: DirectionBundle,
    layer: int,
    coefficients: list[float] | None = None,
) -> dict[float, str]:
    """
    Compare steering effects at different coefficients.

    Returns dict mapping coefficient to generated output.
    """
    from .config import SteeringConfig
    from .core import ActivationSteering

    if coefficients is None:
        coefficients = [-2.0, -1.0, 0.0, 1.0, 2.0]
    steerer = ActivationSteering.from_pretrained(model_id)
    steerer.add_directions(directions)

    config = SteeringConfig(layers=[layer])
    return steerer.compare_steering(prompt, coefficients, config)


def format_functiongemma_prompt(user_query: str, tools: list[dict] | None = None) -> str:
    """Format a prompt for FunctionGemma using its expected template."""
    if tools is None:
        tools = [
            {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string", "description": "City name"}},
                    "required": ["location"],
                },
            },
            {
                "name": "send_email",
                "description": "Send an email to a recipient",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Email recipient"},
                        "subject": {"type": "string", "description": "Email subject"},
                        "body": {"type": "string", "description": "Email body"},
                    },
                    "required": ["to", "subject", "body"],
                },
            },
            {
                "name": "set_timer",
                "description": "Set a timer for a specified duration",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "duration_minutes": {
                            "type": "integer",
                            "description": "Duration in minutes",
                        }
                    },
                    "required": ["duration_minutes"],
                },
            },
        ]

    tools_json = json.dumps(tools)
    return f"""<start_of_turn>developer
You are a model that can do function calling with the following functions:
{tools_json}
<end_of_turn>
<start_of_turn>user
{user_query}
<end_of_turn>
<start_of_turn>model
"""
