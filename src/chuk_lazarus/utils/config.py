"""Unified configuration utilities."""

import json
import logging
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

logger = logging.getLogger(__name__)

T = TypeVar("T")


def load_config(path: str, config_class: type[T] = None) -> T | dict:
    """
    Load configuration from YAML or JSON file.

    Args:
        path: Path to config file
        config_class: Optional dataclass to instantiate

    Returns:
        Config dataclass instance or dict
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        if path.suffix in [".yaml", ".yml"]:
            raw_config = yaml.safe_load(f)
        elif path.suffix == ".json":
            raw_config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    if config_class is None:
        return raw_config

    # Map to dataclass
    return _dict_to_dataclass(raw_config, config_class)


def save_config(config: Any, path: str):
    """
    Save configuration to YAML or JSON file.

    Args:
        config: Config dict or dataclass
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if is_dataclass(config):
        data = asdict(config)
    else:
        data = config

    with open(path, "w") as f:
        if path.suffix in [".yaml", ".yml"]:
            yaml.safe_dump(data, f, default_flow_style=False)
        elif path.suffix == ".json":
            json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    logger.info(f"Saved config to {path}")


def _dict_to_dataclass(data: dict, cls: type[T]) -> T:
    """Convert dict to dataclass, handling nested structures."""
    if not is_dataclass(cls):
        raise ValueError(f"{cls} is not a dataclass")

    field_names = {f.name: f for f in fields(cls)}
    kwargs = {}

    # Handle flat dict
    for key, value in data.items():
        if key in field_names:
            field = field_names[key]
            # Handle nested dataclass
            if is_dataclass(field.type) and isinstance(value, dict):
                kwargs[key] = _dict_to_dataclass(value, field.type)
            else:
                kwargs[key] = value
        elif isinstance(value, dict):
            # Handle legacy nested config (flatten it)
            for nested_key, nested_value in value.items():
                if nested_key in field_names:
                    kwargs[nested_key] = nested_value

    return cls(**kwargs)


def merge_configs(*configs: dict) -> dict:
    """Merge multiple config dicts, later ones override earlier."""
    result = {}
    for config in configs:
        _deep_merge(result, config)
    return result


def _deep_merge(base: dict, override: dict):
    """Deep merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
