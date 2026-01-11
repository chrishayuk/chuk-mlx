"""
Base classes for the experiments framework.

ExperimentConfig - Configuration dataclass for experiments
ExperimentBase - Abstract base class that experiments must inherit from
ExperimentResult - Structured result from running an experiment
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment.

    Required fields:
        name: Unique experiment identifier (matches directory name)
        description: Human-readable description

    Optional fields:
        model: Model path or HuggingFace ID
        training: Training configuration passed to trainers
        parameters: Experiment-specific parameters

    Auto-populated:
        experiment_dir: Path to experiment directory
        data_dir: Path to data/ subdirectory
        checkpoint_dir: Path to checkpoints/ subdirectory
        results_dir: Path to results/ subdirectory
    """

    name: str
    description: str

    # Model settings
    model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Training settings (passed to existing trainers)
    training: dict = field(default_factory=dict)

    # Experiment-specific parameters
    parameters: dict = field(default_factory=dict)

    # Paths (auto-populated by framework)
    experiment_dir: Path | None = None
    data_dir: Path | None = None
    checkpoint_dir: Path | None = None
    results_dir: Path | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """Load config from YAML file.

        Extra fields not in the dataclass are added to parameters.
        """
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        # Known fields
        known_fields = {
            "name",
            "description",
            "model",
            "training",
            "parameters",
            "experiment_dir",
            "data_dir",
            "checkpoint_dir",
            "results_dir",
        }

        # Separate known and extra fields
        known = {}
        extra = {}
        for key, value in data.items():
            if key in known_fields:
                known[key] = value
            else:
                extra[key] = value

        # Merge extra into parameters
        if extra:
            if "parameters" not in known:
                known["parameters"] = {}
            known["parameters"].update(extra)

        return cls(**known)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "training": self.training,
            "parameters": self.parameters,
            "experiment_dir": str(self.experiment_dir) if self.experiment_dir else None,
            "data_dir": str(self.data_dir) if self.data_dir else None,
            "checkpoint_dir": str(self.checkpoint_dir) if self.checkpoint_dir else None,
            "results_dir": str(self.results_dir) if self.results_dir else None,
        }


@dataclass
class ExperimentResult:
    """Result from running an experiment."""

    experiment_name: str
    status: str  # "success", "failed", "partial"
    started_at: str
    finished_at: str
    duration_seconds: float
    run_results: dict  # Results from run()
    eval_results: dict  # Results from evaluate()
    config: dict  # Config snapshot
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_name": self.experiment_name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "run_results": self.run_results,
            "eval_results": self.eval_results,
            "config": self.config,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentResult":
        """Create from dictionary."""
        return cls(**data)


class ExperimentBase(ABC):
    """Abstract base class for experiments.

    Experiments must implement:
        - setup(): Initialize resources (models, data)
        - run(): Main experiment logic, returns results dict
        - evaluate(): Compute final metrics, returns metrics dict

    Optional:
        - cleanup(): Release resources

    Built-in utilities:
        - load_model(): Load model using lazarus infrastructure
        - create_trainer(): Create trainer (sft, dpo, grpo, etc.)
        - save_results(): Save results to JSON
        - log(): Structured logging
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._setup_paths()
        self._logger = logging.getLogger(f"experiment.{config.name}")

    def _setup_paths(self) -> None:
        """Ensure all paths are set up."""
        if self.config.experiment_dir:
            exp_dir = Path(self.config.experiment_dir)

            if self.config.data_dir is None:
                self.config.data_dir = exp_dir / "data"

            if self.config.checkpoint_dir is None:
                self.config.checkpoint_dir = exp_dir / "checkpoints"

            if self.config.results_dir is None:
                self.config.results_dir = exp_dir / "results"

            # Create directories
            self.config.data_dir.mkdir(parents=True, exist_ok=True)
            self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.config.results_dir.mkdir(parents=True, exist_ok=True)

    # === Required methods ===

    @abstractmethod
    def setup(self) -> None:
        """Initialize resources before running.

        Called before run(). Use this to:
        - Load models
        - Load or generate data
        - Initialize any state
        """
        pass

    @abstractmethod
    def run(self) -> dict:
        """Run the main experiment logic.

        Returns:
            Dictionary of results from the experiment run.
        """
        pass

    @abstractmethod
    def evaluate(self) -> dict:
        """Compute final evaluation metrics.

        Called after run(). Use this to:
        - Calculate accuracy, loss, or other metrics
        - Compare against baselines
        - Generate summary statistics

        Returns:
            Dictionary of evaluation metrics.
        """
        pass

    # === Optional hooks ===

    def cleanup(self) -> None:  # noqa: B027
        """Release resources after running.

        Called after evaluate(), even if run() or evaluate() failed.
        """
        pass

    # === Built-in utilities ===

    def load_model(self, model_path: str | None = None, adapter_path: str | None = None):
        """Load a model using lazarus infrastructure.

        Args:
            model_path: Model path or HuggingFace ID.
                       Defaults to config.model.
            adapter_path: Optional path to LoRA adapter weights.
                         If provided, loads and applies adapter.

        Returns:
            LoadedModel with model, tokenizer, config attributes.
        """
        from ..models_v2 import load_model

        path = model_path or self.config.model
        self.log(f"Loading model: {path}")
        return load_model(path, adapter_path=adapter_path)

    def load_model_with_lora(self, model_path: str | None = None, adapter_path: str | None = None):
        """Load a model with LoRA adapter for training.

        Creates fresh LoRA layers (for training). If adapter_path is provided,
        loads pre-trained adapter weights into the LoRA layers.

        For inference with a trained adapter, use load_model(adapter_path=...) instead.

        Args:
            model_path: Base model path. Defaults to config.model.
            adapter_path: Path to pre-trained LoRA adapter weights.

        Returns:
            LoadedModelWithLoRA with model, tokenizer, config, lora_layers.
        """
        from ..models_v2 import load_model_with_lora
        from ..models_v2.adapters.lora import LoRAConfig
        from ..models_v2.loader import AdapterConfig

        path = model_path or self.config.model

        # If adapter_path provided, load its config to get LoRA parameters
        if adapter_path:
            adapter_cfg = AdapterConfig.from_directory(adapter_path)
            lora_config = LoRAConfig(
                rank=adapter_cfg.rank,
                alpha=adapter_cfg.alpha,
                target_modules=adapter_cfg.target_modules,
            )
        else:
            # Default LoRA config from training settings
            training = self.config.training or {}
            lora_config = LoRAConfig(
                rank=training.get("lora_rank", 8),
                alpha=training.get("lora_alpha", 16.0),
                target_modules=training.get("lora_targets", ["q_proj", "v_proj"]),
            )

        self.log(f"Loading model with LoRA: {path}")
        return load_model_with_lora(path, lora_config, adapter_path=adapter_path)

    def create_trainer(self, trainer_type: str, model, tokenizer, **kwargs):
        """Create a trainer from the lazarus training infrastructure.

        Args:
            trainer_type: One of "sft", "dpo", "grpo", "ppo"
            model: The model to train
            tokenizer: The tokenizer
            **kwargs: Additional arguments for the trainer config

        Returns:
            Configured trainer instance.
        """
        from ..training.trainers import DPOTrainer, GRPOTrainer, SFTTrainer

        # Merge with training config from experiment config
        merged_kwargs = {**self.config.training, **kwargs}

        trainers = {
            "sft": SFTTrainer,
            "dpo": DPOTrainer,
            "grpo": GRPOTrainer,
        }

        if trainer_type not in trainers:
            raise ValueError(
                f"Unknown trainer type: {trainer_type}. Available: {list(trainers.keys())}"
            )

        trainer_class = trainers[trainer_type]
        self.log(f"Creating {trainer_type} trainer")
        return trainer_class(model=model, tokenizer=tokenizer, **merged_kwargs)

    def save_results(self, results: dict, name: str = "results") -> Path:
        """Save results to JSON file in results directory.

        Args:
            results: Dictionary of results to save
            name: Base name for the file (timestamp will be appended)

        Returns:
            Path to saved file.
        """
        if not self.config.results_dir:
            raise ValueError("results_dir not set")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        path = self.config.results_dir / filename

        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        self.log(f"Saved results to: {path}")
        return path

    def load_latest_results(self, name: str = "results") -> dict | None:
        """Load the most recent results file.

        Args:
            name: Base name pattern to match

        Returns:
            Dictionary of results or None if no results found.
        """
        if not self.config.results_dir:
            return None

        results_files = sorted(
            self.config.results_dir.glob(f"{name}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not results_files:
            return None

        with open(results_files[0]) as f:
            return json.load(f)

    def log(self, msg: str, level: str = "info") -> None:
        """Log a message with experiment context.

        Args:
            msg: Message to log
            level: Log level ("debug", "info", "warning", "error")
        """
        log_fn = getattr(self._logger, level, self._logger.info)
        log_fn(f"[{self.config.name}] {msg}")

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get an experiment-specific parameter.

        Args:
            key: Parameter key (supports dot notation: "model.layers")
            default: Default value if not found

        Returns:
            Parameter value or default.
        """
        keys = key.split(".")
        value = self.config.parameters

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value
