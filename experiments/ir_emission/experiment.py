"""
IR Emission Experiment.

Neural Compiler: NL → WASM IR → Execute

This experiment demonstrates that:
1. Transformers can serve as semantic frontends (NL → canonical)
2. Logit lens classification extracts operation intent (canonical → IR)
3. Deterministic runtimes handle computation (WASM execution)
4. The combination achieves Turing completeness via loops

Uses the chuk_lazarus experiments framework.
"""

import logging
from pathlib import Path

from chuk_lazarus.experiments import ExperimentBase, ExperimentConfig

# Import pipelines - use relative import from same package
from .pipelines import (
    LoopPipeline,
    MultiOpPipeline,
    NeuralCompilerBase,
    SingleOpPipeline,
)

logger = logging.getLogger(__name__)


class IREmissionExperiment(ExperimentBase):
    """
    Neural Compiler experiment.

    Runs three pipelines to test different capabilities:
    - single_op: Single arithmetic operations (100% accuracy)
    - multi_op: Multi-operation chains (100% accuracy)
    - loop: Loop constructs for Turing completeness (100% accuracy)
    """

    def setup(self) -> None:
        """Load models and prepare classifier."""
        self.log("Loading base model for normalization...")
        base_result = self.load_model()
        self.base_model = base_result.model
        self.tokenizer = base_result.tokenizer
        self.model_config = base_result.config

        # Load classifier model with LoRA using framework
        self.log("Loading classifier model with LoRA...")
        classifier_checkpoint = self.config.parameters.get(
            "classifier_checkpoint", "checkpoints/dual_reward/final"
        )
        # Handle both directory and file path formats
        classifier_path = self.config.experiment_dir / classifier_checkpoint
        if classifier_path.suffix == ".safetensors":
            classifier_path = classifier_path.parent

        if classifier_path.exists():
            # Use framework's load_model with adapter_path to handle all LoRA loading
            cls_result = self.load_model(adapter_path=str(classifier_path))
            self.cls_model = cls_result.model
            self.log(f"Loaded classifier with LoRA from {classifier_path}")
        else:
            self.log(f"Warning: Classifier checkpoint not found at {classifier_path}")
            self.log("Using base model for classification (accuracy may be lower)")
            self.cls_model = self.base_model

        # Freeze models
        self.base_model.freeze()
        self.cls_model.freeze()

        # Classifier tokens
        self.classifier_tokens = self.config.parameters.get(
            "classifier_tokens",
            {
                "add": 788,
                "subtract": 23197,
                "multiply": 22932,
                "divide": 16429,
            },
        )

        # Decision layer
        decision_pct = self.config.parameters.get("decision_layer_pct", 0.55)
        self.decision_layer = int(self.model_config.num_hidden_layers * decision_pct)
        self.log(f"Decision layer: {self.decision_layer} ({decision_pct * 100:.0f}% depth)")

        # Create compiler
        self.compiler = NeuralCompilerBase(
            base_model=self.base_model,
            cls_model=self.cls_model,
            tokenizer=self.tokenizer,
            config=self.model_config,
            classifier_tokens=self.classifier_tokens,
            decision_layer=self.decision_layer,
        )

        # Initialize pipelines
        self.pipelines = {
            "single_op": SingleOpPipeline(),
            "multi_op": MultiOpPipeline(),
            "loop": LoopPipeline(),
        }

        # Filter to configured pipelines
        enabled = self.config.parameters.get("pipelines", ["single_op", "multi_op", "loop"])
        self.enabled_pipelines = [p for p in enabled if p in self.pipelines]
        self.log(f"Enabled pipelines: {self.enabled_pipelines}")

    def run(self) -> dict:
        """Run all enabled pipelines."""
        results = {}

        for pipeline_name in self.enabled_pipelines:
            pipeline = self.pipelines[pipeline_name]
            self.log(f"Running pipeline: {pipeline_name}")

            result = pipeline.run(self.compiler)
            results[pipeline_name] = result.to_dict()

            self.log(
                f"  {pipeline_name}: {result.passed}/{result.total_tests} "
                f"({result.accuracy * 100:.1f}%)"
            )

        return results

    def evaluate(self) -> dict:
        """Compute aggregate metrics across all pipelines."""
        # Load results from run()
        latest = self.load_latest_results("results")
        if not latest:
            self.log("No results found to evaluate")
            return {}

        total_tests = 0
        total_passed = 0
        pipeline_accuracies = {}

        for pipeline_name, pipeline_results in latest.items():
            if isinstance(pipeline_results, dict) and "total_tests" in pipeline_results:
                total_tests += pipeline_results["total_tests"]
                total_passed += pipeline_results["passed"]
                pipeline_accuracies[pipeline_name] = pipeline_results["accuracy"]

        overall_accuracy = total_passed / total_tests if total_tests > 0 else 0

        return {
            "overall_accuracy": overall_accuracy,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "pipeline_accuracies": pipeline_accuracies,
        }

    def cleanup(self) -> None:
        """Release model resources."""
        self.log("Cleaning up...")
        # MLX handles memory automatically, but we can clear references
        self.base_model = None
        self.cls_model = None
        self.compiler = None


# For backwards compatibility with direct script execution
if __name__ == "__main__":
    from pathlib import Path

    import yaml

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    config = ExperimentConfig(
        experiment_dir=Path(__file__).parent,
        **config_data,
    )

    # Run experiment
    experiment = IREmissionExperiment(config)
    experiment.setup()
    results = experiment.run()
    eval_results = experiment.evaluate()
    experiment.cleanup()

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {eval_results.get('overall_accuracy', 0) * 100:.1f}%")
    print(f"Tests: {eval_results.get('total_passed', 0)}/{eval_results.get('total_tests', 0)}")

    for name, acc in eval_results.get("pipeline_accuracies", {}).items():
        print(f"  {name}: {acc * 100:.1f}%")
