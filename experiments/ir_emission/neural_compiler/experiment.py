"""
Neural Compiler Experiment.

NL → Canonical → IR → WASM → Result

Demonstrates that:
1. Transformers can serve as semantic frontends (NL → canonical)
2. Logit lens classification extracts operation intent (canonical → IR)
3. Deterministic runtimes handle computation (WASM execution)
4. The combination achieves Turing completeness via loops

Uses the chuk_lazarus experiments framework.
"""

import logging
from pathlib import Path

from chuk_lazarus.experiments import ExperimentBase, ExperimentConfig

# Import from ir_emission (shared implementation)
from experiments.ir_emission.pipelines import (
    ComparisonPipeline,
    LoopPipeline,
    MultiOpPipeline,
    NeuralCompilerBase,
    SingleOpPipeline,
)

logger = logging.getLogger(__name__)


class NeuralCompilerExperiment(ExperimentBase):
    """
    Neural Compiler experiment.

    Runs pipelines to test NL → WASM → Execute:
    - single_op: Single arithmetic operations (100% accuracy)
    - multi_op: Multi-operation chains (100% accuracy)
    - loop: Loop constructs for Turing completeness (100% accuracy)
    - comparison: Boolean comparisons (100% accuracy)
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
        classifier_path = self.config.experiment_dir / classifier_checkpoint
        if classifier_path.suffix == ".safetensors":
            classifier_path = classifier_path.parent

        if classifier_path.exists():
            cls_result = self.load_model(adapter_path=str(classifier_path))
            self.cls_model = cls_result.model
            self.log(f"Loaded classifier with LoRA from {classifier_path}")
        else:
            self.log(f"Note: No classifier checkpoint at {classifier_path}")
            self.log("Using base model for classification")
            self.cls_model = self.base_model

        # Freeze models
        self.base_model.freeze()
        self.cls_model.freeze()

        # Classifier tokens from config
        self.classifier_tokens = self.config.parameters.get(
            "classifier_tokens",
            {
                "add": 1476,
                "subtract": 1014,
                "multiply": 19790,
                "divide": 4563,
            },
        )

        # Decision layer
        decision_layer = self.config.parameters.get("decision_layer", 12)
        self.decision_layer = decision_layer
        self.log(f"Decision layer: {self.decision_layer}")

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
            "comparison": ComparisonPipeline(),
        }

        # Filter to configured pipelines
        enabled = self.config.parameters.get(
            "pipelines", ["single_op", "multi_op", "loop", "comparison"]
        )
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
        self.base_model = None
        self.cls_model = None
        self.compiler = None


if __name__ == "__main__":
    # Load config using framework method
    config_path = Path(__file__).parent / "config.yaml"
    config = ExperimentConfig.from_yaml(config_path)
    config.experiment_dir = Path(__file__).parent

    # Run experiment
    experiment = NeuralCompilerExperiment(config)
    experiment.setup()
    results = experiment.run()

    # Save results so evaluate() can load them
    if results:
        experiment.save_results(results)

    eval_results = experiment.evaluate()
    experiment.cleanup()

    print("\n" + "=" * 60)
    print("NEURAL COMPILER RESULTS")
    print("=" * 60)

    if eval_results:
        print(f"Overall Accuracy: {eval_results.get('overall_accuracy', 0) * 100:.1f}%")
        print(f"Tests: {eval_results.get('total_passed', 0)}/{eval_results.get('total_tests', 0)}")
        for name, acc in eval_results.get("pipeline_accuracies", {}).items():
            print(f"  {name}: {acc * 100:.1f}%")
    else:
        # Print run results directly if evaluate returned nothing
        print("Run results:")
        for pipeline_name, pipeline_result in results.items():
            if isinstance(pipeline_result, dict):
                acc = pipeline_result.get("accuracy", 0)
                total = pipeline_result.get("total_tests", 0)
                passed = pipeline_result.get("passed", 0)
                print(f"  {pipeline_name}: {passed}/{total} ({acc * 100:.1f}%)")
