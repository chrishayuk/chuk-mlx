#!/bin/bash
# Classifier Emergence Experiment Runner
#
# This script runs the complete classifier emergence experiment using the Lazarus CLI.
#
# Phases:
#   1. Generate training data
#   2. Verify base model has NO vocab-aligned classifiers (0% logit lens)
#   3. Train Phase 1: Dual-reward V/O training (creates classifiers)
#   4. Verify trained model HAS vocab-aligned classifiers
#   5. Train Phase 2: Freeze classifier, train routing layers
#   6. Verify routing works (correct answers)
#
# Usage:
#   ./experiments/cli_classifier_emergence/run_experiment.sh [phase]
#
# Options:
#   all       Run all phases (default)
#   generate  Generate training data only
#   baseline  Run baseline measurements only
#   phase1    Run Phase 1 training only
#   phase2    Run Phase 2 training only
#   verify    Run verification only

set -e

EXPERIMENT_DIR="experiments/cli_classifier_emergence"
DATA_DIR="$EXPERIMENT_DIR/data"
CHECKPOINT_DIR="$EXPERIMENT_DIR/checkpoints"
RESULTS_DIR="$EXPERIMENT_DIR/results"

MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Create directories
mkdir -p "$DATA_DIR" "$CHECKPOINT_DIR" "$RESULTS_DIR"

run_generate() {
    echo "========================================"
    echo "PHASE 0: Generate Training Data"
    echo "========================================"
    python "$EXPERIMENT_DIR/generate_data.py" \
        --output "$DATA_DIR/arithmetic_sft.jsonl" \
        --samples 1000
    echo ""
}

run_baseline() {
    echo "========================================"
    echo "BASELINE: Verify Base Model"
    echo "========================================"
    echo "Checking linear probe (should be 100%)..."
    lazarus introspect classifier -m "$MODEL" \
        --classes "multiply:7 * 8 = |12 * 5 = |3 * 9 = |6 * 7 = " \
        --classes "add:23 + 45 = |17 + 38 = |11 + 22 = |5 + 9 = " \
        --classes "subtract:50 - 23 = |89 - 34 = |77 - 11 = |40 - 15 = " \
        --classes "divide:48 / 6 = |81 / 9 = |36 / 4 = |24 / 3 = " \
        --test "11 * 12 = |6 * 9 = |13 + 14 = |25 + 17 = " \
        --output "$RESULTS_DIR/baseline_classifier.json"

    echo ""
    echo "Checking logit lens (should be 0%)..."
    lazarus introspect logit-lens -m "$MODEL" \
        --prompts "7 * 8 = |12 * 5 = |23 + 45 = |17 + 38 = " \
        --targets "multiply" --targets "add" --targets "subtract" --targets "divide" \
        --output "$RESULTS_DIR/baseline_logit_lens.json"
    echo ""
}

run_phase1() {
    echo "========================================"
    echo "PHASE 1: Dual-Reward V/O Training"
    echo "========================================"
    echo "Training V/O projections with classification loss..."
    echo ""
    echo "NOTE: This requires the dual-reward training script."
    echo "The generic CLI doesn't yet support intermediate-layer loss."
    echo ""
    echo "Run manually:"
    echo "  python $EXPERIMENT_DIR/train_dual_reward.py \\"
    echo "    --model $MODEL \\"
    echo "    --output $CHECKPOINT_DIR/phase1_classifier \\"
    echo "    --steps 500 \\"
    echo "    --cls-weight 0.4"
    echo ""
}

run_phase2() {
    echo "========================================"
    echo "PHASE 2: Frozen Classifier Routing"
    echo "========================================"
    echo "Training routing layers with frozen classifier..."
    echo ""
    echo "Run:"
    echo "  lazarus train sft \\"
    echo "    --model $MODEL \\"
    echo "    --data $DATA_DIR/arithmetic_sft.jsonl \\"
    echo "    --freeze-layers 0-12 \\"
    echo "    --use-lora \\"
    echo "    --lora-targets v_proj,o_proj \\"
    echo "    --output $CHECKPOINT_DIR/phase2_routing \\"
    echo "    --max-steps 300"
    echo ""
}

run_verify() {
    echo "========================================"
    echo "VERIFY: Check Results"
    echo "========================================"

    if [ -f "$RESULTS_DIR/baseline_classifier.json" ]; then
        echo "Baseline classifier results:"
        cat "$RESULTS_DIR/baseline_classifier.json" | python -m json.tool | head -20
    fi

    if [ -f "$RESULTS_DIR/baseline_logit_lens.json" ]; then
        echo ""
        echo "Baseline logit lens results:"
        cat "$RESULTS_DIR/baseline_logit_lens.json" | python -m json.tool | head -20
    fi
}

# Main
PHASE="${1:-all}"

case "$PHASE" in
    generate)
        run_generate
        ;;
    baseline)
        run_baseline
        ;;
    phase1)
        run_phase1
        ;;
    phase2)
        run_phase2
        ;;
    verify)
        run_verify
        ;;
    all)
        run_generate
        run_baseline
        run_phase1
        run_phase2
        run_verify
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Options: all, generate, baseline, phase1, phase2, verify"
        exit 1
        ;;
esac

echo "Done!"
