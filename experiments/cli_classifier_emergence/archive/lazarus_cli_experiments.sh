#!/bin/bash
# Classifier Emergence Experiments using Lazarus CLI
#
# This script runs comprehensive classifier emergence detection experiments
# using the lazarus CLI commands. It checks ALL layers for classifiers.
#
# Usage:
#   ./lazarus_cli_experiments.sh [model] [--save]
#
# Models: llama3.2, tinyllama, granite, all
# Options:
#   --save   Save results to JSON files

set -e

MODEL="${1:-llama3.2}"
SAVE_RESULTS=""

# Check for --save flag
for arg in "$@"; do
    if [ "$arg" == "--save" ]; then
        SAVE_RESULTS="yes"
    fi
done

# Create results directory if saving
RESULTS_DIR="experiments/cli_classifier_emergence/results"
if [ "$SAVE_RESULTS" == "yes" ]; then
    mkdir -p "$RESULTS_DIR"
fi

# Model mappings
get_model_id() {
    case "$1" in
        llama3.2|llama)
            echo "meta-llama/Llama-3.2-1B"
            ;;
        tinyllama)
            echo "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            ;;
        granite)
            echo "ibm-granite/granite-3.1-2b-base"
            ;;
        *)
            echo "$1"
            ;;
    esac
}

# Define prompts for each operation class
MULTIPLY_PROMPTS="7 * 8 = |12 * 5 = |3 * 9 = |6 * 7 = "
ADD_PROMPTS="23 + 45 = |17 + 38 = |11 + 22 = |5 + 9 = "
SUBTRACT_PROMPTS="50 - 23 = |89 - 34 = |77 - 11 = |40 - 15 = "
DIVIDE_PROMPTS="48 / 6 = |81 / 9 = |36 / 4 = |24 / 3 = "

# Test prompts (different numbers than training)
TEST_PROMPTS="11 * 12 = |6 * 9 = |13 + 14 = |25 + 17 = |15 - 6 = |20 - 8 = |12 / 4 = |15 / 3 = "

# Logit lens prompts
LOGIT_LENS_PROMPTS="7 * 8 = |12 * 5 = |23 + 45 = |17 + 38 = |50 - 23 = |89 - 34 = |48 / 6 = |81 / 9 = "

run_experiments_for_model() {
    local MODEL_NAME="$1"
    local MODEL_ID=$(get_model_id "$MODEL_NAME")
    local SAFE_NAME=$(echo "$MODEL_NAME" | tr '/' '_' | tr '.' '_')

    echo "========================================"
    echo "CLASSIFIER EMERGENCE EXPERIMENTS"
    echo "Model: $MODEL_ID"
    echo "========================================"
    echo

    echo "========================================"
    echo "EXPERIMENT 1: Linear Probe Classification (All Layers)"
    echo "========================================"
    echo "Testing which layers can distinguish operation types"
    echo

    if [ "$SAVE_RESULTS" == "yes" ]; then
        lazarus introspect classifier -m "$MODEL_ID" \
            --classes "multiply:$MULTIPLY_PROMPTS" \
            --classes "add:$ADD_PROMPTS" \
            --classes "subtract:$SUBTRACT_PROMPTS" \
            --classes "divide:$DIVIDE_PROMPTS" \
            --test "$TEST_PROMPTS" \
            --output "$RESULTS_DIR/${SAFE_NAME}_classifier.json"
    else
        lazarus introspect classifier -m "$MODEL_ID" \
            --classes "multiply:$MULTIPLY_PROMPTS" \
            --classes "add:$ADD_PROMPTS" \
            --classes "subtract:$SUBTRACT_PROMPTS" \
            --classes "divide:$DIVIDE_PROMPTS" \
            --test "$TEST_PROMPTS"
    fi

    echo
    echo "========================================"
    echo "EXPERIMENT 2: Logit Lens Analysis"
    echo "========================================"
    echo "Checking if classifiers map to vocabulary tokens"
    echo

    if [ "$SAVE_RESULTS" == "yes" ]; then
        lazarus introspect logit-lens -m "$MODEL_ID" \
            --prompts "$LOGIT_LENS_PROMPTS" \
            --targets "multiply" \
            --targets "add" \
            --targets "subtract" \
            --targets "divide" \
            --output "$RESULTS_DIR/${SAFE_NAME}_logit_lens.json"
    else
        lazarus introspect logit-lens -m "$MODEL_ID" \
            --prompts "$LOGIT_LENS_PROMPTS" \
            --targets "multiply" \
            --targets "add" \
            --targets "subtract" \
            --targets "divide"
    fi

    echo
    echo "========================================"
    echo "SUMMARY FOR: $MODEL_ID"
    echo "========================================"
    echo
    echo "Linear Probe: Detects classifiers in hidden state space"
    echo "  - Accuracy shown at EACH layer above"
    echo "  - 100% accuracy = strong classifier signal"
    echo
    echo "Logit Lens: Checks vocabulary-mappable classifiers"
    echo "  - Target tokens appear: classifier maps to tokens"
    echo "  - Usually 0% for untrained models (expected)"
    echo
    echo "----------------------------------------"
    echo
}

# Main execution
if [ "$MODEL" == "all" ]; then
    echo "Running experiments on ALL models..."
    echo

    for m in llama3.2 tinyllama granite; do
        run_experiments_for_model "$m"
    done

    echo "========================================"
    echo "ALL EXPERIMENTS COMPLETE"
    echo "========================================"
    if [ "$SAVE_RESULTS" == "yes" ]; then
        echo "Results saved to: $RESULTS_DIR/"
        ls -la "$RESULTS_DIR/"
    fi
else
    run_experiments_for_model "$MODEL"

    if [ "$SAVE_RESULTS" == "yes" ]; then
        echo "Results saved to: $RESULTS_DIR/"
    fi
fi
