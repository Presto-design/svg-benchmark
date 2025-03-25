#!/bin/bash

# Default values
PARALLEL=8
MODELS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --presto-model)
            PRESTO_MODEL="$2"
            MODELS="$MODELS --presto-model $2"
            shift 2
            ;;
        --use-claude)
            MODELS="$MODELS --use-claude"
            shift
            ;;
        --use-gpt4)
            MODELS="$MODELS --use-gpt4"
            shift
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if at least one model is selected
if [ -z "$MODELS" ]; then
    echo "Error: No models selected. Please use at least one of: --use-claude, --use-gpt4, --presto-model <path>"
    exit 1
fi

# Run the benchmark
poetry run python -m svg_benchmark.benchmark $MODELS --parallel $PARALLEL $DRY_RUN 