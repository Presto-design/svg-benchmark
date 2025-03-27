#!/bin/bash

# Default values
PARALLEL=8
MODELS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --use-presto)
            MODELS="$MODELS --use-presto"
            shift
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
    echo "Error: No models selected. Please use at least one of: --use-claude, --use-gpt4, --use-presto"
    exit 1
fi

# Run the generation step
echo "Running generation..."
RUN_TIME=$(date +%Y-%m-%d_%H-%M-%S)
poetry run python -m svg_benchmark.generate $MODELS --parallel $PARALLEL $DRY_RUN

# Run the scoring step if not a dry run
if [ -z "$DRY_RUN" ]; then
    echo -e "\nRunning scoring..."
    poetry run python -m svg_benchmark.score --run-time "$RUN_TIME"
fi 