import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def create_comparison_plot(mean_scores_path: str, output_path: str):
    """Create and save visualization comparing model performances"""
    # Read mean scores
    mean_scores = pd.read_csv(mean_scores_path)

    plt.figure(figsize=(10, 6))
    metrics = ["bleu", "structural", "pixel"]
    x = np.arange(len(mean_scores))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = mean_scores[metric]
        plt.bar(x + i * width, values, width, label=metric)

    plt.xlabel("Models")
    plt.ylabel("Scores")
    plt.title("Model Performance Comparison")
    plt.xticks(x + width, mean_scores["model"])
    plt.legend()

    plt.savefig(output_path, format="svg")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize SVG benchmark results")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="output",
        help="Directory containing benchmark results CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save visualization files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Check if required files exist
    mean_scores_path = input_dir / "mean_scores.csv"
    if not mean_scores_path.exists():
        print(f"Error: Could not find mean scores file at {mean_scores_path}")
        return

    print("Generating visualization...")
    create_comparison_plot(
        mean_scores_path=mean_scores_path, output_path=output_dir / "comparison.svg"
    )
    print(f"Visualization saved to {output_dir}/comparison.svg")


if __name__ == "__main__":
    main()
