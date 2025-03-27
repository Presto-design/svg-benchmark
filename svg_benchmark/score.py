import os
import pandas as pd
import numpy as np
from evaluate import load
from pathlib import Path
import argparse
import tempfile
from datasets import load_dataset
from .utils import compute_pixel_similarity, render_svg


def get_failure_type(row):
    """
    Returns:
    - 'render' if SVG exists but PNG doesn't
    - 'generation' if both SVG and PNG are missing
    - None if not a failure
    """
    svg_exists = Path(f"output/{row['model']}/{row['record']}.svg").exists()
    png_exists = Path(f"output/{row['model']}/{row['record']}.png").exists()

    if svg_exists and not png_exists:
        return "render"
    elif not svg_exists and not png_exists:
        return "generation"
    return None


def is_failure(row):
    """Check if a record represents a failure (all scores are zero)"""
    return row["bleu"] == 0 and row["structural"] == 0 and row["pixel"] == 0


def format_float(x):
    """Format float to 3 significant figures"""
    if isinstance(x, (int, float)):
        return float(f"{float(x):.3g}")
    return x


def main():
    # Load dataset for target SVGs
    print("Loading dataset...")
    dataset = load_dataset("Presto-Design/svg_basic_benchmark_v0")
    eval_data = dataset["test"]
    eval_data = [dict(example) for example in eval_data][:32]  # Get first 32 examples

    # Initialize BLEU scorer
    bleu = load("bleu")

    # Initialize results list
    results = []

    # Get all model directories
    output_dir = Path("output")
    if not output_dir.exists():
        print("Error: output directory not found. Please run generate.py first.")
        return

    model_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
    if not model_dirs:
        print(
            "Error: No model directories found in output/. Please run generate.py first."
        )
        return

    # Create temporary directory for target PNGs
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Pre-render all target SVGs
        print("\nRendering target SVGs...")
        for record in range(32):
            target_svg = eval_data[record]["completion"]
            target_png_path = temp_dir / f"{record}_target.png"
            if not render_svg(target_svg, str(target_png_path)):
                print(f"Error: Failed to render target SVG for record {record}")
                return

        # Process each model's results
        for model_dir in model_dirs:
            model = model_dir.name
            print(f"\nProcessing {model} results...")

            # Process all 32 records for this model
            for record in range(32):
                svg_file = model_dir / f"{record}.svg"
                png_path = model_dir / f"{record}.png"
                target_png_path = temp_dir / f"{record}_target.png"

                # If any required files are missing, record zero scores
                missing_files = []
                if not svg_file.exists():
                    missing_files.append(f"{record}.svg")
                if not png_path.exists():
                    missing_files.append(f"{record}.png")

                if missing_files:
                    print(
                        f"Warning: Missing files for {model} record {record}: {', '.join(missing_files)}, recording zero scores"
                    )
                    results.append(
                        {
                            "record": record,
                            "model": model,
                            "bleu": 0.0,
                            "structural": 0.0,
                            "pixel": 0.0,
                        }
                    )
                    continue

                try:
                    # Read generated SVG
                    with open(svg_file) as f:
                        generated_svg = f.read()

                    # Get target SVG from dataset
                    target_svg = eval_data[record]["completion"]

                    # Compute BLEU score
                    bleu_score = bleu.compute(
                        predictions=[generated_svg],
                        references=[target_svg],
                    )["bleu"]

                    # Compute image similarities
                    structural_sim, pixel_sim = compute_pixel_similarity(
                        str(png_path), str(target_png_path)
                    )

                    results.append(
                        {
                            "record": record,
                            "model": model,
                            "bleu": bleu_score,
                            "structural": structural_sim,
                            "pixel": pixel_sim,
                        }
                    )

                except Exception as e:
                    print(f"Error processing {model} record {record}: {str(e)}")
                    results.append(
                        {
                            "record": record,
                            "model": model,
                            "bleu": 0.0,
                            "structural": 0.0,
                            "pixel": 0.0,
                        }
                    )

    if not results:
        print("No results were processed. Please check the output directory structure.")
        return

    # Convert results to DataFrame
    scores_df = pd.DataFrame(results)

    # Format numeric columns to 3 significant figures
    numeric_cols = ["bleu", "structural", "pixel"]
    for col in numeric_cols:
        scores_df[col] = scores_df[col].apply(format_float)

    # Verify we have the expected number of records
    expected_records = 32 * len(model_dirs)
    if len(scores_df) != expected_records:
        print(f"Error: Expected {expected_records} records but found {len(scores_df)}")
        return

    # Save detailed scores
    scores_df.to_csv("output/scores.csv", index=False)

    # Calculate and save mean scores per model
    mean_scores = (
        scores_df.groupby("model")
        .agg(
            {
                "bleu": "mean",
                "structural": "mean",
                "pixel": "mean",
                "record": "count",
            }
        )
        .rename(columns={"record": "count"})
        .reset_index()
    )

    # Add scores for successful examples only (filtering out failures)
    successful_scores = scores_df[~scores_df.apply(is_failure, axis=1)]
    if not successful_scores.empty:
        successful_means = (
            successful_scores.groupby("model")
            .agg(
                {
                    "bleu": "mean",
                    "structural": "mean",
                    "pixel": "mean",
                    "record": "count",
                }
            )
            .rename(
                columns={
                    "bleu": "bleu_successful",
                    "structural": "structural_successful",
                    "pixel": "pixel_successful",
                    "record": "successful_count",
                }
            )
            .reset_index()
        )
        mean_scores = mean_scores.merge(successful_means, on="model", how="left")
    else:
        # If no successful examples, add columns with zeros
        mean_scores["bleu_successful"] = 0
        mean_scores["structural_successful"] = 0
        mean_scores["pixel_successful"] = 0
        mean_scores["successful_count"] = 0

    # Add failure counts
    failure_types = scores_df.apply(get_failure_type, axis=1)
    render_failures = failure_types.value_counts().get("render", 0)
    generation_failures = failure_types.value_counts().get("generation", 0)

    failure_counts = pd.DataFrame(
        {
            "model": scores_df["model"].unique(),
            "render_failures": [
                failure_types[failure_types == "render"][
                    scores_df["model"] == model
                ].count()
                for model in scores_df["model"].unique()
            ],
            "generation_failures": [
                failure_types[failure_types == "generation"][
                    scores_df["model"] == model
                ].count()
                for model in scores_df["model"].unique()
            ],
        }
    )

    mean_scores = mean_scores.merge(failure_counts, on="model")

    # Format numeric columns to 3 significant figures
    numeric_cols = [
        "bleu",
        "structural",
        "pixel",
        "bleu_successful",
        "structural_successful",
        "pixel_successful",
    ]
    for col in numeric_cols:
        mean_scores[col] = mean_scores[col].apply(format_float)

    # Verify each model has exactly 32 records
    if not all(mean_scores["count"] == 32):
        print("Error: Not all models have exactly 32 records")
        return

    # Reorder columns
    mean_scores = mean_scores[
        [
            "model",
            "count",
            "successful_count",
            "render_failures",
            "generation_failures",
            "bleu",
            "structural",
            "pixel",
            "bleu_successful",
            "structural_successful",
            "pixel_successful",
        ]
    ]

    # Save mean scores
    mean_scores.to_csv("output/mean_scores.csv", index=False)

    # Print sorted scores by metric for successful examples
    print("\nScores for successful examples only (sorted by metric):")
    metrics = [
        ("BLEU", "bleu_successful"),
        ("Structural Similarity", "structural_successful"),
        ("Pixel Similarity", "pixel_successful"),
    ]

    for metric_name, col in metrics:
        print(f"\n{metric_name} Scores:")
        sorted_scores = mean_scores.sort_values(col, ascending=False)[
            ["model", col, "successful_count"]
        ]
        for _, row in sorted_scores.iterrows():
            print(
                f"  {row['model']}: {row[col]} ({row['successful_count']}/32 examples)"
            )

    print("\nScoring complete!")
    print("Results saved to:")
    print("- output/scores.csv")
    print("- output/mean_scores.csv")


if __name__ == "__main__":
    main()
