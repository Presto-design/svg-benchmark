import os
import csv
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from evaluate import load
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import sys

from .utils import (
    extract_data_urls,
    replace_urls,
    extract_svg,
    render_svg,
    compute_pixel_similarity,
    create_image_message,
)


def check_api_keys():
    """Check if required API keys are present"""
    load_dotenv()
    missing_keys = []

    if not os.getenv("ANTHROPIC_API_KEY"):
        missing_keys.append("ANTHROPIC_API_KEY")
    if not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")

    if missing_keys:
        print("Error: Missing required API keys:", ", ".join(missing_keys))
        print("Please copy .env.template to .env and fill in your API keys")
        sys.exit(1)


# Create output directories
Path("output").mkdir(exist_ok=True)
for model in ["claude", "gpt4"]:
    Path(f"output/{model}").mkdir(exist_ok=True)


def main():
    # Check API keys
    check_api_keys()

    # Load models
    models = {
        "claude": ChatAnthropic(model="claude-3-opus-20240229", temperature=0),
        "gpt4": ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=4096),
    }

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("Presto-Design/svg_basic_benchmark_v0")
    eval_data = dataset["test"].select(range(32))

    # Initialize results storage
    raw_results = []
    scores = []

    # Load BLEU scorer
    bleu = load("bleu")

    # Process each example
    total_examples = len(eval_data) * len(models)
    with tqdm(total=total_examples, desc="Processing examples") as pbar:
        for idx, example in enumerate(eval_data):
            for model_id, model in models.items():
                pbar.set_description(f"Processing {model_id} example {idx}")

                # Create message with image
                message_content = create_image_message(
                    example["image"],
                    width=example.get("width", 400),  # Default values if not in dataset
                    height=example.get("height", 400),
                    url_mapping={},  # No need for URL mapping as images are in the completion
                )

                try:
                    # Get model response
                    response = model.invoke([HumanMessage(content=message_content)])
                    response_text = response.content

                    # Save raw response
                    raw_results.append(
                        {"record": idx, "model": model_id, "response": response_text}
                    )

                    # Extract and process SVG
                    svg_text = extract_svg(response_text)
                    if not svg_text:
                        scores.append(
                            {
                                "record": idx,
                                "model": model_id,
                                "bleu": 0.0,
                                "structural": 0.0,
                                "pixel": 0.0,
                            }
                        )
                        pbar.update(1)
                        continue

                    # Save SVG and render to PNG
                    svg_path = f"output/{model_id}/{idx}.svg"
                    png_path = f"output/{model_id}/{idx}.png"
                    target_png_path = f"output/{model_id}/{idx}_target.png"

                    with open(svg_path, "w") as f:
                        f.write(svg_text)

                    # Render SVG
                    if not render_svg(svg_text, png_path):
                        scores.append(
                            {
                                "record": idx,
                                "model": model_id,
                                "bleu": 0.0,
                                "structural": 0.0,
                                "pixel": 0.0,
                            }
                        )
                        pbar.update(1)
                        continue

                    # Save target SVG and render to PNG
                    target_svg = example["completion"]
                    render_svg(target_svg, target_png_path)

                    # Compute BLEU score using completion as reference
                    bleu_score = bleu.compute(
                        predictions=[svg_text],
                        references=[example["completion"]],
                    )["bleu"]

                    # Compute image similarities
                    structural_sim, pixel_sim = compute_pixel_similarity(
                        png_path, target_png_path
                    )

                    # Record scores
                    scores.append(
                        {
                            "record": idx,
                            "model": model_id,
                            "bleu": bleu_score,
                            "structural": structural_sim,
                            "pixel": pixel_sim,
                        }
                    )
                except Exception as e:
                    print(f"\nError processing {model_id} example {idx}: {str(e)}")
                    scores.append(
                        {
                            "record": idx,
                            "model": model_id,
                            "bleu": 0.0,
                            "structural": 0.0,
                            "pixel": 0.0,
                        }
                    )
                finally:
                    pbar.update(1)

    print("\nSaving results...")
    # Save raw results
    with open("output/raw.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["record", "model", "response"])
        writer.writeheader()
        writer.writerows(raw_results)

    # Save detailed scores
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv("output/scores.csv", index=False)

    # Calculate and save mean scores per model
    mean_scores = scores_df.groupby("model").mean().reset_index()
    mean_scores.to_csv("output/mean_scores.csv", index=False)

    print("Generating visualization...")
    # Create visualization
    plt.figure(figsize=(10, 6))
    metrics = ["bleu", "structural", "pixel"]
    x = np.arange(len(models))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = mean_scores[metric]
        plt.bar(x + i * width, values, width, label=metric)

    plt.xlabel("Models")
    plt.ylabel("Scores")
    plt.title("Model Performance Comparison")
    plt.xticks(x + width, mean_scores["model"])
    plt.legend()

    plt.savefig("output/comparison.svg", format="svg")
    plt.close()

    print("\nBenchmark complete! Results saved in output/")


if __name__ == "__main__":
    main()
