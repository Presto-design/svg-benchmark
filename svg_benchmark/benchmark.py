import copy
import os
import csv
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from evaluate import load
import argparse
import asyncio
import aiohttp
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import sys
import json

from .models.presto_model import PrestoModel
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
    missing_keys = []

    if not os.getenv("ANTHROPIC_API_KEY"):
        missing_keys.append("ANTHROPIC_API_KEY")
    if not os.getenv("OPENAI_API_KEY"):
        missing_keys.append("OPENAI_API_KEY")

    if missing_keys:
        print("Error: Missing required API keys:", ", ".join(missing_keys))
        print("Please copy .env.template to .env and fill in your API keys")
        sys.exit(1)


async def process_example(
    example: Dict[str, Any],
    model_id: str,
    model: Any,
    idx: int,
    args: argparse.Namespace,
    bleu: Optional[Any] = None,
) -> Dict[str, Any]:
    """Process a single example asynchronously"""
    try:
        # Validate example structure
        if not isinstance(example, dict):
            raise ValueError(f"Example must be a dictionary, got {type(example)}")
        if "image" not in example or "completion" not in example:
            raise ValueError(
                f"Example missing required keys. Found keys: {list(example.keys())}"
            )
        if not example["completion"]:
            raise ValueError("Example completion is empty")

        # Extract data URLs from completion and create mapping
        url_mapping = extract_data_urls(example["completion"])

        # Validate image object
        if not hasattr(example["image"], "width") or not hasattr(
            example["image"], "height"
        ):
            raise ValueError(f"Invalid image object: missing width/height attributes")

        message_content = create_image_message(
            example["image"],
            width=example["image"].width,
            height=example["image"].height,
            url_mapping=url_mapping,
        )

        if args.dry_run:
            print(f"\n\n=== Model Call for {model_id} example {idx} ===")
            print("Message content:")
            print("Data URLs found in completion:", len(url_mapping))
            message_content_display = copy.deepcopy(message_content)
            for item in message_content_display:
                if item.get("type") == "image_url" and "url" in item.get(
                    "image_url", {}
                ):
                    url = item["image_url"]["url"]
                    if url.startswith("data:"):
                        item["image_url"]["url"] = url[:30] + "..." + url[-10:]
            print(json.dumps(message_content_display, indent=2))
            return None

        # Get model response
        response = model.invoke([HumanMessage(content=message_content)])
        response_text = response.content

        # Extract and process SVG
        svg_text = extract_svg(response_text)
        if not svg_text:
            return {
                "record": idx,
                "model": model_id,
                "bleu": 0.0,
                "structural": 0.0,
                "pixel": 0.0,
                "raw_response": response_text,
            }

        # Save SVG and render to PNG
        svg_path = f"output/{model_id}/{idx}.svg"
        png_path = f"output/{model_id}/{idx}.png"
        target_png_path = f"output/{model_id}/{idx}_target.png"

        with open(svg_path, "w") as f:
            f.write(svg_text)

        # Render SVG
        if not render_svg(svg_text, png_path):
            return {
                "record": idx,
                "model": model_id,
                "bleu": 0.0,
                "structural": 0.0,
                "pixel": 0.0,
                "raw_response": response_text,
            }

        # Save target SVG and render to PNG
        target_svg = example["completion"]
        render_svg(target_svg, target_png_path)

        # Compute BLEU score using completion as reference
        bleu_score = bleu.compute(
            predictions=[svg_text],
            references=[example["completion"]],
        )["bleu"]

        # Compute image similarities
        structural_sim, pixel_sim = compute_pixel_similarity(png_path, target_png_path)

        return {
            "record": idx,
            "model": model_id,
            "bleu": bleu_score,
            "structural": structural_sim,
            "pixel": pixel_sim,
            "raw_response": response_text,
        }

    except Exception as e:
        print(f"\nError processing {model_id} example {idx}: {str(e)}")
        return {
            "record": idx,
            "model": model_id,
            "bleu": 0.0,
            "structural": 0.0,
            "pixel": 0.0,
            "raw_response": str(e),
        }


async def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run SVG generation benchmark")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print model inputs without running"
    )
    parser.add_argument(
        "--presto-model", type=str, help="Path to local Presto model checkpoints"
    )
    parser.add_argument(
        "--parallel", type=int, default=20, help="Number of parallel calls per model"
    )
    args = parser.parse_args()

    # Define model configurations
    model_configs = {
        "claude": {
            "class": ChatAnthropic,
            "params": {"model": "claude-3-7-sonnet-20250219", "temperature": 0},
        },
        "gpt4": {
            "class": ChatOpenAI,
            "params": {"model": "gpt-4o", "temperature": 0, "max_tokens": 4096},
        },
    }

    # Add Presto model if path provided
    if args.presto_model:
        print(f"Loading Presto model from {args.presto_model}...")
        model_configs["presto"] = {
            "class": PrestoModel,
            "params": {"model_path": args.presto_model},
        }

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("Presto-Design/svg_basic_benchmark_v0")
    eval_data = dataset["test"]

    # Convert to list of dictionaries
    eval_data = [dict(example) for example in eval_data]
    if args.dry_run:
        eval_data = eval_data[:6]
    else:
        eval_data = eval_data[:32]

    # Verify dataset format
    if not eval_data:
        raise ValueError("Failed to load dataset")

    first_example = eval_data[0]
    if (
        not isinstance(first_example, dict)
        or "image" not in first_example
        or "completion" not in first_example
    ):
        raise ValueError(f"Invalid dataset format. Example structure: {first_example}")

    # Calculate total examples to process
    total_examples = len(eval_data) * len(model_configs)

    # Initialize models if not in dry-run mode
    models = {}
    if not args.dry_run:
        check_api_keys()
        for model_id, config in model_configs.items():
            models[model_id] = config["class"](**config["params"])

        # Create output directories
        Path("output").mkdir(exist_ok=True)
        for model in model_configs:
            Path(f"output/{model}").mkdir(exist_ok=True)

    # Initialize results storage
    results = []

    # Load BLEU scorer if not in dry-run mode
    bleu = None if args.dry_run else load("bleu")

    # Process examples in parallel batches
    with tqdm(total=total_examples, desc="Processing examples") as pbar:
        for model_id, config in model_configs.items():
            model = models.get(model_id)
            if args.dry_run:
                print(f"\nProcessing model: {model_id}")
                print(f"Model config: {config}")
                print(f"First example type: {type(eval_data[0])}")
                print(
                    f"First example keys: {list(eval_data[0].keys()) if isinstance(eval_data[0], dict) else 'Not a dict'}"
                )

            # Process examples in parallel batches
            for i in range(0, len(eval_data), args.parallel):
                batch = eval_data[i : i + args.parallel]
                if args.dry_run:
                    print(f"\nBatch {i//args.parallel + 1}:")
                    print(f"Batch size: {len(batch)}")
                    print(f"First batch item type: {type(batch[0])}")

                tasks = [
                    process_example(
                        example,  # Already a dictionary
                        model_id,
                        model,
                        idx + i,
                        args,
                        bleu,
                    )
                    for idx, example in enumerate(batch)
                ]

                batch_results = await asyncio.gather(*tasks)
                results.extend([r for r in batch_results if r is not None])
                pbar.update(len(batch))

    if args.dry_run:
        print("\nDry run complete!")
        return

    print("\nSaving results...")
    # Split results into raw responses and scores
    raw_results = [
        {"record": r["record"], "model": r["model"], "response": r["raw_response"]}
        for r in results
    ]
    scores = [
        {k: v for k, v in r.items() if k not in ["raw_response"]} for r in results
    ]

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

    print("\nBenchmark complete! Results saved in output/")
    print(
        "To generate visualizations, run: poetry run python -m svg_benchmark.visualize"
    )


if __name__ == "__main__":
    asyncio.run(main())
