import copy
import os
import csv
from pathlib import Path
import multiprocessing
from multiprocessing import Pool
from typing import Optional, Dict, Any, Tuple
from tqdm import tqdm
import sys
import json
from datetime import datetime
from datasets import load_dataset
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import argparse

from .models.presto_model import PrestoModel
from .utils import (
    extract_data_urls,
    replace_urls,
    extract_svg,
    render_svg,
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


def process_example(
    args_tuple: Tuple[
        Dict[str, Any], str, Dict[str, Any], int, argparse.Namespace, Optional[Any]
    ],
) -> Dict[str, Any]:
    """Process a single example using multiprocessing"""
    example, model_id, model_config, idx, args, model_instance = args_tuple
    try:
        # Create or reuse model instance
        model = (
            model_instance
            if model_instance
            else model_config["class"](**model_config["params"])
        )

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
                "raw_response": response_text,
            }

        # Save target SVG and render to PNG
        target_svg = example["completion"]
        target_svg_path = f"output/{model_id}/{idx}_target.svg"
        with open(target_svg_path, "w") as f:
            f.write(target_svg)
        render_svg(target_svg, target_png_path)

        return {
            "record": idx,
            "model": model_id,
            "raw_response": response_text,
        }

    except Exception as e:
        print(f"\nError processing {model_id} example {idx}: {str(e)}")
        return {
            "record": idx,
            "model": model_id,
            "raw_response": str(e),
        }


def process_model(
    model_id: str,
    config: dict,
    eval_data: list,
    args: argparse.Namespace,
    progress_bar: Optional[tqdm] = None,
) -> list:
    """Process all examples for a single model in parallel using multiprocessing"""
    results = []

    if args.dry_run:
        print(f"\nProcessing model: {model_id}")
        print(f"Model config: {config}")
        print(f"First example type: {type(eval_data[0])}")
        print(
            f"First example keys: {list(eval_data[0].keys()) if isinstance(eval_data[0], dict) else 'Not a dict'}"
        )

    # Create model-specific progress bar
    model_pbar = tqdm(
        total=len(eval_data),
        desc=f"{model_id:8}",
        position=1,
        leave=False,
        disable=args.dry_run,
    )

    # Process examples - serially for Presto, in parallel for others
    if model_id == "presto":
        # Create Presto model instance once
        model_instance = config["class"](**config["params"])

        # Process Presto examples serially
        for idx, example in enumerate(eval_data):
            result = process_example(
                (example, model_id, config, idx, args, model_instance)
            )
            if result is not None:
                results.append(result)
            model_pbar.update(1)
            if progress_bar:
                progress_bar.update(1)
    else:
        # Process other models in parallel batches
        with Pool(processes=args.parallel) as pool:
            for i in range(0, len(eval_data), args.parallel):
                batch = eval_data[i : i + args.parallel]
                if args.dry_run:
                    print(f"\nBatch {i//args.parallel + 1}:")
                    print(f"Batch size: {len(batch)}")
                    print(f"First batch item type: {type(batch[0])}")

                # Prepare arguments for each process
                process_args = [
                    (example, model_id, config, idx + i, args, None)
                    for idx, example in enumerate(batch)
                ]

                # Process batch in parallel
                batch_results = pool.map(process_example, process_args)
                valid_results = [r for r in batch_results if r is not None]
                results.extend(valid_results)

                # Update both progress bars
                model_pbar.update(len(batch))
                if progress_bar:
                    progress_bar.update(len(batch))

    model_pbar.close()
    return results


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate SVGs using specified models")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print model inputs without running"
    )
    parser.add_argument(
        "--use-presto", action="store_true", help="Enable Presto model for benchmarking"
    )
    parser.add_argument(
        "--parallel", type=int, default=8, help="Number of parallel processes per model"
    )
    parser.add_argument(
        "--use-claude", action="store_true", help="Enable Claude model for benchmarking"
    )
    parser.add_argument(
        "--use-gpt4", action="store_true", help="Enable GPT-4 model for benchmarking"
    )
    args = parser.parse_args()

    # Define model configurations based on flags
    model_configs = {}

    if args.use_claude:
        model_configs["claude"] = {
            "class": ChatAnthropic,
            "params": {"model": "claude-3-7-sonnet-20250219", "temperature": 0},
        }

    if args.use_gpt4:
        model_configs["gpt4"] = {
            "class": ChatOpenAI,
            "params": {"model": "gpt-4o", "temperature": 0, "max_tokens": 4096},
        }

    # Add Presto model if enabled
    if args.use_presto:
        print("Loading Presto model...")
        model_configs["presto"] = {
            "class": PrestoModel,
            "params": {
                "adapter_path": "Presto-Design/qwen2.5-vl-3b-poster-2m-variety-adapter-sft-resume1",
                "use_flash_attention": False,
            },
        }

    if not model_configs:
        print(
            "Error: No models selected. Please enable at least one model using --use-claude, --use-gpt4, or --use-presto"
        )
        sys.exit(1)

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

    # Initialize results storage
    results = []

    if not args.dry_run:
        check_api_keys()
        # Create output directories
        Path("output").mkdir(exist_ok=True)
        for model in model_configs:
            Path(f"output/{model}").mkdir(exist_ok=True)

    # Calculate total work to be done
    total_examples = len(eval_data) * len(model_configs)

    # Create main progress bar
    main_pbar = tqdm(
        total=total_examples,
        desc="Total    ",
        position=0,
        leave=True,
        disable=args.dry_run,
    )

    # Process all models sequentially (each model uses parallel processing internally)
    for model_id, config in model_configs.items():
        model_results = process_model(model_id, config, eval_data, args, main_pbar)
        results.extend(model_results)

    main_pbar.close()

    if args.dry_run:
        print("\nDry run complete!")
        return

    print("\nSaving results...")
    # Add timestamp to results
    run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save raw results (append mode)
    raw_file = "output/raw.csv"
    file_exists = os.path.isfile(raw_file)
    with open(raw_file, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["record", "model", "response", "run_time"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(
            [
                {
                    "record": r["record"],
                    "model": r["model"],
                    "response": r["raw_response"],
                    "run_time": run_time,
                }
                for r in results
            ]
        )

    print("\nGeneration complete! Results saved to output/raw.csv")
    print("To calculate scores, run: poetry run python -m svg_benchmark.score")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required for Windows compatibility
    main()
