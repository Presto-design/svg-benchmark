from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset
from typing import List, Optional


def create_comparison_grid(
    indices: List[int] = [0, 5, 8, 9, 23], gap: int = 8, include_presto: bool = False
):
    # Load dataset for target images
    dataset = load_dataset("Presto-Design/svg_basic_benchmark_v0")
    eval_data = dataset["test"]

    # Define models to compare
    models = ["target", "claude", "gpt4"]
    if include_presto:
        models.append("presto")

    # Get image dimensions from first target image
    first_image = eval_data[0]["image"]
    cell_width, cell_height = first_image.width, first_image.height

    # Create the full canvas
    # Width = number of models (columns) * cell_width + gaps between columns
    # Height = number of examples (rows) * (cell_height + space for labels) + gaps between rows
    label_height = 30
    canvas_width = cell_width * len(models) + gap * (
        len(models) - 1
    )  # Add horizontal gaps
    canvas_height = (cell_height + label_height) * len(indices) + gap * (
        len(indices) - 1
    )  # Add vertical gaps
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()

    # Place images and labels
    for row_idx, data_idx in enumerate(indices):
        if data_idx >= len(eval_data):
            print(f"Warning: Index {data_idx} is out of range, skipping...")
            continue

        example = eval_data[data_idx]
        y = row_idx * (cell_height + label_height + gap)  # Add gap to y position

        for col, model in enumerate(models):
            x = col * (cell_width + gap)  # Add gap to x position

            # Add column header on first row
            if row_idx == 0:
                header_text = f"{model}"
                draw.text(
                    (x + cell_width // 2, y),
                    header_text,
                    fill="black",
                    font=font,
                    anchor="mt",
                )

            # Add index label for target column
            if model == "target":
                index_text = f"#{data_idx}"
                draw.text(
                    (x + 5, y + label_height + 5),  # Small padding from top-left corner
                    index_text,
                    fill="black",
                    font=font,
                )

            # Load and paste appropriate image
            if model == "target":
                img = example["image"]
            else:
                model_png = Path("output") / model / f"{data_idx}.png"
                if model_png.exists():
                    img = Image.open(model_png)
                else:
                    # Create a blank image with error message if file doesn't exist
                    img = Image.new("RGB", (cell_width, cell_height), "white")
                    draw_error = ImageDraw.Draw(img)
                    draw_error.text(
                        (cell_width // 2, cell_height // 2),
                        "Image not found",
                        fill="red",
                        font=font,
                        anchor="mm",
                    )

            canvas.paste(img, (x, y + label_height))

    # Save the result
    output_dir = Path("output")
    output_path = output_dir / "model_comparison.png"
    canvas.save(output_path)
    print(f"Comparison grid saved to {output_path}")


if __name__ == "__main__":
    # Example usage with all options
    create_comparison_grid(include_presto=True)
