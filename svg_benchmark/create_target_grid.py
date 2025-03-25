import math
from pathlib import Path
from PIL import Image
from datasets import load_dataset


def create_target_grid(num_images=32, grid_cols=8, gap=8):
    # Load dataset
    dataset = load_dataset("Presto-Design/svg_basic_benchmark_v0")
    eval_data = dataset["test"]

    # Calculate grid dimensions
    grid_rows = math.ceil(num_images / grid_cols)

    # Create a blank canvas
    # Get size from first image
    first_image = eval_data[0]["image"]
    cell_width, cell_height = first_image.width, first_image.height

    # Create the full canvas with gaps
    canvas_width = cell_width * grid_cols + gap * (
        grid_cols - 1
    )  # Add gaps between columns
    canvas_height = cell_height * grid_rows + gap * (
        grid_rows - 1
    )  # Add gaps between rows
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    # Place each image in the grid
    for idx in range(min(num_images, len(eval_data))):
        # Calculate position with gaps
        row = idx // grid_cols
        col = idx % grid_cols
        x = col * (cell_width + gap)  # Add gap to x position
        y = row * (cell_height + gap)  # Add gap to y position

        # Get image and paste it
        img = eval_data[idx]["image"]
        canvas.paste(img, (x, y))

    # Ensure output directory exists
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Save the result
    output_path = output_dir / "target_grid.png"
    canvas.save(output_path)
    print(f"Grid saved to {output_path}")


if __name__ == "__main__":
    create_target_grid()
