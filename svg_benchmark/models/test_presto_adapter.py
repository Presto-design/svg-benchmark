import os
from dotenv import load_dotenv
from PIL import Image, ImageDraw
import base64
import io
import torch
from langchain.schema import HumanMessage
from .presto_model import PrestoModel
from huggingface_hub import login


def create_test_image() -> Image.Image:
    """Create a simple test image"""
    print("Creating test image...")
    # Create a new image with a white background
    image = Image.new("RGB", (280, 280), "white")
    draw = ImageDraw.Draw(image)

    # Draw a simple shape
    draw.rectangle([100, 100, 180, 180], outline="black", width=2)
    draw.line([100, 100, 180, 180], fill="black", width=2)
    draw.line([180, 100, 100, 180], fill="black", width=2)

    return image


def main():
    """Test the Presto model with adapter in isolation"""
    # Load environment variables
    load_dotenv()

    # Login to Hugging Face Hub
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment variables")
    login(token=hf_token)

    # Check if MPS is available
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS device not available")

    print("Initializing Presto model with adapter...")
    model = PrestoModel(
        adapter_path="Presto-Design/qwen2.5-vl-3b-poster-2m-variety-adapter-sft-resume1",
        use_flash_attention=False,  # Use standard attention implementation
        device="mps",  # Use MPS device
    )

    # Create a simple test image
    img = create_test_image()
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    base64_image = base64.b64encode(img_byte_arr).decode()

    # Create test message
    messages = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Write SVG code to recreate this 280 x 280 px design",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ]
        )
    ]

    print("Generating response...")
    response = model.invoke(messages, max_new_tokens=200)

    print("\nModel Response:")
    print("=" * 80)
    print(response.content)
    print("=" * 80)


if __name__ == "__main__":
    main()
