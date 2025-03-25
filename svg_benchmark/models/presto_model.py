from typing import List, Dict, Any, Optional
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
)
from langchain.schema import HumanMessage
from concurrent.futures import ThreadPoolExecutor
import base64
from io import BytesIO
from PIL import Image
import requests
from peft import PeftModel
import io
import re


def download_image(url: str) -> Image.Image:
    """Download an image from a URL or load from base64"""
    if url.startswith("data:image/"):
        # Extract base64 data
        base64_data = url.split(",")[1]
        image_data = base64.b64decode(base64_data)
        return Image.open(BytesIO(image_data))
    else:
        response = requests.get(url)
        return Image.open(BytesIO(response.content))


def process_vision_info(messages: List[Dict[str, Any]]) -> List[Image.Image]:
    """Extract and process images from messages"""
    images = []
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if item.get("type") == "image":
                    # Handle base64 image data
                    if item["image"].startswith("data:image"):
                        # Extract base64 data after the comma
                        base64_data = item["image"].split(",")[1]
                        image_data = base64.b64decode(base64_data)
                        image = Image.open(io.BytesIO(image_data))
                        # Convert to RGB mode if needed
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                        # Resize to model's expected size (448x448 is standard for Qwen2.5-VL)
                        image = image.resize((448, 448))
                        images.append(image)
    return images


class PrestoModel:
    def __init__(
        self,
        adapter_path: str = None,
        device: str = None,
        use_flash_attention: bool = False,
        min_pixels: int = None,
        max_pixels: int = None,
    ):
        """Initialize the Presto model with base model and adapter"""
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Set dtype based on device
        if device == "cuda":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Load base model
        base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        print(f"Loading base model from {base_model_id}...")

        # Load model with appropriate configuration
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",  # Use SDPA for better compatibility
            device_map=None,  # Don't use device map for better control
        )

        # Load adapter if specified
        if adapter_path:
            print(f"Loading adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                is_trainable=False,  # We're using it for inference
            )

        # Move model to device after all components are loaded
        print(f"Moving model to {device}...")
        self.model = self.model.to(device)

        # Initialize processor
        print("Initializing processor...")
        processor_kwargs = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels

        self.processor = AutoProcessor.from_pretrained(
            base_model_id, **processor_kwargs
        )

    def _convert_langchain_to_qwen_messages(
        self, messages: List[HumanMessage]
    ) -> List[Dict[str, Any]]:
        """Convert LangChain messages to Qwen format"""
        qwen_messages = []
        for msg in messages:
            if isinstance(msg.content, list):
                content = []
                for item in msg.content:
                    if item.get("type") == "text":
                        content.append({"type": "text", "text": item["text"]})
                    elif item.get("type") == "image_url":
                        content.append(
                            {"type": "image", "image": item["image_url"]["url"]}
                        )
                qwen_messages.append({"role": "user", "content": content})
            else:
                qwen_messages.append(
                    {"role": "user", "content": [{"type": "text", "text": msg.content}]}
                )
        return qwen_messages

    def invoke(
        self, messages: List[HumanMessage], max_new_tokens: int = 200
    ) -> HumanMessage:
        """Generate a response for the given messages"""
        qwen_messages = self._convert_langchain_to_qwen_messages(messages)

        # Extract images from messages
        images = process_vision_info(qwen_messages)
        if not images:
            raise ValueError("No images found in messages")

        # Create text prompt using the processor's chat template
        text = self.processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs with text as a list and ensure correct image dimensions
        model_inputs = self.processor(
            text=[text], images=images, return_tensors="pt", padding=True
        )

        # Move inputs to device
        model_inputs = {
            k: v.to(self.device)
            for k, v in model_inputs.items()
            if isinstance(v, torch.Tensor)
        }

        # Print tensor dimensions for debugging
        print("\nModel input tensor dimensions:")
        for k, v in model_inputs.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape} (dtype: {v.dtype})")
        print()

        # Generate response
        with torch.inference_mode():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

            # Move outputs back to CPU for decoding
            outputs = outputs.to("cpu")

        # Decode the response
        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Extract the generated text after the last message
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[-1].strip()

        return HumanMessage(content=response)

    def __call__(self, *args, **kwargs):
        """Alias for invoke method"""
        return self.invoke(*args, **kwargs)
