from typing import List, Dict, Any, Optional
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
)
from langchain.schema import HumanMessage, AIMessage
from concurrent.futures import ThreadPoolExecutor
import base64
from io import BytesIO
from PIL import Image
import requests
from peft import PeftModel
import io
import re
from qwen_vl_utils import process_vision_info


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


class PrestoModel:
    def __init__(
        self,
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
        base_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        print(f"Loading base model from {base_model_id}...")

        # Load model with appropriate configuration
        # spdf was crashing
        attn_impl = "flash_attention_2" if use_flash_attention else "eager"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
            device_map=None,  # Don't use device map for better control
        )

        # Load adapter if specified
        adapter_path = "Presto-Design/llm_adapter_vectorizer_qwen7b"
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
                        # Convert image_url to the format Qwen expects
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

        # Process messages using Qwen's utility function
        text = self.processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(qwen_messages)

        # print(text)

        # Process inputs
        model_inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to device
        model_inputs = {
            k: v.to(self.device)
            for k, v in model_inputs.items()
            if isinstance(v, torch.Tensor)
        }

        # # Decode and print input_ids for debugging
        # decoded_input_ids = self.processor.tokenizer.batch_decode(
        #     model_inputs["input_ids"], skip_special_tokens=False
        # )
        # print("\nDecoded input_ids:")
        # for i, decoded in enumerate(decoded_input_ids):
        #     print(f"Sequence {i}:")
        #     print(decoded)

        # # Print tensor dimensions for debugging
        # print("\nModel input tensor dimensions:")
        # for k, v in model_inputs.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"{k}: {v.shape} (dtype: {v.dtype})")

        # print()

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
        # Get the length of the input_ids to determine where the response starts
        input_length = model_inputs["input_ids"].shape[1]
        # Decode only the generated part (after input_length)
        response = self.processor.decode(
            outputs[0][input_length:], skip_special_tokens=True
        )

        # Extract the generated text after the last message
        # if "<|im_end|>" in response:
        #     print("response", response)
        #     response = response.split("<|im_end|>")[-1].strip()

        return AIMessage(content=response)

    def __call__(self, *args, **kwargs):
        """Alias for invoke method"""
        return self.invoke(*args, **kwargs)
