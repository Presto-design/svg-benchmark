import base64
from io import BytesIO
from PIL import Image


def image_to_base64(image):
    """Convert a PIL Image to base64 string"""
    if isinstance(image, str):
        return image

    # Convert PIL Image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def create_image_message(image_data, width, height, url_mapping):
    """Create a message with embedded images and their mappings in multimodal format"""
    message = []

    # Add the main prompt
    message.append(
        {
            "type": "text",
            "text": f"Write SVG code to recreate this {width} x {height} px design.",
        }
    )

    # Add the main image - ensure proper base64 formatting
    image_b64 = image_to_base64(image_data)
    if not image_b64.startswith("data:"):
        image_b64 = f"data:image/png;base64,{image_b64}"
    message.append(
        {
            "type": "image_url",
            "image_url": {"url": image_b64},
        }
    )

    # Add any referenced images with their short URLs
    for short_url, data_url in url_mapping.items():
        message.append({"type": "text", "text": f"Asset {short_url}"})
        # Ensure proper base64 formatting for referenced images
        data_b64 = image_to_base64(data_url)
        if not data_b64.startswith("data:"):
            data_b64 = f"data:image/png;base64,{data_b64}"
        message.append({"type": "image_url", "image_url": {"url": data_b64}})

    return message


# Tests
def test_image_to_base64():
    # Test with string input
    assert image_to_base64("already_base64") == "already_base64"

    # Test with PIL Image
    img = Image.new("RGB", (100, 100), color="red")
    b64 = image_to_base64(img)
    assert isinstance(b64, str)
    assert base64.b64decode(b64)  # Should be valid base64


def test_create_image_message():
    # Test basic message creation with raw base64
    message = create_image_message("base64data", 100, 200, {})
    assert len(message) == 2
    assert message[0]["type"] == "text"
    assert "100 x 200" in message[0]["text"]
    assert message[1]["type"] == "image_url"
    assert "data:image/png;base64,base64data" in message[1]["image_url"]["url"]

    # Test with PIL Image
    img = Image.new("RGB", (100, 100), color="red")
    message = create_image_message(img, 100, 200, {})
    assert len(message) == 2
    assert message[1]["type"] == "image_url"
    assert "data:image/png;base64," in message[1]["image_url"]["url"]

    # Test with already formatted base64
    message = create_image_message("data:image/png;base64,base64data", 100, 200, {})
    assert len(message) == 2
    assert "data:image/png;base64,base64data" in message[1]["image_url"]["url"]

    # Test with URL mapping
    url_mapping = {
        "cdn://1.jpg": "base64ABC",
        "cdn://2.jpg": "data:image/jpeg;base64,DEF",
    }
    message = create_image_message("base64data", 100, 200, url_mapping)
    assert len(message) == 6  # 2 base + 2 pairs of (text, image) for mappings

    # Check asset entries
    asset_texts = [
        m["text"] for m in message if m["type"] == "text" and "Asset" in m["text"]
    ]
    assert len(asset_texts) == 2
    assert "Asset cdn://1.jpg" in asset_texts
    assert "Asset cdn://2.jpg" in asset_texts

    # Check image URLs
    image_urls = [m["image_url"]["url"] for m in message if m["type"] == "image_url"]
    assert len(image_urls) == 3  # Main image + 2 asset images
    assert "data:image/png;base64,base64ABC" in image_urls
    assert "data:image/jpeg;base64,DEF" in image_urls
