import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import cairosvg
import base64
import io


def render_svg(svg_text, output_path):
    """Render SVG to PNG using CairoSVG"""
    try:
        cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), write_to=output_path)
        return True
    except Exception as e:
        print(f"Error rendering SVG: {e}")
        return False


def compute_pixel_similarity(img1_path, img2_path):
    """Compute pixel-wise similarity between two images"""
    try:
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        # Resize images to match
        width = max(img1.size[0], img2.size[0])
        height = max(img1.size[1], img2.size[1])
        img1 = img1.resize((width, height))
        img2 = img2.resize((width, height))

        # Convert to numpy arrays
        arr1 = np.array(img1)
        arr2 = np.array(img2)

        # Compute structural similarity
        ssim = structural_similarity(arr1, arr2, channel_axis=2)

        # Compute pixel-wise similarity
        pixel_diff = np.mean(np.abs(arr1 - arr2)) / 255.0
        pixel_sim = 1 - pixel_diff

        return ssim, pixel_sim
    except Exception as e:
        print(f"Error computing image similarity: {e}")
        return 0.0, 0.0


# Tests
def test_render_svg(tmp_path):
    # Create a simple SVG
    svg = '<svg width="100" height="100"><circle cx="50" cy="50" r="40" fill="red"/></svg>'
    output_path = tmp_path / "test.png"

    # Test successful rendering
    assert render_svg(svg, str(output_path)) == True
    assert output_path.exists()

    # Test invalid SVG
    invalid_svg = "<svg>invalid</svg>"
    assert render_svg(invalid_svg, str(tmp_path / "invalid.png")) == False


def test_compute_pixel_similarity(tmp_path):
    # Create two similar images
    img1 = Image.new("RGB", (100, 100), color="red")
    img2 = Image.new("RGB", (100, 100), color="red")

    path1 = tmp_path / "img1.png"
    path2 = tmp_path / "img2.png"
    img1.save(path1)
    img2.save(path2)

    # Test identical images
    ssim, pixel_sim = compute_pixel_similarity(path1, path2)
    assert ssim == 1.0
    assert pixel_sim == 1.0

    # Create different image
    img3 = Image.new("RGB", (100, 100), color="blue")
    path3 = tmp_path / "img3.png"
    img3.save(path3)

    # Test different images
    ssim, pixel_sim = compute_pixel_similarity(path1, path3)
    assert ssim < 1.0
    assert pixel_sim < 1.0
