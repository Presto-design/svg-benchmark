import re


def extract_data_urls(svg_text):
    """Extract all data: URLs from an SVG and assign short URLs"""
    data_urls = re.findall(r'data:image[^"\']*', svg_text)
    url_mapping = {}
    for i, url in enumerate(data_urls):
        short_url = f"cdn://{i+1}.jpg"
        url_mapping[short_url] = url
    return url_mapping


def replace_urls(svg_text, url_mapping):
    """Replace short URLs with their original data: URLs"""
    result = svg_text
    for short_url, data_url in url_mapping.items():
        result = result.replace(short_url, data_url)
    return result


def extract_svg(text):
    """Extract SVG content from a text response"""
    svg_match = re.search(r"<svg[\s\S]*?</svg>", text)
    return svg_match.group(0) if svg_match else None


# Tests
def test_extract_data_urls():
    svg = """<svg><image href="data:image/png;base64,ABC"/><image href="data:image/jpeg;base64,DEF"/></svg>"""
    mapping = extract_data_urls(svg)
    assert len(mapping) == 2
    assert "cdn://1.jpg" in mapping
    assert "cdn://2.jpg" in mapping
    assert mapping["cdn://1.jpg"] == "data:image/png;base64,ABC"
    assert mapping["cdn://2.jpg"] == "data:image/jpeg;base64,DEF"


def test_replace_urls():
    mapping = {"cdn://1.jpg": "data:image/png;base64,ABC"}
    svg = '<svg><image href="cdn://1.jpg"/></svg>'
    result = replace_urls(svg, mapping)
    assert result == '<svg><image href="data:image/png;base64,ABC"/></svg>'


def test_extract_svg():
    text = "Some text before <svg>content</svg> some text after"
    result = extract_svg(text)
    assert result == "<svg>content</svg>"

    text_no_svg = "No SVG content here"
    result = extract_svg(text_no_svg)
    assert result is None
