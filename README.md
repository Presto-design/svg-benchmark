# SVG Generation Benchmark Suite

This benchmark suite, created by [Presto Design](https://github.com/Presto-design), evaluates large language models on their ability to generate SVG code from visual designs. It's part of our mission to advance the field of machine-assisted graphic design.

## Why This Benchmark Matters

### The Importance of LLMs in Graphic Design
Large Language Models (LLMs) are becoming vital tools for professional graphic design, offering advantages that image generation models like Stable Diffusion cannot match:
- Ability to work with branded assets and specific stock photos
- Support for brand fonts and typography
- Creation of scalable, resolution-independent designs
- Direct manipulation of design elements through code

### The Current State of LLM Graphic Design
LLMs have historically struggled with graphic design tasks. This benchmark focuses on testing their "fifth grader" abilities - given an image, can they replicate it perfectly? This fundamental capability is a necessary stepping stone toward more sophisticated design tasks.

### About the Benchmark Dataset
The [benchmark dataset](https://huggingface.co/datasets/Presto-Design/svg_basic_benchmark_v0) contains 2,000 images and their associated SVG code, testing comprehension of key SVG features:
- Colors and gradients
- Basic and complex shapes
- Image handling and masks
- Text and font manipulation
- Icons and strokes
- Advanced SVG features

For more insights, read our article: ["Why LLMs are Bad at Creating SVGs and Graphic Design - And How to Make Them Good"](https://prestodesign.ai/)

## Contributing

We welcome contributions to improve the benchmark! Here's how you can help:

1. **Run the Benchmark**: Test new models and share your results
2. **Improve the Code**: Submit PRs to enhance the benchmark suite
3. **Add Test Cases**: Help expand the test dataset
4. **Share Findings**: Publish your insights and improvements

For contribution guidelines, check our [GitHub repository](https://github.com/Presto-design/svg-benchmark).

## Setup

1. Make sure you have Python 3.9+ installed
2. Install dependencies using Poetry:
```bash
poetry install
```

3. Set up your environment variables:
```bash
cp .env.template .env
```
Then edit `.env` and add your API keys:
- `ANTHROPIC_API_KEY`: Your Anthropic API key for Claude (only needed if using Claude)
- `OPENAI_API_KEY`: Your OpenAI API key for GPT-4 (only needed if using GPT-4)

## Running the Benchmark

The benchmark supports multiple models that can be enabled via command-line flags:

```bash
# Run with Claude only
./run_benchmark.sh --use-claude

# Run with GPT-4 only
./run_benchmark.sh --use-gpt4

# Run with Presto model
./run_benchmark.sh --presto-model /path/to/model

# Run with multiple models
./run_benchmark.sh --use-claude --use-gpt4 --presto-model /path/to/model

# Additional options:
--parallel N     # Number of parallel processes per model (default: 8)
--dry-run       # Print inputs without running models
```

The script will:
1. Load the first 32 examples from the Presto-Design SVG benchmark dataset (6 examples for dry runs)
2. Test the selected models on SVG generation using parallel processing
3. Generate output files in the `output/` directory:
   - `raw.csv`: Raw model responses
   - `scores.csv`: Detailed scores for each example
   - `mean_scores.csv`: Average scores per model
   - `comparison.svg`: Visual comparison of model performance
   - Model-specific directories containing generated SVGs and PNGs

## Metrics

The benchmark evaluates models on three metrics:
1. BLEU score: Comparing generated SVG code with reference code
2. Structural similarity: Visual comparison of rendered images
3. Pixel-wise similarity: Direct pixel comparison of rendered images

## Output Structure

```
output/
├── raw.csv
├── scores.csv
├── mean_scores.csv
├── comparison.svg
├── claude/          # Only present if Claude is used
│   ├── 0.svg
│   ├── 0.png
│   └── ...
├── gpt4/           # Only present if GPT-4 is used
│   ├── 0.svg
│   ├── 0.png
│   └── ...
└── presto/         # Only present if Presto is used
    ├── 0.svg
    ├── 0.png
    └── ...
```

## Performance Notes

The benchmark uses Python's multiprocessing to parallelize example processing within each model. Models are processed sequentially to avoid API rate limits and resource contention. The number of parallel processes can be adjusted using the `--parallel` flag.

## Running Tests

To run the test suite:
```bash
./run_tests.sh
``` 