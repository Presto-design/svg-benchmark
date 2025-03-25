# SVG Generation Benchmark Suite

This benchmark suite evaluates large language models on their ability to generate SVG code from visual designs.

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
- `ANTHROPIC_API_KEY`: Your Anthropic API key for Claude
- `OPENAI_API_KEY`: Your OpenAI API key for GPT-4

## Running the Benchmark

Run the benchmark using Poetry:
```bash
./run_benchmark.sh
```

The script will:
1. Load the first 32 examples from the Presto-Design SVG benchmark dataset
2. Test Claude 3 Opus and GPT-4 on SVG generation
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
├── claude/
│   ├── 0.svg
│   ├── 0.png
│   └── ...
└── gpt4/
    ├── 0.svg
    ├── 0.png
    └── ...
```

## Running Tests

To run the test suite:
```bash
./run_tests.sh
``` 