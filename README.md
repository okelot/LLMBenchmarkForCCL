# LLM Benchmark for Canadian Case Law (CCL)

A comprehensive benchmarking tool for evaluating frontier Large Language Models (LLMs) on their ability to analyze and understand Canadian case law. This project tests models against human-annotated case briefs to measure their accuracy and performance across multiple dimensions.

All models are accessed through a single [OpenRouter](https://openrouter.ai/) API key, so adding or swapping a frontier model only means editing one row in `ai_models.csv`.

![Sample Benchmark Results](charts/model_similarity_chart_2025-07-04_10-28-08.png)

The above chart shows a sample benchmark comparing different models' performance across various aspects of case analysis.

## Features

- Automated testing of multiple LLM models
- Comprehensive evaluation across key legal analysis dimensions
- CSV output for detailed performance analysis
- Support for various LLM providers through a unified interface

## Supported Models

Any model available on OpenRouter. The default `ai_models.csv` ships with a
curated frontier set spanning the major labs:

- Anthropic — Claude Opus 4.8, Claude Sonnet 4.6
- OpenAI — GPT-5.5, GPT-5.4
- Google — Gemini 2.5 Pro
- xAI — Grok 4.20
- DeepSeek — DeepSeek Chat v3.1
- Meta — Llama 4 Maverick

Run `python -m llm.openrouter_models <filter>` to browse current ids and pricing,
or `python -m llm.openrouter_models --csv <filter>` to regenerate the catalogue.

## Evaluation Categories

Models are evaluated on their ability to analyze Canadian case law across these dimensions:

1. **Factual Accuracy**: Precision in extracting and summarizing case facts
2. **Ratio Accuracy**: Accuracy in identifying the ratio decidendi
3. **Issue Accuracy**: Precision in identifying legal issues
4. **Hallucination**: Measurement of false or invented information

## Setup

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd LLMBenchmarkForCCL
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your API key:
   - `OPENROUTER_API_KEY`: Your OpenRouter API key (get one at https://openrouter.ai/keys). To use a different provider, implement the `LLM_Wrapper` interface (see `llm/openrouter.py`).

3. Prepare your test cases:
   - Place your test cases in `random_cases.csv` or use `app_get_random_cases_from_fandom.py` to scrape random cases from Fandom.

## Usage

1. Generate test cases (optional):
   ```bash
   python app_get_random_cases_from_fandom.py
   ```
   This will scrape case briefs from Fandom with human generated case briefs.

2. Run the benchmark:
   ```bash
   python benchmark.py
   ```
   This queries every model in `ai_models.csv` for a brief of every case in
   `random_cases.csv` and writes raw AI-vs-human comparisons to `results/`.
   (Run benchmarking and scoring in one step with `python benchmark_and_rate.py`.)

3. Evaluate results:
   ```bash
   python rate_embedding.py results/case_model_results_*.csv
   ```
   This computes semantic similarity scores between AI and human annotations. (You will need strong CPU to run this otherwise it will take a long time)

4. Generate visualizations:
   ```bash
   python chart.py       # standalone PNG bar chart in charts/
   python report.py results/evaluated_case_model_results_with_section_similarity.csv
   ```
   `report.py` builds a self-contained HTML report (embedded chart + side-by-side
   AI/human briefs) at `results/report.html`.

5. Refresh the LexBench site:
   ```bash
   python lexbench.py
   ```
   This regenerates the published landing page at `docs/index.html` from the
   latest evaluated results (see **Publishing** below).

6. View results:
   - Raw results: `results/case_model_results_YYYY-MM-DD_HH-MM-SS.csv`
   - Evaluated results: `results/evaluated_case_model_results_with_section_similarity.csv`
   - HTML report: `results/report.html`
   - Public site: `docs/index.html`

> `python benchmark_and_rate.py` runs steps 2–5 end to end: benchmark → score →
> report → refresh the LexBench site.

## Publishing (LexBench)

**LexBench** is the public landing page for the benchmark — a self-contained
static site (`docs/index.html`) with the current leaderboard, a section-by-section
chart, and methodology. It is regenerated from the latest evaluated results on
every run, so publishing stays as simple as committing `docs/` and pushing.

To publish it for free with **GitHub Pages**:

1. Commit and push the `docs/` folder.
2. On GitHub: **Settings → Pages → Build and deployment → Source: Deploy from a
   branch**, then choose **Branch: `main`, Folder: `/docs`** and save.
3. The site goes live at `https://okelot.github.io/LLMBenchmarkForCCL/`.

Each subsequent run overwrites `docs/index.html`; commit and push to update the
live site (`docs/.nojekyll` is included so Pages serves the files as-is).

## Output Format

The benchmark generates a CSV file with the following columns:
- Model_ID: Identifier for the LLM model
- Case_Name: Name of the case being analyzed
- AI outputs (ai_facts, ai_issue, ai_decision, ai_reasons, ai_ratio)
- Human annotations (human_facts, human_issue, human_decision, human_reasons, human_ratio)

## Project Structure

```
├── benchmark.py                        # Main benchmarking script
├── benchmark_and_rate.py               # Benchmark -> score -> report -> site
├── rate_embedding.py                   # Semantic similarity evaluation
├── report.py                           # Per-run HTML report (briefs + chart)
├── lexbench.py                         # LexBench landing-page generator
├── chart.py                            # Standalone PNG chart
├── app_get_random_cases_from_fandom.py # Test case scraper
├── llm/
│   ├── llm_wrapper.py                  # Base wrapper interface
│   ├── openrouter.py                   # OpenRouter implementation
│   ├── openrouter_models.py            # Catalogue browser / CSV refresher
│   └── test_openrouter.py              # Smoke test
├── ai_models.csv                       # Frontier model configurations
├── random_cases.csv                    # Test cases (human-annotated briefs)
├── docs/                               # Published LexBench site (GitHub Pages)
└── charts/                             # Generated benchmark visualizations
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## Future Improvements

1. Automated accuracy rating using GPT-4 as a judge
2. Support for additional LLM providers
3. Enhanced evaluation metrics
4. Interactive results visualization
