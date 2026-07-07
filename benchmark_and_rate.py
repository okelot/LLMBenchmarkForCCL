#!/usr/bin/env python3
"""End-to-end pipeline: benchmark frontier models, score, and build the report.

    python benchmark_and_rate.py
"""

import lexbench
import report
from benchmark import run_benchmark
from rate_embedding import EmbeddingEvaluator


def run_benchmark_and_rate(
    models_csv: str = "ai_models.csv",
    cases_csv: str = "random_cases.csv",
    sleep_seconds: float = 0.5,
    max_cases: int = None,
) -> None:
    benchmark_file = run_benchmark(
        models_csv=models_csv,
        cases_csv=cases_csv,
        sleep_seconds=sleep_seconds,
        max_cases=max_cases,
    )

    print("\nStarting semantic-similarity evaluation...")
    evaluator = EmbeddingEvaluator()
    rated_file = evaluator.evaluate_results(benchmark_file)

    print("\nBuilding HTML report...")
    report_file = report.main([rated_file, "results/report.html"])

    print("Refreshing LexBench landing page...")
    site_file = lexbench.generate(rated_file)

    print("\nComplete pipeline results:")
    print(f"1. Benchmark results: {benchmark_file}")
    print(f"2. Evaluation results: {rated_file}")
    print(f"3. HTML report:       {report_file}")
    print(f"4. LexBench site:     {site_file}")


if __name__ == "__main__":
    run_benchmark_and_rate()
