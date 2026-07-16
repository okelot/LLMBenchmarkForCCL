#!/usr/bin/env python3
"""End-to-end LexBench pipeline.

Closed-book track : recall brief from case name -> cosine + LLM-judge (reference)
Open-book  track  : brief from A2AJ decision text -> LLM-judge (source-grounded)
Then combine, build the report and refresh the published site.

    python benchmark_and_rate.py                          # closed-book only
    python benchmark_and_rate.py --holdout random_cases_holdout.csv   # both tracks
    python benchmark_and_rate.py --no-judge               # skip the LLM judge
"""

import argparse
from typing import Optional

import pandas as pd

import lexbench
import report
from benchmark import run_benchmark
from judge import DEFAULT_JUDGE, LLMJudge
from rate_embedding import EmbeddingEvaluator

COMBINED = "results/evaluated_case_model_results_with_section_similarity.csv"


def _score_track(results_csv: str, mode: str, judge_model: Optional[str],
                 max_workers: int = 1) -> str:
    """Add cosine (closed-book only) and judge columns to a results CSV."""
    out = results_csv
    if mode == "closed":
        out = EmbeddingEvaluator().evaluate_results(
            out, output_file=results_csv.replace(".csv", "_scored.csv"))
    if judge_model:
        judge = LLMJudge(judge_model=judge_model)
        out = judge.evaluate_results(
            out, output_file=results_csv.replace(".csv", "_judged.csv"),
            mode="reference" if mode == "closed" else "source",
            max_workers=max_workers)
    return out


def run_pipeline(models_csv="ai_models.csv", cases_csv="random_cases.csv",
                 holdout_csv: Optional[str] = None, judge_model: Optional[str] = DEFAULT_JUDGE,
                 max_cases: Optional[int] = None, sleep_seconds: float = 0.5,
                 use_rubric: bool = True, parallel: bool = False) -> None:
    workers = 6 if parallel else 1
    frames = []

    print("\n### Closed-book track ###")
    closed_raw = run_benchmark(models_csv=models_csv, cases_csv=cases_csv,
                               sleep_seconds=sleep_seconds, max_cases=max_cases,
                               mode="closed", parallel_models=parallel)
    frames.append(_score_track(closed_raw, "closed", judge_model, max_workers=workers))

    if holdout_csv:
        print("\n### Open-book track (temporal holdout) ###")
        open_raw = run_benchmark(models_csv=models_csv, cases_csv=holdout_csv,
                                 sleep_seconds=sleep_seconds, max_cases=max_cases,
                                 mode="open", parallel_models=parallel)
        frames.append(_score_track(open_raw, "open", judge_model, max_workers=workers))

    # combine tracks (union of columns; ensure a mode column)
    dfs = []
    for path in frames:
        d = pd.read_csv(path)
        if "mode" not in d.columns:
            d["mode"] = "closed"
        dfs.append(d)
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(COMBINED, index=False)
    print(f"\nCombined evaluated results -> {COMBINED}")

    if use_rubric:
        print("\n### Rubric stage (author missing rubrics, grade all briefs) ###")
        from benchmark import SECTIONS, load_cases
        from rubric import RubricGrader, author_missing, load_rubrics, save_rubrics
        case_pool = load_cases(cases_csv, max_cases=max_cases)
        if holdout_csv:
            case_pool += load_cases(holdout_csv, max_cases=max_cases)
        rubrics = author_missing(case_pool, load_rubrics(), max_workers=workers)
        save_rubrics(rubrics)
        RubricGrader().evaluate_results(COMBINED, output_file=COMBINED,
                                        max_workers=workers)

    report_file = report.main([COMBINED, "results/report.html"])
    site_file = lexbench.generate(COMBINED)
    print("\nPipeline complete:")
    print(f"  evaluated: {COMBINED}")
    print(f"  report:    {report_file}")
    print(f"  site:      {site_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ai_models.csv")
    ap.add_argument("--cases", default="random_cases.csv")
    ap.add_argument("--holdout", default=None, help="open-book holdout CSV (A2AJ)")
    ap.add_argument("--judge", default=DEFAULT_JUDGE, help="judge model id")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--no-rubric", action="store_true")
    ap.add_argument("--parallel", action="store_true",
                    help="parallelize across models (generation) and rows (grading)")
    ap.add_argument("--max-cases", type=int, default=None)
    a = ap.parse_args()
    run_pipeline(models_csv=a.models, cases_csv=a.cases, holdout_csv=a.holdout,
                 judge_model=None if a.no_judge else a.judge, max_cases=a.max_cases,
                 use_rubric=not a.no_rubric, parallel=a.parallel)
