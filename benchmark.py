"""Benchmark frontier LLMs on Canadian case-law briefing.

For every model in `ai_models.csv` and every case in `random_cases.csv`, the
model is asked to produce a structured case brief (facts, issue, decision,
reasons, ratio) from the case name alone. The AI brief is written alongside the
human-authored brief so `rate_embedding.py` can score their semantic similarity.
"""

import csv
import datetime
import json
import os
import re
from pathlib import Path
from time import sleep
from typing import Dict, List, Optional

from dotenv import load_dotenv

from llm.openrouter import OpenRouterWrapper

load_dotenv()

# Sections that make up a case brief.
SECTIONS = ["facts", "issue", "decision", "reasons", "ratio"]

# Cap on completion tokens per request. Keeps briefs bounded and avoids
# OpenRouter reserving each model's full output window (which can exceed an
# account's credit balance and fail the affordability pre-check). Raise this if
# you benchmark heavy reasoning models that need more room.
MAX_TOKENS = 4096

# Mapping from `random_cases.csv` columns to brief sections.
HUMAN_COLUMNS = {
    "facts": "Facts",
    "issue": "Issue",
    "decision": "Decision",
    "reasons": "Reasons",
    "ratio": "Ratio",
}

SYSTEM_PROMPT = (
    "You are an expert Canadian legal scholar with deep knowledge of Canadian "
    "case law, including decisions of the Supreme Court of Canada and the "
    "provincial appellate courts. You write precise, factually accurate case "
    "briefs grounded strictly in the actual decided case. You never invent "
    "facts, holdings, parties, or citations."
)

# Sentinel returned by a model that does not recognise a case.
UNKNOWN_SENTINEL = "I don't know"


def load_models(csv_path: str = "ai_models.csv") -> List[Dict]:
    """Build per-model OpenRouter configs from the model catalogue CSV."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    models: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            models.append(
                {
                    "model_id": row["model_id"],
                    "display_name": row.get("display_name", row["model_id"]),
                    "developer": row.get("developer", ""),
                    "context_window": row.get("context_window", ""),
                    "input_price": row.get("input_price_per_million", ""),
                    "output_price": row.get("output_price_per_million", ""),
                    "api_key": api_key,
                    "temperature": 0,
                    "max_tokens": MAX_TOKENS,
                }
            )
    return models


def load_cases(csv_path: str = "random_cases.csv", max_cases: Optional[int] = None) -> List[Dict]:
    """Load cases from a CSV.

    Human-brief columns (Facts/Issue/...) are optional — the temporal-holdout set
    from A2AJ has none. A `case_text` column (full decision text) enables the
    open-book track. Pass `max_cases` to load only the first N (handy for tests).
    """
    cases: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            title = (row.get("Title") or "").strip()
            if not title:
                continue
            cases.append(
                {
                    "title": title,
                    "citation": (row.get("Citation") or "").strip(),
                    "human": {s: (row.get(col) or "").strip() for s, col in HUMAN_COLUMNS.items()},
                    "case_text": (row.get("case_text") or "").strip(),
                }
            )
    if max_cases is not None:
        cases = cases[:max_cases]
    return cases


def make_valid_json(input_str: str) -> str:
    """Extract the first complete JSON object from a model response.

    Strips markdown fences and any surrounding prose. Raises ValueError when no
    balanced object is present.
    """
    input_str = re.sub(r"```(?:json)?", "", input_str).strip()

    stack = []
    start_index = None
    for i, char in enumerate(input_str):
        if char == "{":
            if not stack:
                start_index = i
            stack.append(char)
        elif char == "}":
            if stack:
                stack.pop()
                if not stack and start_index is not None:
                    return input_str[start_index : i + 1]

    raise ValueError("No valid JSON object found in the input string.")


# Max characters of decision text fed to a model in open-book mode.
OPEN_CHAR_BUDGET = 200_000

_JSON_TAIL = (
    "Respond strictly as a single JSON object with exactly these string keys: "
    "{{{keys}}}. Output only the JSON, with no markdown, commentary, or "
    "surrounding text."
)


def build_prompt(case_name: str, citation: str = "") -> str:
    """Closed-book: the model must recall the case from parametric knowledge."""
    citation_part = f" (citation: {citation})" if citation else ""
    keys = ", ".join(f'"{s}"' for s in SECTIONS)
    return (
        f'Provide a case brief for the Canadian case "{case_name}"{citation_part}. '
        f"Include only these sections: {', '.join(SECTIONS)} (ratio = ratio decidendi). "
        + _JSON_TAIL.format(keys=keys)
        + f" If you do not know this specific case, respond with exactly: {UNKNOWN_SENTINEL}"
    )


def build_open_prompt(case_name: str, citation: str, case_text: str) -> str:
    """Open-book: brief the case from the supplied decision text (reading task)."""
    keys = ", ".join(f'"{s}"' for s in SECTIONS)
    citation_part = f" ({citation})" if citation else ""
    text = case_text[:OPEN_CHAR_BUDGET]
    if len(case_text) > OPEN_CHAR_BUDGET:
        text += "\n[...decision text truncated...]"
    return (
        f'Below is the full text of the Canadian decision "{case_name}"{citation_part}. '
        "Read it and write a case brief based ONLY on this text.\n\n"
        f"=== DECISION TEXT ===\n{text}\n=== END DECISION TEXT ===\n\n"
        f"Include only these sections: {', '.join(SECTIONS)} (ratio = ratio decidendi). "
        + _JSON_TAIL.format(keys=keys)
    )


def generate_brief(llm: OpenRouterWrapper, prompt: str) -> tuple:
    """Invoke the model and parse a brief. Returns (ai_dict, meta).

    meta carries call telemetry (latency, tokens, finish_reason) plus derived
    quality flags: format_ok (valid JSON returned) and refused (declined /
    "I don't know"). These feed the reporting-rigor metrics.
    """
    call = llm.complete(prompt, context=SYSTEM_PROMPT)
    response = call["text"]
    refused = UNKNOWN_SENTINEL.lower() in response.lower()
    try:
        parsed = json.loads(make_valid_json(response))
        ai = {f"ai_{s}": str(parsed.get(s, "")) for s in SECTIONS}
        format_ok = True
    except (ValueError, json.JSONDecodeError):
        fill = UNKNOWN_SENTINEL if refused else "ERROR"
        ai = {f"ai_{s}": fill for s in SECTIONS}
        format_ok = False
    meta = {
        "latency_s": call["latency_s"],
        "tokens_in": call["prompt_tokens"],
        "tokens_out": call["completion_tokens"],
        "finish_reason": call["finish_reason"],
        "format_ok": int(format_ok),
        "refused": int(refused),
    }
    return ai, meta


def get_case_brief(case_name: str, llm: OpenRouterWrapper, citation: str = "") -> Dict[str, str]:
    """Backwards-compatible closed-book helper: returns just the section dict."""
    ai, _ = generate_brief(llm, build_prompt(case_name, citation))
    return ai


def _price(config: Dict, key: str) -> float:
    try:
        return float(config.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


META_COLUMNS = [
    "mode", "Citation", "latency_s", "tokens_in", "tokens_out", "cost_usd",
    "finish_reason", "format_ok", "refused",
]


def run_benchmark(
    models_csv: str = "ai_models.csv",
    cases_csv: str = "random_cases.csv",
    sleep_seconds: float = 1.0,
    output_path: Optional[str] = None,
    max_cases: Optional[int] = None,
    mode: str = "closed",
) -> str:
    """Run every model over every case and write a results CSV. Returns its path.

    mode="closed": recall the brief from the case name (parametric knowledge).
    mode="open":   brief the case from the supplied decision text (case_text).
    Per-call telemetry (latency, tokens, cost, format/refusal flags) is written
    alongside each row for the reporting-rigor metrics.
    """
    if mode not in ("closed", "open"):
        raise ValueError("mode must be 'closed' or 'open'")
    Path("results").mkdir(exist_ok=True)
    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = f"results/case_model_results_{timestamp}.csv"

    models = load_models(models_csv)
    cases = load_cases(cases_csv, max_cases=max_cases)
    print(f"Loaded {len(models)} models and {len(cases)} cases (mode={mode})")
    if mode == "open" and not any(c["case_text"] for c in cases):
        raise ValueError("open mode requires a 'case_text' column with decision text")

    fieldnames = (
        ["Model_ID", "Case_Name"]
        + [f"ai_{s}" for s in SECTIONS]
        + [f"human_{s}" for s in SECTIONS]
        + META_COLUMNS
        + (["case_text"] if mode == "open" else [])
    )

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for config in models:
            model_id = config["display_name"]
            in_price, out_price = _price(config, "input_price"), _price(config, "output_price")
            print(f"\n=== Testing model: {model_id} ({config['model_id']}) ===")
            try:
                llm = OpenRouterWrapper(config)
            except Exception as e:
                print(f"  Error initializing {model_id}: {e}")
                continue

            for case in cases:
                case_name = case["title"]
                print(f"  Processing case: {case_name}")
                prompt = (
                    build_open_prompt(case_name, case["citation"], case["case_text"])
                    if mode == "open"
                    else build_prompt(case_name, case["citation"])
                )
                try:
                    ai_output, meta = generate_brief(llm, prompt)
                except Exception as e:
                    print(f"    Error on {case_name} with {model_id}: {e}")
                    continue

                cost = meta["tokens_in"] / 1e6 * in_price + meta["tokens_out"] / 1e6 * out_price
                row = {"Model_ID": model_id, "Case_Name": case_name, "mode": mode,
                       "Citation": case["citation"], "cost_usd": round(cost, 6)}
                row.update(ai_output)
                row.update({f"human_{s}": case["human"][s] for s in SECTIONS})
                row.update({k: meta[k] for k in
                            ["latency_s", "tokens_in", "tokens_out", "finish_reason",
                             "format_ok", "refused"]})
                if mode == "open":
                    row["case_text"] = case["case_text"]
                writer.writerow(row)
                csvfile.flush()
                if sleep_seconds:
                    sleep(sleep_seconds)

    print(f"\nResults saved to {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Benchmark LLMs on Canadian case-law briefing")
    ap.add_argument("--mode", choices=["closed", "open"], default="closed")
    ap.add_argument("--cases", default="random_cases.csv")
    ap.add_argument("--models", default="ai_models.csv")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=1.0)
    a = ap.parse_args()
    run_benchmark(models_csv=a.models, cases_csv=a.cases, sleep_seconds=a.sleep,
                  max_cases=a.max_cases, mode=a.mode)
