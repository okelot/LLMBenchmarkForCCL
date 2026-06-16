"""Helper to browse OpenRouter's catalogue and refresh `ai_models.csv`.

OpenRouter's /models endpoint is public (no key required). Use this to discover
current model ids and pricing when curating the benchmark's frontier model list.

    python -m llm.openrouter_models                 # list everything
    python -m llm.openrouter_models anthropic gpt-5 # filter by substring(s)
    python -m llm.openrouter_models --csv anthropic # write ai_models_openrouter.csv
"""

import csv
import sys
from typing import Dict, List

import requests

MODELS_URL = "https://openrouter.ai/api/v1/models"


def fetch_models() -> List[Dict]:
    response = requests.get(MODELS_URL, timeout=60)
    response.raise_for_status()
    return response.json()["data"]


def _per_million(price_per_token: str) -> float:
    try:
        return round(float(price_per_token) * 1_000_000, 4)
    except (TypeError, ValueError):
        return 0.0


def filter_models(models: List[Dict], terms: List[str]) -> List[Dict]:
    if not terms:
        return models
    terms = [t.lower() for t in terms]
    return [m for m in models if any(t in m["id"].lower() for t in terms)]


def write_csv(models: List[Dict], path: str = "ai_models_openrouter.csv") -> str:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_id",
                "display_name",
                "developer",
                "context_window",
                "input_price_per_million",
                "output_price_per_million",
            ]
        )
        for m in models:
            pricing = m.get("pricing", {})
            writer.writerow(
                [
                    m["id"],
                    m.get("name", m["id"]),
                    m["id"].split("/")[0],
                    m.get("context_length", ""),
                    _per_million(pricing.get("prompt")),
                    _per_million(pricing.get("completion")),
                ]
            )
    print(f"Wrote {len(models)} models to {path}")
    return path


def main(argv: List[str]) -> None:
    write = "--csv" in argv
    terms = [a for a in argv if a != "--csv"]

    models = filter_models(fetch_models(), terms)
    models.sort(key=lambda m: m["id"])

    if write:
        write_csv(models)
        return

    print(f"{len(models)} model(s):")
    for m in models:
        pricing = m.get("pricing", {})
        print(
            f"  {m['id']:48s} ctx={str(m.get('context_length','?')):>9s} "
            f"in=${_per_million(pricing.get('prompt')):<8} "
            f"out=${_per_million(pricing.get('completion'))}/M"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
