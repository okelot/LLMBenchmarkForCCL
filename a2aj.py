"""Client for the A2AJ Canadian Legal Data API (https://api.a2aj.ca).

A2AJ (Access to Algorithmic Justice, Osgoode / Lincoln Alexander) offers free,
open access to 191k+ full-text Canadian court and tribunal decisions, updated
weekly. We use it for two things the fandom dataset can't give us:

  * open-book judgment text (test reading/extraction, not memorization)
  * a temporal holdout of recent decisions that postdate model training
    cutoffs (contamination-resistant knowledge test)

License note: the A2AJ code/methods are MIT, but each decision keeps its
`upstream_license` (many are non-commercial, e.g. SCC/Lexum terms). This is fine
for research/evaluation, but DO NOT republish full decision text — the public
LexBench site shows only case name, citation, court, date, and scores, and links
to the official source URL.

    python a2aj.py holdout --start 2025-06-01 --limit 20 --out random_cases_holdout.csv
"""

import argparse
import csv
import sys
from time import sleep
from typing import Dict, List, Optional

import requests

BASE_URL = "https://api.a2aj.ca"

# Broad seed queries used to surface a spread of recent decisions for the holdout
# (the API ranks by relevance to a query; there is no "list all" endpoint).
SEED_QUERIES = [
    "judicial review", "negligence", "charter", "sentencing", "contract",
    "damages", "appeal", "employment", "tax", "regulatory",
]


def _get(endpoint: str, params: Dict) -> Dict:
    r = requests.get(f"{BASE_URL}/{endpoint.lstrip('/')}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def coverage() -> List[Dict]:
    return _get("coverage", {}).get("results", [])


def search(query: str, dataset: Optional[str] = None, start_date: Optional[str] = None,
           end_date: Optional[str] = None, size: int = 25) -> List[Dict]:
    params: Dict = {"query": query, "size": size}
    if dataset:
        params["dataset"] = dataset
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    return _get("search", params).get("results", [])


def fetch(citation: str) -> Optional[Dict]:
    """Fetch a full decision record (includes unofficial_text_en) by citation."""
    results = _get("fetch", {"citation": citation}).get("results", [])
    return results[0] if results else None


def fetch_text(citation: str) -> str:
    rec = fetch(citation)
    return (rec or {}).get("unofficial_text_en", "") or ""


def build_holdout(out_csv: str, datasets=("FC", "FCA", "ONCA", "SCC", "TCC"),
                  start_date: str = "2025-06-01", per_query: int = 25,
                  limit: int = 20, min_chars: int = 4000,
                  sleep_seconds: float = 0.3) -> str:
    """Collect recent decisions into a holdout dataset with full text.

    Cases are gathered across seed queries and courts, de-duplicated by citation,
    filtered to `start_date` or later, and written with their full decision text.
    """
    seen: Dict[str, Dict] = {}
    for ds in datasets:
        for q in SEED_QUERIES:
            try:
                hits = search(q, dataset=ds, start_date=start_date, size=per_query)
            except Exception as e:
                print(f"  search failed ({ds}/{q}): {e}")
                continue
            for rec in hits:
                cit = rec.get("citation_en")
                date = (rec.get("document_date_en") or "")[:10]
                if cit and cit not in seen and date >= start_date:
                    seen[cit] = rec
            if sleep_seconds:
                sleep(sleep_seconds)
        print(f"  {ds}: pool now {len(seen)} unique recent cases")

    # Newest first, then fetch full text until we have `limit` usable cases.
    ordered = sorted(seen.values(), key=lambda r: r.get("document_date_en", ""), reverse=True)
    rows: List[Dict] = []
    for rec in ordered:
        if len(rows) >= limit:
            break
        cit = rec["citation_en"]
        try:
            text = fetch_text(cit)
        except Exception as e:
            print(f"  fetch failed ({cit}): {e}")
            continue
        if len(text) < min_chars:
            continue
        rows.append({
            "Title": rec.get("name_en", ""),
            "Citation": cit,
            "Court": rec.get("dataset", ""),
            "Date": (rec.get("document_date_en") or "")[:10],
            "Url": rec.get("url_en", ""),
            "case_text": text,
        })
        print(f"  + {cit} {rows[-1]['Date']} {rows[-1]['Title'][:50]} ({len(text)} chars)")
        if sleep_seconds:
            sleep(sleep_seconds)

    fields = ["Title", "Citation", "Court", "Date", "Url", "case_text"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nHoldout written: {out_csv} ({len(rows)} cases, all >= {start_date})")
    return out_csv


def main(argv: List[str]) -> None:
    ap = argparse.ArgumentParser(description="A2AJ Canadian legal data client")
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("holdout", help="build a recent-cases holdout dataset")
    h.add_argument("--out", default="random_cases_holdout.csv")
    h.add_argument("--start", default="2025-06-01", help="earliest decision date (YYYY-MM-DD)")
    h.add_argument("--limit", type=int, default=20)
    h.add_argument("--datasets", default="FC,FCA,ONCA,SCC,TCC")

    c = sub.add_parser("coverage", help="print dataset coverage")

    args = ap.parse_args(argv)
    if args.cmd == "coverage":
        for r in sorted(coverage(), key=lambda x: x["dataset"]):
            print(f"{r['dataset']:6s} {r['earliest_document_date'][:10]}..{r['latest_document_date'][:10]} "
                  f"n={r['number_of_documents']}  {r['description_en']}")
    elif args.cmd == "holdout":
        build_holdout(args.out, datasets=tuple(args.datasets.split(",")),
                      start_date=args.start, limit=args.limit)


if __name__ == "__main__":
    main(sys.argv[1:])
