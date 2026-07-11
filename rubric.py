"""HealthBench-style per-case rubric evaluation.

Instead of holistic 1-5 judging (which showed a ceiling effect — most frontier
briefs rate "perfect"), each case gets a checklist of specific, binary,
point-weighted criteria, and a grader model only verifies whether each
criterion is met. Two phases:

  authoring  — a strong model writes 12-20 atomic criteria per case, grounded
               in the human reference brief (closed-book cases) or the full
               decision text (open-book/holdout cases). Includes negative
               criteria ("must not misstate the parties") to catch
               hallucinations. Rubrics are cached in rubrics/rubrics.json,
               versioned, and reused across runs.
  grading    — for each (model, case) brief, the grader checks every criterion
               (met / not met) and the score is met-weight / total-weight,
               overall and per section.

Following HealthBench's design, checking a specific criterion is a far easier
and more verifiable task than holistic scoring, which is what restores
discrimination at the top of the leaderboard.

    python rubric.py author              # write missing rubrics for known cases
    python rubric.py grade <results.csv> # add rubric_* columns
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from benchmark import SECTIONS, make_valid_json
from llm.openrouter import OpenRouterWrapper

load_dotenv()

RUBRICS_PATH = "rubrics/rubrics.json"

# Advanced models, chosen for capability (family separation deliberately waived
# — documented on the site). Author needs deep legal knowledge; grader needs
# precise long-context verification.
AUTHOR_MODEL = os.environ.get("RUBRIC_AUTHOR_MODEL", "anthropic/claude-opus-4.8")
GRADER_MODEL = os.environ.get("RUBRIC_GRADER_MODEL", "anthropic/claude-opus-4.8")

SOURCE_CHAR_BUDGET = 120_000

AUTHOR_SYSTEM = (
    "You are a senior Canadian law professor writing grading rubrics for case "
    "briefs. Your criteria must be atomic (one verifiable fact or point each), "
    "specific to this case, and objectively checkable by reading a brief. "
    "Respond with a single JSON object and nothing else."
)

GRADER_SYSTEM = (
    "You are a precise grader. For each rubric criterion, decide strictly "
    "whether the candidate brief meets it. A criterion is met only if the brief "
    "explicitly satisfies it; generous inference is not allowed. For "
    "must_not criteria, 'met' means the brief does NOT commit the error. "
    "Respond with a single JSON object and nothing else."
)


def _author_prompt(case_name: str, citation: str, grounding_label: str, grounding: str) -> str:
    sections = ", ".join(SECTIONS)
    return f"""Write a grading rubric for a case brief of the Canadian case "{case_name}" ({citation}).

Ground every criterion strictly in this material:
=== {grounding_label} ===
{grounding}
=== END ===

Requirements:
- 12 to 20 criteria total, spread across the sections: {sections}.
- Each criterion is atomic and verifiable, e.g. "States that the parties signed a separation agreement after ~15 months of negotiation" — never vague ("accurately describes the facts").
- Weight each criterion 1-3 (3 = essential holding/ratio elements, 2 = important, 1 = supporting detail).
- Include 2-4 negative criteria (polarity "must_not") for likely errors or hallucinations, e.g. "Does not misstate the names of the parties" or "Does not invent a dissent that did not occur".
- Criteria must be answerable from the brief alone.

Return JSON only, in exactly this shape:
{{"criteria": [{{"id": "c1", "section": "facts|issue|decision|reasons|ratio", "text": "...", "weight": 1, "polarity": "must_include|must_not"}}, ...]}}"""


def _grade_prompt(case_name: str, criteria: List[Dict], ai: Dict[str, str]) -> str:
    crit_lines = "\n".join(
        f'  {c["id"]} [{c["section"]}, {c["polarity"]}, weight {c["weight"]}]: {c["text"]}'
        for c in criteria
    )
    brief = "\n".join(f"[{s}] {ai.get(f'ai_{s}', '')}" for s in SECTIONS)
    ids = ", ".join(f'"{c["id"]}": true|false' for c in criteria)
    return f"""Case: "{case_name}".

Rubric criteria:
{crit_lines}

=== CANDIDATE BRIEF ===
{brief}
=== END CANDIDATE BRIEF ===

For each criterion, answer true (met) or false (not met). For must_not criteria,
true means the brief does NOT commit the error. If the brief is empty, "ERROR",
or "I don't know", every must_include criterion is false and every must_not
criterion is true.

Return JSON only: {{{ids}}}"""


# ---------------------------------------------------------------- authoring
def load_rubrics(path: str = RUBRICS_PATH) -> Dict:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"version": 1, "author_model": AUTHOR_MODEL, "cases": {}}


def save_rubrics(data: Dict, path: str = RUBRICS_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def author_rubric(llm: OpenRouterWrapper, case_name: str, citation: str,
                  reference: Optional[Dict[str, str]] = None,
                  source_text: Optional[str] = None) -> Optional[List[Dict]]:
    """Author one case's rubric from its reference brief or decision text."""
    if source_text:
        grounding_label = "FULL DECISION TEXT"
        grounding = source_text[:SOURCE_CHAR_BUDGET]
    elif reference and any(v for v in reference.values()):
        grounding_label = "HUMAN REFERENCE BRIEF"
        grounding = "\n".join(f"[{s}] {reference.get(s, '')}" for s in SECTIONS)
    else:
        return None

    raw = llm.invoke(_author_prompt(case_name, citation, grounding_label, grounding),
                     context=AUTHOR_SYSTEM)
    try:
        criteria = json.loads(make_valid_json(raw)).get("criteria", [])
    except (ValueError, json.JSONDecodeError):
        return None

    clean = []
    for i, c in enumerate(criteria):
        sec = str(c.get("section", "")).lower()
        if sec not in SECTIONS:
            continue
        try:
            weight = max(1, min(3, int(c.get("weight", 1))))
        except (TypeError, ValueError):
            weight = 1
        clean.append({
            "id": f"c{i+1}",
            "section": sec,
            "text": str(c.get("text", "")).strip(),
            "weight": weight,
            "polarity": "must_not" if c.get("polarity") == "must_not" else "must_include",
        })
    return clean if len(clean) >= 8 else None


def author_missing(cases: List[Dict], rubrics: Dict, sleep_seconds: float = 0.3) -> Dict:
    """Author rubrics for any case not yet in the cache."""
    llm = OpenRouterWrapper({
        "model_id": AUTHOR_MODEL, "temperature": 0, "max_tokens": 8192,
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
    })
    for case in cases:
        name = case["title"]
        if name in rubrics["cases"]:
            continue
        print(f"  authoring rubric: {name}")
        crit = author_rubric(llm, name, case.get("citation", ""),
                             reference=case.get("human"),
                             source_text=case.get("case_text") or None)
        if crit:
            rubrics["cases"][name] = {
                "citation": case.get("citation", ""),
                "grounding": "text" if case.get("case_text") else "reference",
                "author_model": AUTHOR_MODEL,
                "created": datetime.now().strftime("%Y-%m-%d"),
                "criteria": crit,
            }
            print(f"    {len(crit)} criteria "
                  f"({sum(1 for c in crit if c['polarity']=='must_not')} negative)")
        else:
            print("    FAILED to author (skipping)")
        if sleep_seconds:
            sleep(sleep_seconds)
    return rubrics


# ---------------------------------------------------------------- grading
class RubricGrader:
    def __init__(self, grader_model: str = GRADER_MODEL):
        self.grader_model = grader_model
        self.llm = OpenRouterWrapper({
            "model_id": grader_model, "temperature": 0, "max_tokens": 2048,
            "api_key": os.environ.get("OPENROUTER_API_KEY"),
        })

    def grade_row(self, case_name: str, criteria: List[Dict],
                  ai: Dict[str, str]) -> Dict:
        raw = self.llm.invoke(_grade_prompt(case_name, criteria, ai),
                              context=GRADER_SYSTEM)
        try:
            verdicts = json.loads(make_valid_json(raw))
        except (ValueError, json.JSONDecodeError):
            verdicts = {}

        out: Dict = {}
        got, tot = {s: 0.0 for s in SECTIONS}, {s: 0.0 for s in SECTIONS}
        details = {}
        for c in criteria:
            met = bool(verdicts.get(c["id"], False))
            details[c["id"]] = met
            tot[c["section"]] += c["weight"]
            if met:
                got[c["section"]] += c["weight"]
        for s in SECTIONS:
            out[f"rubric_{s}"] = round(got[s] / tot[s], 4) if tot[s] else None
        total_w = sum(tot.values())
        out["rubric_overall"] = round(sum(got.values()) / total_w, 4) if total_w else None
        out["rubric_details"] = json.dumps(details)
        return out

    def evaluate_results(self, input_file: str, rubrics_path: str = RUBRICS_PATH,
                         output_file: Optional[str] = None,
                         sleep_seconds: float = 0.15) -> str:
        df = pd.read_csv(input_file)
        rubrics = load_rubrics(rubrics_path)

        records = []
        for i, row in df.iterrows():
            name = str(row.get("Case_Name", ""))
            entry = rubrics["cases"].get(name)
            if not entry:
                records.append({})
                continue
            ai = {f"ai_{s}": row.get(f"ai_{s}", "") for s in SECTIONS}
            try:
                records.append(self.grade_row(name, entry["criteria"], ai))
            except Exception as e:
                print(f"  grade error row {i} ({row.get('Model_ID','?')}/{name}): {e}")
                records.append({})
            if (i + 1) % 10 == 0:
                print(f"  graded {i+1}/{len(df)} rows")
            if sleep_seconds:
                sleep(sleep_seconds)

        rub_df = pd.DataFrame(records)
        for c in rub_df.columns:
            df[c] = rub_df[c].values
        df["rubric_model"] = self.grader_model

        if output_file is None:
            output_file = input_file.replace(".csv", "_rubric.csv")
        df.to_csv(output_file, index=False)
        print(f"Rubric grading complete ({self.grader_model}). Saved to {output_file}")
        return output_file


# ---------------------------------------------------------------- cli
def main(argv: List[str]) -> None:
    if not argv:
        print("Usage: python rubric.py author [cases.csv holdout.csv ...] | grade <results.csv> [output.csv]")
        sys.exit(1)
    cmd = argv[0]
    if cmd == "author":
        from benchmark import load_cases
        rubrics = load_rubrics()
        files = argv[1:] or ["random_cases.csv"]
        for f in files:
            cases = load_cases(f)
            rubrics = author_missing(cases, rubrics)
        save_rubrics(rubrics)
        print(f"Rubrics saved: {RUBRICS_PATH} ({len(rubrics['cases'])} cases)")
    elif cmd == "grade":
        grader = RubricGrader()
        out = argv[2] if len(argv) > 2 else None
        grader.evaluate_results(argv[1], output_file=out)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
