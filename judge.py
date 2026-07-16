"""LLM-as-judge scorer for Canadian case-law briefs.

A strong judge model grades each generated brief section on a 1-5 rubric —
accuracy, completeness, and groundedness (the inverse of hallucination) — with a
short rationale, returning structured JSON. This complements the embedding
cosine score in rate_embedding.py: cosine measures topical closeness, the judge
measures whether the brief is actually correct and free of invented content.

Two modes:
  * reference : grade the brief against the human-authored reference brief
                (columns human_facts ... human_ratio). Used for the closed-book
                dataset that ships with human briefs.
  * source    : grade the brief for faithfulness to the full decision text
                (a `case_text` column). Reference-free and contamination-
                resistant, so it works on recent / open-book cases that have no
                curated human brief.

Usage:
    python judge.py results/evaluated_case_model_results_with_section_similarity.csv
    python judge.py <input.csv> <output.csv> --mode source --judge google/gemini-2.5-pro

Caveats (documented on purpose): LLM judges carry self-preference bias and cost.
Use a judge from a different family than the leading contestants, keep the judge
model + prompt pinned for comparability, and validate against a small
human-labelled sample before trusting absolute numbers.
"""

import os
import sys
from pathlib import Path
from time import sleep
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from benchmark import SECTIONS, make_valid_json
from llm.openrouter import OpenRouterWrapper

load_dotenv()

# A capable judge from a family other than the current leaders (Anthropic /
# OpenAI), to blunt self-preference bias. Override with --judge or JUDGE_MODEL.
DEFAULT_JUDGE = os.environ.get("JUDGE_MODEL", "google/gemini-2.5-pro")

CRITERIA = ["accuracy", "completeness", "groundedness"]

# Max characters of decision text handed to the judge in source mode. Large
# enough for most decisions; long judgments are truncated with a marker.
SOURCE_CHAR_BUDGET = 120_000

_RUBRIC = (
    "Score each of accuracy, completeness, and groundedness from 1 to 5:\n"
    "  accuracy      1 = wrong/contradicts the source, 5 = fully correct.\n"
    "  completeness  1 = misses the key points, 5 = captures all key points.\n"
    "  groundedness  1 = contains invented facts, parties, holdings or citations,\n"
    "                5 = every claim is supported by the source (no hallucination).\n"
    "Grade like a demanding law professor. Before scoring, silently enumerate every "
    "error, omission, or unsupported claim you can find; base scores on that list. "
    "Reserve 5 for flawless work — any identified omission or imprecision caps the "
    "criterion at 4, and any substantive error caps it at 3 or below. Use the full "
    "1-5 range; most competent-but-imperfect sections should land at 3 or 4.\n"
    "If a candidate section is empty, 'ERROR', or 'I don't know', score all three 1."
)

SYSTEM_PROMPT = (
    "You are a meticulous Canadian legal evaluator grading AI-generated case "
    "briefs. Be strict, calibrated, and consistent. Judge only against the "
    "material provided — do not reward fluent writing that is not supported. "
    "Respond with a single JSON object and nothing else."
)


def _norm(x) -> Optional[float]:
    """Map a 1-5 rubric score to 0-1; None if unparseable."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    v = max(1.0, min(5.0, v))
    return round((v - 1.0) / 4.0, 4)


class LLMJudge:
    def __init__(self, judge_model: str = DEFAULT_JUDGE, **wrapper_kwargs):
        self.judge_model = judge_model
        cfg = {
            "model_id": judge_model,
            "display_name": judge_model,
            "temperature": 0,
            "max_tokens": 1600,
            "api_key": os.environ.get("OPENROUTER_API_KEY"),
        }
        cfg.update(wrapper_kwargs)
        self.llm = OpenRouterWrapper(cfg)

    @staticmethod
    def _has_reference(reference: Optional[Dict[str, str]], s: str) -> bool:
        v = (reference or {}).get(s)
        return isinstance(v, str) and v.strip() != "" and v.strip().lower() != "nan"

    def _gradable_sections(self, reference: Optional[Dict[str, str]],
                           source_text: Optional[str]) -> list:
        """Sections that can legitimately be graded.

        Source mode grades everything against the decision text. Reference mode
        grades only sections that actually have a human reference — a missing
        reference is a dataset gap, and judging against nothing would score the
        candidate on material the grader never saw.
        """
        if source_text is not None:
            return list(SECTIONS)
        return [s for s in SECTIONS if self._has_reference(reference, s)]

    def _prompt(self, case_name: str, ai: Dict[str, str], reference: Optional[Dict[str, str]],
                source_text: Optional[str], sections: list) -> str:
        keys = ", ".join(f'"{s}"' for s in sections)
        schema = (
            "{" + ", ".join(
                f'"{s}": {{"accuracy": 1-5, "completeness": 1-5, "groundedness": 1-5}}'
                for s in sections
            ) + "}"
        )
        parts = [f'Case: "{case_name}".', "", _RUBRIC, ""]

        if source_text is not None:
            text = source_text[:SOURCE_CHAR_BUDGET]
            if len(source_text) > SOURCE_CHAR_BUDGET:
                text += "\n[...decision text truncated...]"
            parts += [
                "Grade the candidate brief for faithfulness to this decision text:",
                "=== DECISION TEXT ===", text, "=== END DECISION TEXT ===", "",
            ]
        elif reference is not None:
            ref = "\n".join(f"[{s}] {reference.get(s,'')}" for s in sections)
            parts += [
                "Grade the candidate brief against this reference brief:",
                "=== REFERENCE BRIEF ===", ref, "=== END REFERENCE BRIEF ===", "",
            ]

        cand = "\n".join(f"[{s}] {ai.get(f'ai_{s}','')}" for s in sections)
        parts += [
            "=== CANDIDATE BRIEF ===", cand, "=== END CANDIDATE BRIEF ===", "",
            f"Return JSON with exactly these keys {{{keys}}}, each an object of the "
            f"three integer scores. Shape: {schema}. Output only the JSON.",
        ]
        return "\n".join(parts)

    @staticmethod
    def _is_failed_brief(ai: Dict[str, str]) -> bool:
        """True when every section is empty, ERROR, or a refusal."""
        for s in SECTIONS:
            v = str(ai.get(f"ai_{s}", "")).strip().lower()
            if v and v not in ("error", "nan") and "i don't know" not in v:
                return False
        return True

    def judge_row(self, case_name: str, ai: Dict[str, str],
                  reference: Optional[Dict[str, str]] = None,
                  source_text: Optional[str] = None) -> Dict[str, Optional[float]]:
        sections = self._gradable_sections(reference, source_text)
        if not sections:
            out = {f"judge_{s}_{c}": None for s in SECTIONS for c in CRITERIA}
            out.update({f"judge_{s}": None for s in SECTIONS})
            out["judge_overall"] = None
            return out

        # Failed briefs score the rubric minimum without an API call.
        if self._is_failed_brief(ai):
            out: Dict[str, Optional[float]] = {}
            for s in SECTIONS:
                gradable = s in sections
                for c in CRITERIA:
                    out[f"judge_{s}_{c}"] = 0.0 if gradable else None
                out[f"judge_{s}"] = 0.0 if gradable else None
            out["judge_overall"] = 0.0
            return out

        prompt = self._prompt(case_name, ai, reference, source_text, sections)
        raw = self.llm.invoke(prompt, context=SYSTEM_PROMPT)
        import json
        try:
            parsed = json.loads(make_valid_json(raw))
        except (ValueError, json.JSONDecodeError):
            parsed = {}

        out: Dict[str, Optional[float]] = {}
        sec_composites = []
        for s in SECTIONS:
            if s not in sections:
                # No reference for this section — excluded, not scored.
                for c in CRITERIA:
                    out[f"judge_{s}_{c}"] = None
                out[f"judge_{s}"] = None
                continue
            block = parsed.get(s, {}) if isinstance(parsed.get(s), dict) else {}
            crit_vals = []
            for c in CRITERIA:
                v = _norm(block.get(c))
                out[f"judge_{s}_{c}"] = v
                if v is not None:
                    crit_vals.append(v)
            comp = round(sum(crit_vals) / len(crit_vals), 4) if crit_vals else None
            out[f"judge_{s}"] = comp
            if comp is not None:
                sec_composites.append(comp)
        out["judge_overall"] = (
            round(sum(sec_composites) / len(sec_composites), 4) if sec_composites else None
        )
        return out

    def evaluate_results(self, input_file: str, output_file: Optional[str] = None,
                         mode: str = "reference", text_column: str = "case_text",
                         sleep_seconds: float = 0.4, max_workers: int = 1) -> str:
        df = pd.read_csv(input_file)
        for col in [f"ai_{s}" for s in SECTIONS]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        if mode == "source" and text_column not in df.columns:
            raise ValueError(
                f"source mode needs a '{text_column}' column of decision text"
            )

        def _one(payload):
            i, name, ai, ref, src = payload
            try:
                return self.judge_row(name, ai, ref, src)
            except Exception as e:
                print(f"  judge error on row {i}: {e}")
                return {}

        payloads = []
        for i, row in df.iterrows():
            ai = {f"ai_{s}": row.get(f"ai_{s}", "") for s in SECTIONS}
            ref = {s: row.get(f"human_{s}", "") for s in SECTIONS} if mode == "reference" else None
            src = str(row.get(text_column, "")) if mode == "source" else None
            payloads.append((i, str(row.get("Case_Name", "")), ai, ref, src))

        if max_workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                records: List[Dict] = []
                for i, rec in enumerate(ex.map(_one, payloads)):
                    records.append(rec)
                    if (i + 1) % 20 == 0:
                        print(f"  judged {i+1}/{len(df)} rows")
        else:
            records = []
            for p in payloads:
                records.append(_one(p))
                if (p[0] + 1) % 10 == 0:
                    print(f"  judged {p[0]+1}/{len(df)} rows")
                if sleep_seconds:
                    sleep(sleep_seconds)

        judge_df = pd.DataFrame(records)
        for c in judge_df.columns:
            df[c] = judge_df[c].values
        df["judge_model"] = self.judge_model

        Path("results").mkdir(exist_ok=True)
        if output_file is None:
            output_file = "results/judged_case_model_results.csv"
        df.to_csv(output_file, index=False)
        print(f"Judge evaluation complete ({self.judge_model}). Saved to {output_file}")
        return output_file


def main(argv: List[str]) -> str:
    if not argv:
        print("Usage: python judge.py <input_csv> [output_csv] [--mode reference|source] [--judge <model_id>]")
        sys.exit(1)
    input_file = argv[0]
    output_file = argv[1] if len(argv) > 1 and not argv[1].startswith("--") else None
    mode = "reference"
    judge_model = DEFAULT_JUDGE
    for i, a in enumerate(argv):
        if a == "--mode" and i + 1 < len(argv):
            mode = argv[i + 1]
        if a == "--judge" and i + 1 < len(argv):
            judge_model = argv[i + 1]
    judge = LLMJudge(judge_model=judge_model)
    return judge.evaluate_results(input_file, output_file, mode=mode)


if __name__ == "__main__":
    main(sys.argv[1:])
