"""Render an evaluated results CSV into a self-contained HTML report.

    python report.py results/evaluated_case_model_results_with_section_similarity.csv
    python report.py <input.csv> <output.html>

The page shows, per model, the section-by-section similarity scores plus the
AI brief side-by-side with the human-authored brief.
"""

import html
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

SECTIONS = ["facts", "issue", "decision", "reasons", "ratio"]


def _score_color(score: float) -> str:
    """Red (0) -> amber (0.5) -> green (1) background for a similarity score."""
    if pd.isna(score):
        return "#e5e7eb"
    s = max(0.0, min(1.0, float(score)))
    if s < 0.5:
        r, g = 220, int(60 + (s / 0.5) * 150)
    else:
        r, g = int(220 - ((s - 0.5) / 0.5) * 150), 210
    return f"rgb({r},{g},90)"


def _cell(value: str) -> str:
    return html.escape(str(value)) if value and not pd.isna(value) else "<em>—</em>"


def build_html(df: pd.DataFrame, source: str) -> str:
    sim_cols = [f"{s}_similarity" for s in SECTIONS if f"{s}_similarity" in df.columns]
    overall = df[sim_cols].mean(axis=1) if sim_cols else None

    # Summary: average similarity per model.
    summary_rows = ""
    if sim_cols:
        agg = df.groupby("Model_ID")[sim_cols].mean()
        agg["overall"] = agg.mean(axis=1)
        for model, row in agg.sort_values("overall", ascending=False).iterrows():
            cells = "".join(
                f'<td style="background:{_score_color(row[c])}">{row[c]:.3f}</td>'
                for c in sim_cols
            )
            summary_rows += (
                f"<tr><td class='model'>{html.escape(str(model))}</td>{cells}"
                f"<td class='overall' style='background:{_score_color(row['overall'])}'>"
                f"{row['overall']:.3f}</td></tr>"
            )

    summary_head = "".join(f"<th>{s.title()}</th>" for s in SECTIONS if f"{s}_similarity" in df.columns)

    # Detail cards: one per (model, case) with AI vs human briefs.
    cards = ""
    for i, row in df.iterrows():
        score = f"{overall[i]:.3f}" if overall is not None and not pd.isna(overall[i]) else "n/a"
        sec_rows = ""
        for s in SECTIONS:
            sim = row.get(f"{s}_similarity")
            badge = (
                f'<span class="badge" style="background:{_score_color(sim)}">{sim:.3f}</span>'
                if sim is not None and not pd.isna(sim)
                else ""
            )
            sec_rows += (
                f"<tr><th>{s.title()} {badge}</th>"
                f"<td>{_cell(row.get(f'ai_{s}'))}</td>"
                f"<td>{_cell(row.get(f'human_{s}'))}</td></tr>"
            )
        cards += f"""
        <div class="card">
          <h3>{html.escape(str(row.get('Model_ID','?')))} &middot;
              <span class="case">{html.escape(str(row.get('Case_Name','?')))}</span>
              <span class="overall-badge" style="background:{_score_color(overall[i] if overall is not None else float('nan'))}">avg {score}</span>
          </h3>
          <table class="brief">
            <thead><tr><th>Section</th><th>AI brief</th><th>Human brief</th></tr></thead>
            <tbody>{sec_rows}</tbody>
          </table>
        </div>"""

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Canadian Case-Law LLM Benchmark</title>
<style>
  :root {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; }}
  body {{ margin: 0; background: #f8fafc; color: #0f172a; }}
  header {{ background: #0f172a; color: #fff; padding: 24px 32px; }}
  header h1 {{ margin: 0 0 4px; font-size: 22px; }}
  header p {{ margin: 0; color: #94a3b8; font-size: 13px; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 24px 32px 64px; }}
  h2 {{ font-size: 16px; text-transform: uppercase; letter-spacing: .05em; color: #475569; margin-top: 32px; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; border-radius: 8px; overflow: hidden;
           box-shadow: 0 1px 3px rgba(0,0,0,.08); font-size: 14px; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eef2f7; vertical-align: top; }}
  .summary th {{ background: #f1f5f9; }}
  .summary td {{ text-align: center; font-variant-numeric: tabular-nums; }}
  .summary td.model {{ text-align: left; font-weight: 600; }}
  .summary td.overall {{ font-weight: 700; }}
  .card {{ margin-top: 18px; }}
  .card h3 {{ font-size: 15px; margin: 0 0 8px; }}
  .card .case {{ color: #2563eb; }}
  .brief th {{ width: 14%; background: #f8fafc; }}
  .brief td {{ width: 43%; white-space: pre-wrap; }}
  .badge, .overall-badge {{ display: inline-block; padding: 1px 7px; border-radius: 10px; font-size: 12px;
            font-weight: 700; color: #1e293b; }}
  .overall-badge {{ float: right; }}
</style></head>
<body>
<header>
  <h1>LLM Benchmark for Canadian Case Law</h1>
  <p>{len(df)} result row(s) &middot; source: {html.escape(source)} &middot; generated {generated}</p>
</header>
<main>
  <h2>Average section similarity by model</h2>
  <table class="summary">
    <thead><tr><th>Model</th>{summary_head}<th>Overall</th></tr></thead>
    <tbody>{summary_rows or '<tr><td colspan="7"><em>No similarity columns found.</em></td></tr>'}</tbody>
  </table>
  <h2>AI vs human briefs</h2>
  {cards}
</main></body></html>"""


def main(argv):
    if not argv:
        print("Usage: python report.py <evaluated_csv> [output_html]")
        sys.exit(1)
    input_file = argv[0]
    output_file = argv[1] if len(argv) > 1 else "results/report.html"
    df = pd.read_csv(input_file)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(build_html(df, input_file), encoding="utf-8")
    print(f"Report written to {output_file}")
    return output_file


if __name__ == "__main__":
    main(sys.argv[1:])
