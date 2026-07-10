"""Generate the LexBench landing page from evaluated benchmark results.

    python lexbench.py [evaluated_csv] [output_html]

Defaults: reads results/evaluated_case_model_results_with_section_similarity.csv
and writes docs/index.html (plus docs/data.json and docs/.nojekyll). Rerun after
every benchmark to refresh the published site.

Aggregates whatever signals are present per model:
  * LLM-judge score  (primary when judge_* columns exist) with bootstrap CI
  * embedding cosine (secondary / fallback primary)
  * hallucination-safety (judge groundedness), format-compliance, refusal rate,
    cost and latency
If a results file carries a `mode` column with both closed- and open-book rows,
each is rendered as its own leaderboard track.
"""

import html
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

import stats

DEFAULT_INPUT = "results/evaluated_case_model_results_with_section_similarity.csv"
DEFAULT_OUTPUT = "docs/index.html"

SECTIONS = ["facts", "issue", "decision", "reasons", "ratio"]
SECTION_COLOR = {
    "facts": "#4a7c59", "issue": "#4e6d8c", "decision": "#a9762f",
    "reasons": "#7d6b9e", "ratio": "#8c5a4a",
}
TRACK_LABELS = {
    "closed": "Closed-book (recall from case name)",
    "open": "Open-book (brief from decision text)",
}


# ---------------------------------------------------------------- helpers
def _developer_map(models_csv: str = "ai_models.csv") -> dict:
    try:
        m = pd.read_csv(models_csv)
        return dict(zip(m["display_name"], m["developer"]))
    except Exception:
        return {}


def _lerp(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


_ANCHORS = [(0.30, (176, 137, 104)), (0.55, (201, 162, 39)), (0.80, (63, 111, 95))]


def _score_rgb(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return (150, 150, 145)
    v = max(0.0, min(1.0, float(v)))
    if v <= _ANCHORS[0][0]:
        return _ANCHORS[0][1]
    if v >= _ANCHORS[-1][0]:
        return _ANCHORS[-1][1]
    for (x0, c0), (x1, c1) in zip(_ANCHORS, _ANCHORS[1:]):
        if x0 <= v <= x1:
            return _lerp(c0, c1, (v - x0) / (x1 - x0))
    return _ANCHORS[-1][1]


def _rgb(c):
    return f"rgb({c[0]},{c[1]},{c[2]})"


def _rgba(c, a):
    return f"rgba({c[0]},{c[1]},{c[2]},{a})"


def _fmt(v, nd=3):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) and not pd.isna(v) else "—"


def _pct(v):
    return f"{100*v:.0f}%" if isinstance(v, (int, float)) and not pd.isna(v) else "—"


def _col_mean(sub, col):
    if col in sub and sub[col].notna().any():
        return float(sub[col].mean())
    return None


def _row_overall(sub, cols):
    """Per-row mean across the given columns, as a list (for bootstrap)."""
    present = [c for c in cols if c in sub]
    if not present:
        return []
    return sub[present].mean(axis=1, skipna=True).dropna().tolist()


# ---------------------------------------------------------------- aggregation
def aggregate(df: pd.DataFrame):
    dev = _developer_map()
    present = [s for s in SECTIONS if f"{s}_similarity" in df.columns or f"judge_{s}" in df.columns]
    has_judge = any(f"judge_{s}" in df.columns for s in SECTIONS) or "judge_overall" in df.columns
    primary = "judge" if has_judge else "cosine"

    sim_cols = [f"{s}_similarity" for s in SECTIONS if f"{s}_similarity" in df.columns]
    judge_cols = [f"judge_{s}" for s in SECTIONS if f"judge_{s}" in df.columns]
    ground_cols = [f"judge_{s}_groundedness" for s in SECTIONS
                   if f"judge_{s}_groundedness" in df.columns]
    ai_cols = [f"ai_{s}" for s in SECTIONS if f"ai_{s}" in df.columns]

    rows = []
    for model, sub in df.groupby("Model_ID"):
        cosine = {s: _col_mean(sub, f"{s}_similarity") for s in SECTIONS}
        judge = {s: _col_mean(sub, f"judge_{s}") for s in SECTIONS}
        cosine_overall = _col_mean(sub, None) if False else (
            float(pd.Series([v for v in cosine.values() if v is not None]).mean())
            if any(v is not None for v in cosine.values()) else None)
        judge_overall = (float(pd.Series([v for v in judge.values() if v is not None]).mean())
                         if any(v is not None for v in judge.values()) else None)

        # primary per-row values -> bootstrap CI
        if primary == "judge":
            per_row = (sub["judge_overall"].dropna().tolist()
                       if "judge_overall" in sub else _row_overall(sub, judge_cols))
        else:
            per_row = _row_overall(sub, sim_cols)
        mean, lo, hi = stats.bootstrap_ci(per_row)

        rows.append({
            "model": str(model),
            "developer": str(dev.get(model, "")),
            "n": int(len(sub)),
            "cosine": cosine, "judge": judge,
            "cosine_overall": None if cosine_overall is None else round(cosine_overall, 4),
            "judge_overall": None if judge_overall is None else round(judge_overall, 4),
            "primary_overall": mean, "ci_lo": lo, "ci_hi": hi,
            "groundedness": (round(float(sub[ground_cols].mean(axis=1).mean()), 4)
                             if ground_cols else None),
            "format_rate": _col_mean(sub, "format_ok"),
            "refusal_rate": _col_mean(sub, "refused"),
            "error_rate": (int((sub[ai_cols] == "ERROR").all(axis=1).sum()) / len(sub)
                           if ai_cols else None),
            "avg_cost": _col_mean(sub, "cost_usd"),
            "avg_latency": _col_mean(sub, "latency_s"),
        })

    rows.sort(key=lambda r: (r["primary_overall"] is not None, r["primary_overall"] or 0),
              reverse=True)

    # flag ties: CI of a model overlapping the model ranked above it
    for i in range(1, len(rows)):
        a, b = rows[i], rows[i - 1]
        rows[i]["tied_above"] = (a["ci_hi"] is not None and b["ci_lo"] is not None
                                 and a["ci_hi"] >= b["ci_lo"])
    if rows:
        rows[0]["tied_above"] = False

    return {"rows": rows, "sections": present, "primary": primary,
            "primary_label": "Judge score" if primary == "judge" else "Cosine similarity"}


# ---------------------------------------------------------------- svg chart
def build_chart_svg(rows, sections, primary):
    if not rows or not sections:
        return ""
    W, H = 860, 430
    ml, mr, mt, mb = 46, 18, 22, 104
    pw, ph = W - ml - mr, H - mt - mb
    n = len(rows)
    gw = pw / n
    inner = gw * 0.74
    bw = inner / len(sections)
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Score by section and model" preserveAspectRatio="xMidYMid meet">']
    for t in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = mt + ph * (1 - t)
        p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" class="grid"/>')
        p.append(f'<text x="{ml-8}" y="{y+3.5:.1f}" class="ytick">{t:.1f}</text>')
    for gi, r in enumerate(rows):
        gx = ml + gw * gi + (gw - inner) / 2
        for si, s in enumerate(sections):
            v = (r[primary].get(s) if primary in r else None) or 0
            bh = ph * v
            bx = gx + si * bw
            p.append(f'<rect x="{bx:.1f}" y="{mt+ph-bh:.1f}" width="{max(bw-2,1):.1f}" '
                     f'height="{bh:.1f}" fill="{SECTION_COLOR[s]}"><title>{html.escape(r["model"])} · {s}: {v:.3f}</title></rect>')
        cx = ml + gw * gi + gw / 2
        ly = mt + ph + 14
        p.append(f'<text x="{cx:.1f}" y="{ly:.1f}" class="xtick" transform="rotate(-22 {cx:.1f} {ly:.1f})">{html.escape(r["model"])}</text>')
    p.append("</svg>")
    return "".join(p)


# ---------------------------------------------------------------- leaderboard
def build_leaderboard(agg):
    rows, primary_label = agg["rows"], agg["primary_label"]
    max_n = max((r["n"] for r in rows), default=0)
    body = []
    for i, r in enumerate(rows, 1):
        ov = r["primary_overall"]
        c = _score_rgb(ov)
        pct = 0 if ov is None else max(0, min(100, ov * 100))
        ci = (f'<span class="ci">[{r["ci_lo"]:.2f}–{r["ci_hi"]:.2f}]</span>'
              if r["ci_lo"] is not None and r["n"] > 1 else "")
        tie = '<span class="tie" title="CI overlaps the model above — gap not significant">≈</span>' if r.get("tied_above") else ""
        partial = f'<span class="tag">partial · {r["n"]}/{max_n}</span>' if r["n"] < max_n else ""
        body.append(f"""<tr>
  <td class="rank">{i}</td>
  <td class="model"><span class="mname">{html.escape(r["model"])}{tie}</span>
      <span class="dev">{html.escape(r["developer"])}</span></td>
  <td class="overall"><span class="ov-num" style="color:{_rgb(c)}">{_fmt(ov)}</span>
      <span class="bar"><span class="fill" style="width:{pct:.1f}%;background:{_rgb(c)}"></span></span>
      {ci}</td>
  <td class="num">{_fmt(r["cosine_overall"], 2)}</td>
  <td class="num">{_fmt(r["groundedness"], 2)}</td>
  <td class="num">{_pct(r["format_rate"])}</td>
  <td class="num">{_pct(r["refusal_rate"])}</td>
  <td class="num">{('$'+format(r["avg_cost"],'.4f')) if r["avg_cost"] is not None else '—'}</td>
  <td class="cases">{r["n"]}{partial}</td>
</tr>""")
    return f"""<div class="table-wrap"><table class="board">
  <thead><tr>
    <th class="rank">#</th><th class="model">Model</th>
    <th class="overall">{primary_label} <span class="cih">95% CI</span></th>
    <th class="num" title="Embedding cosine similarity">Cosine</th>
    <th class="num" title="Judge groundedness — higher = less hallucination">Halluc-safe</th>
    <th class="num" title="Valid-JSON rate">Format</th>
    <th class="num" title="Share answered 'I don't know'">Refuse</th>
    <th class="num" title="Mean cost per case (USD)">$/case</th>
    <th class="cases">Cases</th>
  </tr></thead>
  <tbody>{''.join(body)}</tbody></table></div>"""


def build_legend(sections):
    return "".join(f'<span><i style="background:{SECTION_COLOR[s]}"></i>{s.title()}</span>'
                   for s in sections)


# ---------------------------------------------------------------- page shell
CSS = """
:root{
  --paper:#e6e5df;--card:#f3f2ec;--ink:#1a1c1f;--soft:#54585e;--faint:#83868b;
  --rule:#cdcbc2;--claret:#7b1e2b;--claret2:#a2404b;
  --serif:"Charter","Iowan Old Style","Palatino Linotype",Palatino,Georgia,"Times New Roman",serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.55;-webkit-font-smoothing:antialiased;
  background-image:radial-gradient(#00000006 1px,transparent 1px);background-size:4px 4px}
.wrap{max-width:1000px;margin:0 auto;padding:0 24px}
a{color:var(--claret);text-underline-offset:2px}
a:hover{color:var(--claret2)}
.mast{padding:56px 0 26px;border-bottom:2px solid var(--ink)}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.22em;text-transform:uppercase;color:var(--soft);margin:0 0 14px}
.wordmark{font-family:var(--serif);font-weight:600;font-size:clamp(46px,9vw,84px);line-height:.92;letter-spacing:-.015em;margin:0;text-wrap:balance}
.wordmark .lex{color:var(--ink)}.wordmark .bench{color:var(--claret)}
.thesis{font-family:var(--serif);font-size:clamp(19px,2.6vw,25px);color:#33363b;max-width:36ch;margin:16px 0 0;text-wrap:balance}
.cite{font-family:var(--mono);font-size:12.5px;color:var(--soft);margin:22px 0 0;padding-top:12px;border-top:1px solid var(--rule);display:flex;flex-wrap:wrap;gap:6px 18px}
.cite b{color:var(--ink);font-weight:600}
.leader{display:flex;align-items:baseline;gap:14px;margin:26px 0 0;flex-wrap:wrap}
.leader .lab{font-family:var(--mono);font-size:11px;letter-spacing:.2em;text-transform:uppercase;color:var(--soft)}
.leader .who{font-family:var(--serif);font-size:22px;font-weight:600}
.leader .sc{font-family:var(--mono);font-weight:600}
section{padding:44px 0;border-bottom:1px solid var(--rule)}
.h2{font-family:var(--serif);font-size:13px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:var(--claret);margin:0 0 4px}
.trackname{font-family:var(--serif);font-size:23px;font-weight:600;margin:0 0 2px}
.lede{color:var(--soft);font-size:14.5px;margin:0 0 22px;max-width:66ch}
.table-wrap{overflow-x:auto;border:1px solid var(--rule);background:var(--card)}
table.board{border-collapse:collapse;width:100%;min-width:820px;font-variant-numeric:tabular-nums}
.board th,.board td{padding:11px 12px;text-align:left;border-bottom:1px solid var(--rule);vertical-align:middle}
.board thead th{font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--soft);font-weight:600;background:#eceae3;border-bottom:1.5px solid var(--ink);white-space:nowrap}
.board thead th .cih{display:block;font-size:9px;letter-spacing:.04em;color:var(--faint)}
.board tbody tr:last-child td{border-bottom:none}
.board tbody tr:hover{background:#ffffff66}
.board .rank{font-family:var(--serif);font-size:22px;font-weight:600;color:var(--claret);width:38px;text-align:center}
.board td.model{min-width:170px}
.mname{font-family:var(--serif);font-size:17px;font-weight:600;display:block;line-height:1.2}
.dev{font-family:var(--mono);font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--faint)}
.tie{color:var(--soft);font-family:var(--mono);margin-left:6px;font-size:13px}
.board th.overall,.board td.overall{min-width:172px}
.ov-num{font-family:var(--mono);font-size:16px;font-weight:600;display:inline-block;width:50px}
.bar{display:inline-block;vertical-align:middle;width:60px;height:7px;background:#00000012;border-radius:1px;overflow:hidden}
.fill{display:block;height:100%;border-radius:1px;animation:grow .9s cubic-bezier(.2,.7,.2,1) both;animation-delay:.2s}
.ci{font-family:var(--mono);font-size:10.5px;color:var(--soft);margin-left:7px}
.board td.num{font-family:var(--mono);font-size:13px;text-align:right;color:#2c2f33}
.cases{font-family:var(--mono);font-size:13px;color:var(--soft);white-space:normal;min-width:70px}
.tag{display:inline-block;font-family:var(--mono);font-size:10px;letter-spacing:.05em;margin-top:5px;padding:1px 6px;border:1px solid var(--claret);color:var(--claret);border-radius:2px}
.chart{margin-top:26px}
.chart svg{width:100%;height:auto;display:block}
.chart .grid{stroke:var(--rule);stroke-width:1}
.chart .ytick{font-family:var(--mono);font-size:10px;fill:var(--soft);text-anchor:end}
.chart .xtick{font-family:var(--mono);font-size:11px;fill:var(--ink);text-anchor:end}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:14px}
.legend span{font-family:var(--mono);font-size:11.5px;color:var(--soft)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:7px;vertical-align:-1px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:26px 40px;font-size:14.5px;color:#2c2f33}
.grid2 h3{font-family:var(--serif);font-size:16px;margin:0 0 6px}
.grid2 p{margin:0 0 10px}
.grid2 b{font-weight:600}
.defs{list-style:none;padding:0;margin:0;font-size:14px}
.defs li{padding:9px 0;border-bottom:1px solid var(--rule)}
.defs li:last-child{border:none}
.defs .k{font-family:var(--mono);font-size:12px;color:var(--claret);letter-spacing:.03em}
.dl{display:grid;grid-template-columns:180px 1fr;gap:4px 18px;font-size:13.5px;margin:0}
.dl dt{font-family:var(--mono);font-size:12px;color:var(--soft)}
.dl dd{margin:0;color:#2c2f33}
.note{font-family:var(--mono);font-size:12px;color:var(--soft);border-left:2px solid var(--claret);padding:8px 0 8px 14px;margin:18px 0 0;background:#00000005}
footer{padding:34px 0 60px;color:var(--soft);font-family:var(--mono);font-size:12px}
footer .wrap{display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px}
@media(max-width:640px){.grid2{grid-template-columns:1fr}.dl{grid-template-columns:1fr}.mast{padding:40px 0 22px}}
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
@keyframes grow{from{width:0}}
.mast,section{animation:rise .6s cubic-bezier(.2,.7,.2,1) both}
@media(prefers-reduced-motion:reduce){*{animation:none!important}}
"""


def _methodology_section(has_judge):
    judge_def = (
        "A strong judge model grades each section 1–5 on <b>accuracy</b>, "
        "<b>completeness</b>, and <b>groundedness</b> (freedom from invented "
        "facts/holdings/citations), normalised to 0–1. It grades against the "
        "reference brief (closed-book) or the decision text (open-book)."
        if has_judge else
        "Not yet run on this data — add judge scores with <span class='k'>judge.py</span> "
        "to make correctness the primary metric."
    )
    return (
        '<section><div class="wrap"><h2 class="h2">Metrics</h2>'
        '<p class="lede">Each brief is measured on several independent axes, '
        'because no single number captures legal quality.</p>'
        '<ul class="defs">'
        f'<li><span class="k">judge score</span> — {judge_def}</li>'
        '<li><span class="k">cosine</span> — embedding similarity (all-mpnet-base-v2) '
        'between the brief and the reference. A cheap topical signal, kept as a '
        'secondary check; it cannot tell a fluent-but-wrong brief from a correct one.</li>'
        '<li><span class="k">halluc-safe</span> — mean judge groundedness; higher means '
        'fewer fabricated facts, parties, holdings, or citations.</li>'
        '<li><span class="k">format</span> — share of responses that returned valid JSON.</li>'
        '<li><span class="k">refuse</span> — share where the model declined ("I don\'t know"), '
        'reported separately so calibrated abstention is not confused with a wrong answer.</li>'
        '<li><span class="k">$/case</span> — mean cost per case from token usage × list price.</li>'
        '</ul>'
        '<p class="note">95% confidence intervals are percentile bootstraps over cases '
        '(fixed seed). A ≈ marks a model whose interval overlaps the one above it — that '
        'gap is not statistically distinguishable.</p>'
        '</div></section>'
    )


def _dataset_section(tracks, meta):
    rows = "".join(
        f'<dt>{html.escape(TRACK_LABELS.get(t["key"], t["key"]))}</dt>'
        f'<dd>{t["n_models"]} models · {t["n_cases"]} cases{t.get("dates","")}</dd>'
        for t in tracks
    )
    return (
        '<section><div class="wrap"><h2 class="h2">Dataset</h2>'
        '<div class="grid2">'
        '<div><h3>Sources</h3>'
        '<p><b>Closed-book briefs</b> — human-written Supreme Court of Canada case '
        'summaries scraped from public case-brief wikis; used as reference answers.</p>'
        '<p><b>Open-book &amp; temporal holdout</b> — full decision text from '
        '<a href="https://a2aj.ca/">A2AJ</a> (Access to Algorithmic Justice), an '
        'open corpus of 191k+ Canadian decisions. Holdout cases are drawn from dates '
        'that postdate current model training cutoffs.</p></div>'
        '<div><h3>Composition</h3>'
        f'<dl class="dl">{rows}</dl>'
        '<p style="margin-top:12px"><b>Licensing.</b> A2AJ methods are MIT; individual '
        'decisions retain upstream terms (often non-commercial). Full decision text is '
        'not republished here — only case name, citation, court, date, and scores.</p>'
        '</div></div></div></section>'
    )


def _repro_section(meta):
    return (
        '<section><div class="wrap"><h2 class="h2">Reproducibility &amp; contamination</h2>'
        '<div class="grid2">'
        '<div><h3>How to reproduce</h3><dl class="dl">'
        '<dt>models</dt><dd>pinned in ai_models.csv (OpenRouter ids)</dd>'
        '<dt>decoding</dt><dd>temperature 0, capped max tokens</dd>'
        f'<dt>judge</dt><dd>{html.escape(meta.get("judge_model","—"))}</dd>'
        '<dt>embedding</dt><dd>all-mpnet-base-v2</dd>'
        '<dt>prompt</dt><dd>fixed system + task prompt (see benchmark.py)</dd>'
        '<dt>code</dt><dd><a href="https://github.com/okelot/LLMBenchmarkForCCL">okelot/LLMBenchmarkForCCL</a></dd>'
        '</dl></div>'
        '<div><h3>Contamination controls</h3>'
        '<p>Closed-book uses public cases whose summaries may appear in training data, '
        'so a high score there can reflect memorisation. The <b>open-book</b> and '
        '<b>temporal holdout</b> tracks counter this: open-book supplies the text (a '
        'reading task, not recall), and the holdout uses decisions issued after model '
        'cutoffs, which cannot have been memorised.</p>'
        '<p class="note">Provider routing on OpenRouter may vary backend/quantization '
        'run to run; pin a provider for byte-exact reproducibility.</p>'
        '</div></div></div></section>'
    )


def render(tracks, meta):
    lead_track = next((t for t in tracks if t["agg"]["rows"]), None)
    leader = lead_track["agg"]["rows"][0] if lead_track else None
    has_judge = any(t["agg"]["primary"] == "judge" for t in tracks)

    p = ["<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         "<title>LexBench — Frontier LLMs on Canadian case law</title>",
         '<meta name="description" content="LexBench benchmarks frontier LLMs on Canadian case-law briefing with an LLM-judge rubric, contamination-resistant open-book and temporal-holdout tracks, and confidence intervals.">',
         "<style>" + CSS + "</style></head><body>"]

    # masthead
    p.append('<header class="mast"><div class="wrap">')
    p.append('<p class="eyebrow">Canadian Case-Law Benchmark</p>')
    p.append('<h1 class="wordmark"><span class="lex">Lex</span><span class="bench">Bench</span></h1>')
    p.append('<p class="thesis">Frontier language models, judged on how faithfully they brief Canadian case law.</p>')
    if leader and leader["primary_overall"] is not None:
        p.append(f'<div class="leader"><span class="lab">Leader · {html.escape(lead_track["label"])}</span>'
                 f'<span class="who">{html.escape(leader["model"])}</span>'
                 f'<span class="sc" style="color:{_rgb(_score_rgb(leader["primary_overall"]))}">'
                 f'{leader["primary_overall"]:.3f}</span></div>')
    p.append('<p class="cite">'
             f'<span>Updated <b>{meta["updated"]}</b></span>'
             f'<span><b>{meta["tracks"]}</b> track(s)</span>'
             f'<span><b>{meta["models"]}</b> models</span>'
             f'<span>primary metric: <b>{"LLM-judge" if has_judge else "cosine"}</b></span>'
             f'<span>with <b>95% CIs</b></span></p>')
    p.append("</div></header>")

    # per-track leaderboards
    for t in tracks:
        agg = t["agg"]
        if not agg["rows"]:
            continue
        p.append('<section><div class="wrap">')
        p.append('<h2 class="h2">Leaderboard</h2>')
        p.append(f'<p class="trackname">{html.escape(t["label"])}</p>')
        p.append(f'<p class="lede">{t["desc"]}</p>')
        p.append(build_leaderboard(agg))
        p.append('<div class="chart">' + build_chart_svg(agg["rows"], agg["sections"], agg["primary"]) + '</div>')
        p.append('<div class="legend">' + build_legend(agg["sections"]) + '</div>')
        p.append("</div></section>")

    p.append(_methodology_section(has_judge))
    p.append(_dataset_section(tracks, meta))
    p.append(_repro_section(meta))

    p.append('<footer><div class="wrap">')
    p.append(f'<span>LexBench · generated {meta["generated"]}</span>')
    p.append('<span><a href="https://github.com/okelot/LLMBenchmarkForCCL">github.com/okelot/LLMBenchmarkForCCL</a></span>')
    p.append("</div></footer></body></html>")
    return "".join(p)


# ---------------------------------------------------------------- entrypoint
def _track_from_df(df, key):
    agg = aggregate(df)
    dates = ""
    if "Citation" in df.columns and df["Citation"].notna().any() and key == "open":
        pass
    n_cases = int(df["Case_Name"].nunique()) if "Case_Name" in df.columns else len(df)
    desc = {
        "closed": "The model receives only the case name and must recall the brief "
                  "from parametric knowledge — a knowledge test, sensitive to training "
                  "contamination.",
        "open": "The model is given the full decision text and must brief it — a "
                "reading/extraction task that is contamination-resistant.",
    }.get(key, "Model briefs scored against references.")
    return {"key": key, "label": TRACK_LABELS.get(key, key.title()), "desc": desc,
            "agg": agg, "n_models": len(agg["rows"]), "n_cases": n_cases, "dates": dates}


def generate(input_file: str = DEFAULT_INPUT, output_file: str = DEFAULT_OUTPUT,
             open_file: str = None) -> str:
    """Build the site. If the primary file has a `mode` column it is split into
    tracks; `open_file` optionally supplies a separate open-book results CSV."""
    frames = []
    df = pd.read_csv(input_file)
    if "mode" in df.columns and df["mode"].nunique() > 1:
        for key, sub in df.groupby("mode"):
            frames.append((key, sub.reset_index(drop=True)))
    else:
        key = df["mode"].iloc[0] if "mode" in df.columns and len(df) else "closed"
        frames.append((str(key), df))
    if open_file and Path(open_file).exists():
        frames.append(("open", pd.read_csv(open_file)))

    tracks = [_track_from_df(f, k) for k, f in frames]

    judge_model = "—"
    for _, f in frames:
        if "judge_model" in f.columns and f["judge_model"].notna().any():
            judge_model = str(f["judge_model"].dropna().iloc[0])
            break

    meta = {
        "updated": datetime.fromtimestamp(Path(input_file).stat().st_mtime).strftime("%d %b %Y"),
        "generated": datetime.now().strftime("%d %b %Y, %H:%M"),
        "models": max((t["n_models"] for t in tracks), default=0),
        "tracks": len(tracks),
        "judge_model": judge_model,
    }

    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(tracks, meta), encoding="utf-8")
    (out.parent / ".nojekyll").write_text("", encoding="utf-8")
    data = {"meta": meta, "tracks": [{"key": t["key"], "label": t["label"],
            "leaderboard": t["agg"]["rows"]} for t in tracks]}
    (out.parent / "data.json").write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"LexBench site written to {output_file}")
    return output_file


def main(argv):
    input_file = argv[0] if len(argv) > 0 else DEFAULT_INPUT
    output_file = argv[1] if len(argv) > 1 else DEFAULT_OUTPUT
    return generate(input_file, output_file)


if __name__ == "__main__":
    main(sys.argv[1:])
