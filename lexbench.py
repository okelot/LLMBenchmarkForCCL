"""Generate the LexBench landing page from evaluated benchmark results.

    python lexbench.py [evaluated_csv] [output_html]

Defaults: reads results/evaluated_case_model_results_with_section_similarity.csv
and writes docs/index.html (plus docs/data.json and docs/.nojekyll). Rerun after
every benchmark to refresh the published site.

The page is a self-contained static file (no external assets), designed to be
served straight from GitHub Pages (Settings -> Pages -> Branch: main /docs).
"""

import html
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = "results/evaluated_case_model_results_with_section_similarity.csv"
DEFAULT_OUTPUT = "docs/index.html"

SECTIONS = ["facts", "issue", "decision", "reasons", "ratio"]

# Muted, earthy section hues that sit on stone paper without fighting the claret
# accent. Kept distinct from the score scale and from the accent.
SECTION_COLOR = {
    "facts": "#4a7c59",
    "issue": "#4e6d8c",
    "decision": "#a9762f",
    "reasons": "#7d6b9e",
    "ratio": "#8c5a4a",
}


# ---------------------------------------------------------------- aggregation
def _developer_map(models_csv: str = "ai_models.csv") -> dict:
    try:
        m = pd.read_csv(models_csv)
        return dict(zip(m["display_name"], m["developer"]))
    except Exception:
        return {}


def aggregate(df: pd.DataFrame):
    """Return (rows sorted by overall desc, list of present sections)."""
    dev = _developer_map()
    present = [s for s in SECTIONS if f"{s}_similarity" in df.columns]
    ai_cols = [f"ai_{s}" for s in SECTIONS if f"ai_{s}" in df.columns]

    rows = []
    for model, sub in df.groupby("Model_ID"):
        sec = {}
        for s in present:
            val = sub[f"{s}_similarity"].mean()
            sec[s] = None if pd.isna(val) else round(float(val), 4)
        vals = [v for v in sec.values() if v is not None]
        overall = round(sum(vals) / len(vals), 4) if vals else None
        errors = int((sub[ai_cols] == "ERROR").all(axis=1).sum()) if ai_cols else 0
        rows.append(
            {
                "model": str(model),
                "developer": str(dev.get(model, "")),
                "n": int(len(sub)),
                "errors": errors,
                "sections": sec,
                "overall": overall,
            }
        )
    rows.sort(key=lambda r: (r["overall"] is not None, r["overall"] or 0), reverse=True)
    return rows, present


# ---------------------------------------------------------------- score color
def _lerp(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


# Earthy sequential scale: clay (low) -> ochre (mid) -> sage (high).
_ANCHORS = [(0.30, (176, 137, 104)), (0.55, (201, 162, 39)), (0.80, (63, 111, 95))]


def _score_rgb(v):
    if v is None:
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


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


# ---------------------------------------------------------------- svg chart
def build_chart_svg(rows, sections):
    if not rows or not sections:
        return ""
    W, H = 860, 430
    ml, mr, mt, mb = 46, 18, 22, 104
    pw, ph = W - ml - mr, H - mt - mb
    n = len(rows)
    gw = pw / n
    inner = gw * 0.74
    bw = inner / len(sections)

    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Average similarity by section and model" preserveAspectRatio="xMidYMid meet">']

    # gridlines + y axis labels
    for t in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = mt + ph * (1 - t)
        p.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" class="grid"/>')
        p.append(f'<text x="{ml-8}" y="{y+3.5:.1f}" class="ytick">{t:.1f}</text>')

    # bars
    for gi, r in enumerate(rows):
        gx = ml + gw * gi + (gw - inner) / 2
        for si, s in enumerate(sections):
            v = r["sections"].get(s) or 0
            bh = ph * v
            bx = gx + si * bw
            by = mt + ph - bh
            p.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{max(bw-2,1):.1f}" height="{bh:.1f}" '
                f'fill="{SECTION_COLOR[s]}"><title>{html.escape(r["model"])} · {s}: {v:.3f}</title></rect>'
            )
        # model label, rotated
        cx = ml + gw * gi + gw / 2
        ly = mt + ph + 14
        label = html.escape(r["model"])
        p.append(
            f'<text x="{cx:.1f}" y="{ly:.1f}" class="xtick" transform="rotate(-22 {cx:.1f} {ly:.1f})">{label}</text>'
        )

    p.append("</svg>")
    return "".join(p)


# ---------------------------------------------------------------- leaderboard
def build_leaderboard(rows, sections):
    max_n = max((r["n"] for r in rows), default=0)
    head_sections = "".join(
        f'<th class="sec"><span class="dot" style="background:{SECTION_COLOR[s]}"></span>{s.title()}</th>'
        for s in sections
    )
    body = []
    for i, r in enumerate(rows, 1):
        ov = r["overall"]
        ov_rgb = _score_rgb(ov)
        pct = 0 if ov is None else max(0, min(100, ov * 100))
        partial = (
            f'<span class="tag">partial · {r["n"]}/{max_n}</span>'
            if r["n"] < max_n
            else ""
        )
        err = f'<span class="tag err">{r["errors"]} err</span>' if r["errors"] else ""
        dev = html.escape(r["developer"]) if r["developer"] else ""

        sec_cells = ""
        for s in sections:
            v = r["sections"].get(s)
            c = _score_rgb(v)
            sec_cells += (
                f'<td class="num" style="background:{_rgba(c,0.16)}">{_fmt(v)}</td>'
            )

        body.append(
            f"""<tr>
  <td class="rank">{i}</td>
  <td class="model"><span class="mname">{html.escape(r["model"])}</span>
      <span class="dev">{dev}</span></td>
  <td class="overall">
      <span class="ov-num" style="color:{_rgb(ov_rgb)}">{_fmt(ov)}</span>
      <span class="bar"><span class="fill" style="width:{pct:.1f}%;--w:{pct:.1f}%;background:{_rgb(ov_rgb)}"></span></span>
  </td>
  {sec_cells}
  <td class="cases">{r["n"]}{partial}{err}</td>
</tr>"""
        )

    return f"""<div class="table-wrap">
<table class="board">
  <thead><tr>
    <th class="rank">#</th><th class="model">Model</th>
    <th class="overall">Overall</th>{head_sections}<th class="cases">Cases</th>
  </tr></thead>
  <tbody>{''.join(body)}</tbody>
</table></div>"""


# ---------------------------------------------------------------- page shell
CSS = """
:root{
  --paper:#e6e5df; --card:#f3f2ec; --ink:#1a1c1f; --soft:#54585e; --faint:#83868b;
  --rule:#cdcbc2; --claret:#7b1e2b; --claret2:#a2404b;
  --serif:"Charter","Iowan Old Style","Palatino Linotype",Palatino,Georgia,"Times New Roman",serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.55;-webkit-font-smoothing:antialiased;
  background-image:radial-gradient(#00000006 1px,transparent 1px);background-size:4px 4px;}
.wrap{max-width:980px;margin:0 auto;padding:0 24px}
a{color:var(--claret);text-underline-offset:2px;text-decoration-thickness:.06em}
a:hover{color:var(--claret2)}

/* masthead */
.mast{padding:56px 0 26px;border-bottom:2px solid var(--ink)}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.22em;text-transform:uppercase;
  color:var(--soft);margin:0 0 14px}
.wordmark{font-family:var(--serif);font-weight:600;font-size:clamp(46px,9vw,84px);line-height:.92;
  letter-spacing:-.015em;margin:0;text-wrap:balance}
.wordmark .lex{color:var(--ink)} .wordmark .bench{color:var(--claret)}
.thesis{font-family:var(--serif);font-size:clamp(19px,2.6vw,25px);color:#33363b;
  max-width:34ch;margin:16px 0 0;text-wrap:balance}
.cite{font-family:var(--mono);font-size:12.5px;color:var(--soft);margin:22px 0 0;
  padding-top:12px;border-top:1px solid var(--rule);display:flex;flex-wrap:wrap;gap:6px 18px}
.cite b{color:var(--ink);font-weight:600}

/* leader callout */
.leader{display:flex;align-items:baseline;gap:14px;margin:26px 0 0;flex-wrap:wrap}
.leader .lab{font-family:var(--mono);font-size:11px;letter-spacing:.2em;text-transform:uppercase;color:var(--soft)}
.leader .who{font-family:var(--serif);font-size:22px;font-weight:600}
.leader .sc{font-family:var(--mono);font-weight:600}

/* sections */
section{padding:44px 0;border-bottom:1px solid var(--rule)}
.h2{font-family:var(--serif);font-size:13px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;
  color:var(--claret);margin:0 0 4px}
.lede{color:var(--soft);font-size:14.5px;margin:0 0 22px;max-width:64ch}

/* board */
.table-wrap{overflow-x:auto;border:1px solid var(--rule);background:var(--card)}
table.board{border-collapse:collapse;width:100%;min-width:720px;font-variant-numeric:tabular-nums}
.board th,.board td{padding:12px 14px;text-align:left;border-bottom:1px solid var(--rule);vertical-align:middle}
.board thead th{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--soft);font-weight:600;background:#eceae3;border-bottom:1.5px solid var(--ink);white-space:nowrap}
.board tbody tr:last-child td{border-bottom:none}
.board tbody tr:hover{background:#ffffff66}
.board .rank{font-family:var(--serif);font-size:22px;font-weight:600;color:var(--claret);width:40px;text-align:center}
.board td.model{min-width:180px}
.mname{font-family:var(--serif);font-size:17px;font-weight:600;display:block;line-height:1.2}
.dev{font-family:var(--mono);font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--faint)}
.board th.overall,.board td.overall{min-width:150px}
.ov-num{font-family:var(--mono);font-size:16px;font-weight:600;display:inline-block;width:52px}
.bar{display:inline-block;vertical-align:middle;width:74px;height:7px;background:#00000012;
  border-radius:1px;overflow:hidden}
.fill{display:block;height:100%;border-radius:1px}
.board th.sec{white-space:nowrap}
.board .dot{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px;vertical-align:middle}
.board td.num{font-family:var(--mono);font-size:13.5px;text-align:right;width:74px}
.cases{font-family:var(--mono);font-size:13px;color:var(--soft);white-space:normal;min-width:76px}
.tag{display:inline-block;font-family:var(--mono);font-size:10px;letter-spacing:.05em;
  margin-top:5px;padding:1px 6px;border:1px solid var(--claret);color:var(--claret);border-radius:2px}
.tag.err{border-color:#a9762f;color:#8a5f22}

/* chart */
.chart svg{width:100%;height:auto;display:block}
.chart .grid{stroke:var(--rule);stroke-width:1}
.chart .ytick{font-family:var(--mono);font-size:10px;fill:var(--soft);text-anchor:end}
.chart .xtick{font-family:var(--mono);font-size:11px;fill:var(--ink);text-anchor:end}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:16px}
.legend span{font-family:var(--mono);font-size:11.5px;color:var(--soft);letter-spacing:.04em}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:7px;vertical-align:-1px}

/* headnote */
.headnote{columns:2;column-gap:40px;font-size:14.5px;color:#2c2f33}
.headnote p{margin:0 0 12px;break-inside:avoid}
.headnote b{font-weight:600}
.note{font-family:var(--mono);font-size:12px;color:var(--soft);border-left:2px solid var(--claret);
  padding:8px 0 8px 14px;margin:18px 0 0;background:#00000005}

footer{padding:34px 0 60px;color:var(--soft);font-family:var(--mono);font-size:12px;
  display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px}

@media(max-width:640px){
  .headnote{columns:1}
  .mast{padding:40px 0 22px}
}

/* motion */
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
@keyframes grow{from{width:0}}
.mast,section{animation:rise .6s cubic-bezier(.2,.7,.2,1) both}
section:nth-of-type(1){animation-delay:.05s}
section:nth-of-type(2){animation-delay:.12s}
section:nth-of-type(3){animation-delay:.19s}
.fill{animation:grow .9s cubic-bezier(.2,.7,.2,1) both;animation-delay:.25s}
@media(prefers-reduced-motion:reduce){*{animation:none!important}}
"""


def render(rows, sections, meta):
    leader = rows[0] if rows else None
    legend = "".join(
        f'<span><i style="background:{SECTION_COLOR[s]}"></i>{s.title()}</span>'
        for s in sections
    )
    leader_html = ""
    if leader and leader["overall"] is not None:
        leader_html = (
            f'<div class="leader"><span class="lab">Current leader</span>'
            f'<span class="who">{html.escape(leader["model"])}</span>'
            f'<span class="sc" style="color:{_rgb(_score_rgb(leader["overall"]))}">'
            f'{leader["overall"]:.3f}</span></div>'
        )

    parts = []
    parts.append("<!doctype html>")
    parts.append('<html lang="en"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width,initial-scale=1">')
    parts.append("<title>LexBench — Frontier LLMs on Canadian case law</title>")
    parts.append(
        '<meta name="description" content="LexBench benchmarks frontier language models on their ability to brief Canadian case law, scored by semantic similarity against human-written briefs.">'
    )
    parts.append("<style>" + CSS + "</style></head><body>")

    # masthead
    parts.append('<header class="mast"><div class="wrap">')
    parts.append('<p class="eyebrow">Canadian Case-Law Benchmark</p>')
    parts.append('<h1 class="wordmark"><span class="lex">Lex</span><span class="bench">Bench</span></h1>')
    parts.append('<p class="thesis">Frontier language models, judged on how faithfully they brief Canadian case law.</p>')
    parts.append(leader_html)
    parts.append(
        '<p class="cite"><span>Updated <b>{updated}</b></span>'
        '<span><b>{models}</b> models</span>'
        '<span><b>{cases}</b> cases</span>'
        '<span><b>{rows}</b> briefs scored</span>'
        '<span>metric: <b>cosine similarity</b></span></p>'.format(**meta)
    )
    parts.append("</div></header>")

    # leaderboard
    parts.append('<section><div class="wrap">')
    parts.append('<h2 class="h2">Leaderboard</h2>')
    parts.append('<p class="lede">Mean cosine similarity between each model\'s brief and the human-authored brief, per section of the case (facts, issue, decision, reasons, ratio). Ranked by the average across sections.</p>')
    parts.append(build_leaderboard(rows, sections))
    parts.append("</div></section>")

    # chart
    parts.append('<section><div class="wrap">')
    parts.append('<h2 class="h2">Section breakdown</h2>')
    parts.append('<p class="lede">Where each model is strong or weak across the five parts of a case brief.</p>')
    parts.append('<div class="chart">' + build_chart_svg(rows, sections) + "</div>")
    parts.append('<div class="legend">' + legend + "</div>")
    parts.append("</div></section>")

    # headnote / methodology
    parts.append('<section><div class="wrap">')
    parts.append('<h2 class="h2">Headnote — how it works</h2>')
    parts.append(
        '<div class="headnote">'
        "<p>Each model is given only a case <b>name</b> (and citation) and asked to produce a structured brief — facts, issue, decision, reasons, and ratio decidendi — as JSON. No case text is provided, so the task probes the model's actual knowledge of Canadian jurisprudence.</p>"
        "<p>Every model is reached through a single <b>OpenRouter</b> endpoint, so results are directly comparable. The model's brief is compared section-by-section against a human-written brief using sentence-embedding <b>cosine similarity</b> (all-mpnet-base-v2), with long passages chunked and mean-pooled.</p>"
        "<p>Human briefs are sourced from public Supreme Court of Canada case summaries. A score near 1.0 means the model's wording is semantically close to the human brief; lower scores flag missing, wrong, or invented content.</p>"
        "</div>"
    )
    parts.append(
        '<p class="note">Similarity measures semantic closeness, not legal correctness — a fluent but wrong brief can still score moderately. Read it as a comparative signal across models, not a verdict on any single answer.</p>'
    )
    parts.append("</div></section>")

    # footer
    parts.append('<footer><div class="wrap" style="display:flex;justify-content:space-between;width:100%;flex-wrap:wrap;gap:10px">')
    parts.append(f'<span>LexBench · generated {meta["generated"]}</span>')
    parts.append('<span><a href="https://github.com/okelot/LLMBenchmarkForCCL">github.com/okelot/LLMBenchmarkForCCL</a></span>')
    parts.append("</div></footer>")

    parts.append("</body></html>")
    return "".join(parts)


# ---------------------------------------------------------------- entrypoint
def generate(input_file: str = DEFAULT_INPUT, output_file: str = DEFAULT_OUTPUT) -> str:
    df = pd.read_csv(input_file)
    rows, sections = aggregate(df)

    updated = datetime.fromtimestamp(Path(input_file).stat().st_mtime).strftime("%d %b %Y")
    meta = {
        "updated": updated,
        "generated": datetime.now().strftime("%d %b %Y, %H:%M"),
        "models": len(rows),
        "cases": int(df["Case_Name"].nunique()) if "Case_Name" in df.columns else "—",
        "rows": len(df),
    }

    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rows, sections, meta), encoding="utf-8")
    (out.parent / ".nojekyll").write_text("", encoding="utf-8")
    (out.parent / "data.json").write_text(
        json.dumps({"meta": meta, "leaderboard": rows}, indent=2), encoding="utf-8"
    )
    print(f"LexBench site written to {output_file}")
    return output_file


def main(argv):
    input_file = argv[0] if len(argv) > 0 else DEFAULT_INPUT
    output_file = argv[1] if len(argv) > 1 else DEFAULT_OUTPUT
    return generate(input_file, output_file)


if __name__ == "__main__":
    main(sys.argv[1:])
