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
def _repair_missing_references(df: pd.DataFrame) -> pd.DataFrame:
    """Mask scores for sections that have no human reference (reference mode).

    Older judge runs scored every section even when the reference was missing
    (e.g. Innisfil's empty Ratio) — grading a candidate against nothing. This
    masks those scores and recomputes judge_overall from the remaining sections,
    so the fix applies retroactively to existing result files. Open-book rows
    are untouched: they are graded against the decision text, not references.
    """
    df = df.copy()
    is_ref = (df["mode"] == "closed") if "mode" in df.columns else pd.Series(True, index=df.index)
    judge_cols_touched = False
    for s in SECTIONS:
        href = f"human_{s}"
        if href not in df.columns:
            continue
        missing = is_ref & (df[href].isna() | (df[href].astype(str).str.strip()
                                               .isin(["", "nan"])))
        if not missing.any():
            continue
        for col in ([f"judge_{s}"] + [f"judge_{s}_{c}" for c in
                                      ["accuracy", "completeness", "groundedness"]]
                    + [f"{s}_similarity"]):
            if col in df.columns:
                df.loc[missing, col] = pd.NA
                judge_cols_touched = True
    if judge_cols_touched and "judge_overall" in df.columns:
        jcols = [f"judge_{s}" for s in SECTIONS if f"judge_{s}" in df.columns]
        recomputed = df[jcols].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        df.loc[is_ref, "judge_overall"] = recomputed[is_ref].round(4)
    return df


def aggregate(df: pd.DataFrame):
    dev = _developer_map()
    df = _repair_missing_references(df)
    present = [s for s in SECTIONS if f"{s}_similarity" in df.columns or f"judge_{s}" in df.columns
               or f"rubric_{s}" in df.columns]
    has_judge = any(f"judge_{s}" in df.columns for s in SECTIONS) or "judge_overall" in df.columns
    has_rubric = "rubric_overall" in df.columns and df["rubric_overall"].notna().any()
    primary = "rubric" if has_rubric else ("judge" if has_judge else "cosine")

    sim_cols = [f"{s}_similarity" for s in SECTIONS if f"{s}_similarity" in df.columns]
    judge_cols = [f"judge_{s}" for s in SECTIONS if f"judge_{s}" in df.columns]
    ground_cols = [f"judge_{s}_groundedness" for s in SECTIONS
                   if f"judge_{s}_groundedness" in df.columns]
    ai_cols = [f"ai_{s}" for s in SECTIONS if f"ai_{s}" in df.columns]
    has_format = "format_ok" in df.columns

    rows = []
    for model, sub in df.groupby("Model_ID"):
        # Quality is measured over VALID ATTEMPTED responses only; execution
        # failures (invalid JSON, truncation, exceptions) and refusals are
        # reported as their own metrics rather than entering the quality mean
        # as zeros. Refusals are excluded whether they arrive as plain text or
        # wrapped in valid JSON — same behaviour, same treatment. A strict
        # all-rows mean is kept alongside for anyone who prefers that policy.
        if has_format:
            mask = sub["format_ok"] == 1
            if "refused" in sub.columns:
                mask &= sub["refused"] != 1
            valid = sub[mask]
        else:
            valid = sub
        scored = valid if len(valid) else sub

        cosine = {s: _col_mean(scored, f"{s}_similarity") for s in SECTIONS}
        judge = {s: _col_mean(scored, f"judge_{s}") for s in SECTIONS}
        rubric = {s: _col_mean(scored, f"rubric_{s}") for s in SECTIONS}
        cosine_overall = (
            float(pd.Series([v for v in cosine.values() if v is not None]).mean())
            if any(v is not None for v in cosine.values()) else None)
        judge_overall = (float(pd.Series([v for v in judge.values() if v is not None]).mean())
                         if any(v is not None for v in judge.values()) else None)
        rubric_overall = _col_mean(scored, "rubric_overall")

        # primary per-row values (valid rows) -> bootstrap CI; keep per-case
        # scores so ranking gaps can be tested with a PAIRED bootstrap.
        overall_col = {"rubric": "rubric_overall", "judge": "judge_overall"}.get(primary)
        if overall_col and overall_col in scored:
            src = scored[overall_col]
            per_row = src.dropna().tolist()
            per_case = (dict(zip(scored["Case_Name"], pd.to_numeric(src, errors="coerce")))
                        if "Case_Name" in scored else {})
        else:
            per_row = _row_overall(scored, sim_cols)
            per_case = {}
        mean, lo, hi = stats.bootstrap_ci(per_row)

        # strict mean over ALL rows (failures counted as scored, usually 0)
        strict_col = overall_col or "judge_overall"
        strict_src = sub[strict_col] if strict_col in sub else None
        strict = (round(float(pd.to_numeric(strict_src, errors="coerce").mean()), 4)
                  if strict_src is not None and strict_src.notna().any() else None)

        trunc_rate = (float((sub["finish_reason"] == "length").mean())
                      if "finish_reason" in sub.columns else None)

        rows.append({
            "model": str(model),
            "developer": str(dev.get(model, "")),
            "n": int(len(sub)),
            "n_valid": int(len(valid)) if has_format else int(len(sub)),
            "cosine": cosine, "judge": judge, "rubric": rubric,
            "cosine_overall": None if cosine_overall is None else round(cosine_overall, 4),
            "judge_overall": None if judge_overall is None else round(judge_overall, 4),
            "rubric_overall": None if rubric_overall is None else round(rubric_overall, 4),
            "primary_overall": mean, "ci_lo": lo, "ci_hi": hi,
            "strict_overall": strict,
            "_per_case": per_case,
            "groundedness": (round(float(scored[ground_cols].mean(axis=1).mean()), 4)
                             if ground_cols and len(scored) else None),
            "format_rate": _col_mean(sub, "format_ok"),
            "refusal_rate": _col_mean(sub, "refused"),
            "truncation_rate": trunc_rate,
            "error_rate": (int((sub[ai_cols] == "ERROR").all(axis=1).sum()) / len(sub)
                           if ai_cols else None),
            "avg_cost": _col_mean(sub, "cost_usd"),
            "avg_latency": _col_mean(sub, "latency_s"),
        })

    rows.sort(key=lambda r: (r["primary_overall"] is not None, r["primary_overall"] or 0),
              reverse=True)

    # Tie flags via PAIRED bootstrap on shared cases (every model sees the same
    # cases, so the paired design is the correct comparison). Falls back to
    # marginal CI overlap when per-case scores are unavailable.
    for i in range(1, len(rows)):
        a, b = rows[i], rows[i - 1]
        pa, pb = a.get("_per_case") or {}, b.get("_per_case") or {}
        shared = [c for c in pa if c in pb]
        if shared:
            res = stats.paired_bootstrap_diff([(pb[c], pa[c]) for c in shared])
            a["tied_above"] = not res["significant"]
            a["gap_above"] = res
        else:
            a["tied_above"] = (a["ci_hi"] is not None and b["ci_lo"] is not None
                               and a["ci_hi"] >= b["ci_lo"])
            a["gap_above"] = None
    if rows:
        rows[0]["tied_above"] = False
        rows[0]["gap_above"] = None
    for r in rows:
        r.pop("_per_case", None)

    labels = {"rubric": "Rubric score", "judge": "Judge score", "cosine": "Cosine similarity"}
    return {"rows": rows, "sections": present, "primary": primary,
            "primary_label": labels[primary]}


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
def _dv(v):
    """data-v attribute for client-side sorting ('' when missing)."""
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{float(v):.6f}"


def build_leaderboard(agg, track_label=""):
    rows, primary_label = agg["rows"], agg["primary_label"]
    show_judge = agg["primary"] == "rubric"
    max_n = max((r["n"] for r in rows), default=0)
    body = []
    for i, r in enumerate(rows, 1):
        ov = r["primary_overall"]
        c = _score_rgb(ov)
        pct = 0 if ov is None else max(0, min(100, ov * 100))
        has_ci = r["ci_lo"] is not None and r["ci_hi"] is not None and r["n"] > 1
        ci = f'<span class="ci">[{r["ci_lo"]:.2f}–{r["ci_hi"]:.2f}]</span>' if has_ci else ""
        whisker, bar_title = "", ""
        if has_ci:
            lo = max(0.0, min(1.0, r["ci_lo"])) * 100
            hi = max(0.0, min(1.0, r["ci_hi"])) * 100
            whisker = f'<span class="wh" style="left:{lo:.1f}%;width:{max(hi-lo,1):.1f}%"></span>'
            bar_title = f' title="95% CI {r["ci_lo"]:.2f}–{r["ci_hi"]:.2f}"'
        gap = r.get("gap_above")
        tie_title = (f"Paired bootstrap vs model above: diff {gap['diff']:+.3f}, "
                     f"95% CI [{gap['lo']:+.3f}, {gap['hi']:+.3f}] over {gap['n']} shared cases — "
                     "not significant" if gap else
                     "CI overlaps the model above — gap not significant")
        tie = f'<span class="tie" title="{tie_title}">≈</span>' if r.get("tied_above") else ""
        partial = f'<span class="tag">partial · {r["n"]}/{max_n}</span>' if r["n"] < max_n else ""
        lat = f'{r["avg_latency"]:.1f}s' if r["avg_latency"] is not None else "—"
        n_valid = r.get("n_valid", r["n"])
        cases_txt = f'{n_valid}/{r["n"]}' if n_valid != r["n"] else f'{r["n"]}'
        judge_cell = (f'<td class="num" data-v="{_dv(r.get("judge_overall"))}">'
                      f'{_fmt(r.get("judge_overall"), 2)}</td>' if show_judge else "")
        body.append(f"""<tr>
  <td class="rank" data-v="{i}">{i}</td>
  <td class="model" data-v="{html.escape(r["model"])}"><span class="mname">{html.escape(r["model"])}{tie}</span>
      <span class="dev">{html.escape(r["developer"])}</span></td>
  <td class="overall" data-v="{_dv(ov)}"><span class="ov-num" style="color:{_rgb(c)}">{_fmt(ov)}</span>
      <span class="bar"{bar_title}><span class="fill" style="width:{pct:.1f}%;background:{_rgb(c)}"></span>{whisker}</span>
      {ci}</td>
  {judge_cell}
  <td class="num" data-v="{_dv(r["cosine_overall"])}">{_fmt(r["cosine_overall"], 2)}</td>
  <td class="num" data-v="{_dv(r["groundedness"])}">{_fmt(r["groundedness"], 2)}</td>
  <td class="num" data-v="{_dv(r["format_rate"])}">{_pct(r["format_rate"])}</td>
  <td class="num" data-v="{_dv(r.get("truncation_rate"))}">{_pct(r.get("truncation_rate"))}</td>
  <td class="num" data-v="{_dv(r["refusal_rate"])}">{_pct(r["refusal_rate"])}</td>
  <td class="num" data-v="{_dv(r["avg_cost"])}">{('$'+format(r["avg_cost"],'.4f')) if r["avg_cost"] is not None else '—'}</td>
  <td class="num" data-v="{_dv(r["avg_latency"])}">{lat}</td>
  <td class="cases" data-v="{n_valid}">{cases_txt}{partial}</td>
</tr>""")
    caption = f'<caption class="sr-only">Leaderboard — {html.escape(track_label)}</caption>' if track_label else ""
    return f"""<div class="table-wrap"><table class="board">{caption}
  <thead><tr>
    <th class="rank" scope="col" data-sort="num" title="Rank by primary metric">#</th>
    <th class="model" scope="col" data-sort="text">Model</th>
    <th class="overall" scope="col" data-sort="num" title="Primary metric over VALID responses — execution failures are reported separately, not folded in as zeros">{primary_label} <span class="cih">valid · 95% CI</span></th>
    {'<th class="num" scope="col" data-sort="num" title="Holistic LLM-judge score (secondary)">Judge</th>' if show_judge else ''}
    <th class="num" scope="col" data-sort="num" title="Embedding cosine similarity">Cosine</th>
    <th class="num" scope="col" data-sort="num" title="Judge groundedness — higher = fewer hallucinated facts, holdings, citations">Grounded</th>
    <th class="num" scope="col" data-sort="num" title="Valid-JSON rate">Format</th>
    <th class="num" scope="col" data-sort="num" title="Share of responses cut off by the token cap (finish_reason=length)">Trunc</th>
    <th class="num" scope="col" data-sort="num" title="Share answered 'I don't know'">Refuse</th>
    <th class="num" scope="col" data-sort="num" title="Mean cost per case (USD)">$/case</th>
    <th class="num" scope="col" data-sort="num" title="Mean wall-clock seconds per case">Latency</th>
    <th class="cases" scope="col" data-sort="num" title="Valid responses / cases attempted">Cases</th>
  </tr></thead>
  <tbody>{''.join(body)}</tbody></table></div>
<p class="hint">Click a column header to sort · hover headers for metric definitions.</p>"""


def build_legend(sections):
    return "".join(f'<span><i style="background:{SECTION_COLOR[s]}"></i>{s.title()}</span>'
                   for s in sections)


# ---------------------------------------------------------------- page shell
# One block of colour variables per scheme; the dark block is applied both by
# explicit user choice ([data-theme=dark]) and by OS preference (media query).
_DARK_VARS = """
  --paper:#181a1c;--card:#212328;--ink:#e9e7e2;--ink2:#cfcdc7;--soft:#a3a7ae;--faint:#83868c;
  --rule:#3a3d43;--claret:#d1737e;--claret2:#de949c;
  --rowhov:#ffffff0d;--bartrack:#ffffff1c;--notebg:#ffffff07;--dot:#ffffff04;
  --navbg:#181a1ce8;--edge:#00000073;--boost:1.45;--cboost:1.18;
"""

CSS = """
:root{
  --paper:#e6e5df;--card:#f3f2ec;--ink:#1a1c1f;--ink2:#2c2f33;--soft:#54585e;--faint:#83868b;
  --rule:#cdcbc2;--claret:#7b1e2b;--claret2:#a2404b;
  --rowhov:#ffffff66;--bartrack:#00000012;--notebg:#00000005;--dot:#00000006;
  --navbg:#e6e5dfe8;--edge:#00000030;--boost:1;--cboost:1;
  --serif:"Charter","Iowan Old Style","Palatino Linotype",Palatino,Georgia,"Times New Roman",serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
}
[data-theme=dark]{@@DARK@@}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){@@DARK@@}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.55;-webkit-font-smoothing:antialiased;
  background-image:radial-gradient(var(--dot) 1px,transparent 1px);background-size:4px 4px}
.wrap{max-width:1000px;margin:0 auto;padding:0 24px}
a{color:var(--claret);text-underline-offset:2px}
a:hover{color:var(--claret2)}
:focus-visible{outline:2px solid var(--claret);outline-offset:2px}
.skip{position:absolute;left:-9999px;font-family:var(--mono);font-size:12px}
.skip:focus{left:12px;top:60px;z-index:99;background:var(--card);color:var(--ink);padding:8px 12px;border:1px solid var(--rule)}
.topnav{position:sticky;top:0;z-index:10;background:var(--navbg);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);border-bottom:1px solid var(--rule)}
.topnav .wrap{display:flex;align-items:center;gap:20px;height:46px}
.topnav .brand{font-family:var(--serif);font-weight:600;font-size:16px;color:var(--ink);text-decoration:none;white-space:nowrap}
.topnav .brand b{color:var(--claret);font-weight:600}
.topnav .links{display:flex;gap:18px;overflow-x:auto;flex:1;scrollbar-width:none;-webkit-overflow-scrolling:touch}
.topnav .links::-webkit-scrollbar{display:none}
.topnav .links a{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--soft);text-decoration:none;white-space:nowrap;padding:4px 0}
.topnav .links a:hover{color:var(--claret)}
.tbtn{background:none;border:1px solid var(--rule);color:var(--soft);font-size:13px;line-height:1;padding:5px 8px;border-radius:3px;cursor:pointer;font-family:var(--mono)}
.tbtn:hover{color:var(--claret);border-color:var(--claret)}
.mast{padding:48px 0 26px;border-bottom:2px solid var(--ink)}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.22em;text-transform:uppercase;color:var(--soft);margin:0 0 14px}
.wordmark{font-family:var(--serif);font-weight:600;font-size:clamp(46px,9vw,84px);line-height:.92;letter-spacing:-.015em;margin:0;text-wrap:balance}
.wordmark .lex{color:var(--ink)}.wordmark .bench{color:var(--claret)}
.thesis{font-family:var(--serif);font-size:clamp(19px,2.6vw,25px);color:var(--ink2);max-width:36ch;margin:16px 0 0;text-wrap:balance}
.cite{font-family:var(--mono);font-size:12.5px;color:var(--soft);margin:22px 0 0;padding-top:12px;border-top:1px solid var(--rule);display:flex;flex-wrap:wrap;gap:6px 18px}
.cite b{color:var(--ink);font-weight:600}
.leader{display:flex;align-items:baseline;gap:14px;margin:26px 0 0;flex-wrap:wrap}
.leader .lab{font-family:var(--mono);font-size:11px;letter-spacing:.2em;text-transform:uppercase;color:var(--soft)}
.leader .who{font-family:var(--serif);font-size:22px;font-weight:600}
.leader .sc{font-family:var(--mono);font-weight:600;filter:brightness(var(--boost))}
section{padding:44px 0;border-bottom:1px solid var(--rule);scroll-margin-top:56px}
.h2{font-family:var(--serif);font-size:13px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:var(--claret);margin:0 0 4px}
.trackname{font-family:var(--serif);font-size:23px;font-weight:600;margin:0 0 2px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.nchip{font-family:var(--mono);font-size:11px;font-weight:400;letter-spacing:.05em;color:var(--soft);border:1px solid var(--rule);border-radius:3px;padding:2px 8px;white-space:nowrap}
.lede{color:var(--soft);font-size:14.5px;margin:0 0 22px;max-width:66ch}
.table-wrap{overflow-x:auto;border:1px solid var(--rule);background-color:var(--card);
  background-image:linear-gradient(90deg,var(--card) 33%,rgba(0,0,0,0)),
    linear-gradient(270deg,var(--card) 33%,rgba(0,0,0,0)),
    radial-gradient(farthest-side at 0 50%,var(--edge),rgba(0,0,0,0)),
    radial-gradient(farthest-side at 100% 50%,var(--edge),rgba(0,0,0,0));
  background-position:0 0,100% 0,0 0,100% 0;background-repeat:no-repeat;
  background-size:60px 100%,60px 100%,16px 100%,16px 100%;
  background-attachment:local,local,scroll,scroll}
table.board{border-collapse:collapse;width:100%;min-width:920px;font-variant-numeric:tabular-nums}
.board th,.board td{padding:11px 9px;text-align:left;border-bottom:1px solid var(--rule);vertical-align:middle}
.board th:first-child,.board td:first-child{padding-left:14px}
.board th:last-child,.board td:last-child{padding-right:14px}
.board thead th{font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--soft);font-weight:600;border-bottom:1.5px solid var(--ink);white-space:nowrap}
.board thead th .cih{display:block;font-size:9px;letter-spacing:.04em;color:var(--faint)}
.board thead th[data-sort]{cursor:pointer;user-select:none}
.board thead th[data-sort]:hover{color:var(--claret)}
.board thead th[data-sort]::after{content:"↕";opacity:.35;margin-left:4px;font-size:9px}
.board thead th[aria-sort=descending]::after{content:"↓";opacity:1;color:var(--claret)}
.board thead th[aria-sort=ascending]::after{content:"↑";opacity:1;color:var(--claret)}
.board tbody tr:last-child td{border-bottom:none}
.board tbody tr:hover{background:var(--rowhov)}
.board .rank{font-family:var(--serif);font-size:22px;font-weight:600;color:var(--claret);width:38px;text-align:center}
.board td.model{min-width:170px}
.mname{font-family:var(--serif);font-size:17px;font-weight:600;display:block;line-height:1.2}
.dev{font-family:var(--mono);font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--faint)}
.tie{color:var(--soft);font-family:var(--mono);margin-left:6px;font-size:13px}
.board th.overall,.board td.overall{min-width:218px;white-space:nowrap}
.ov-num{font-family:var(--mono);font-size:16px;font-weight:600;display:inline-block;width:50px;filter:brightness(var(--boost))}
.bar{position:relative;display:inline-block;vertical-align:middle;width:60px;height:7px;background:var(--bartrack);border-radius:1px}
.fill{display:block;height:100%;border-radius:1px;animation:grow .9s cubic-bezier(.2,.7,.2,1) both;animation-delay:.2s;filter:brightness(var(--boost))}
.wh{position:absolute;top:100%;margin-top:2px;height:2px;background:var(--ink);opacity:.4;display:block;border-radius:1px}
.ci{font-family:var(--mono);font-size:10.5px;color:var(--soft);margin-left:7px}
.board td.num{font-family:var(--mono);font-size:12.5px;text-align:right;color:var(--ink2)}
.cases{font-family:var(--mono);font-size:12.5px;color:var(--soft);white-space:normal;min-width:52px}
.tag{display:inline-block;font-family:var(--mono);font-size:10px;letter-spacing:.05em;margin-top:5px;padding:1px 6px;border:1px solid var(--claret);color:var(--claret);border-radius:2px}
.hint{font-family:var(--mono);font-size:10.5px;color:var(--faint);margin:8px 0 0}
.chartcap{font-family:var(--mono);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);margin:28px 0 8px}
.chart{margin-top:0}
.chart svg{width:100%;height:auto;display:block}
.chart rect{filter:brightness(var(--cboost));transition:opacity .15s ease}
.chart svg:hover rect{opacity:.45}
.chart svg rect:hover{opacity:1}
.chart .grid{stroke:var(--rule);stroke-width:1}
.chart .ytick{font-family:var(--mono);font-size:10px;fill:var(--soft);text-anchor:end}
.chart .xtick{font-family:var(--mono);font-size:11px;fill:var(--ink);text-anchor:end}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:14px}
.legend span{font-family:var(--mono);font-size:11.5px;color:var(--soft)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:7px;vertical-align:-1px;filter:brightness(var(--cboost))}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:26px 40px;font-size:14.5px;color:var(--ink2)}
.grid2 h3{font-family:var(--serif);font-size:16px;margin:0 0 6px}
.grid2 p{margin:0 0 10px}
.grid2 b{font-weight:600}
.defs{list-style:none;padding:0;margin:0;font-size:14px}
.defs li{padding:9px 0;border-bottom:1px solid var(--rule)}
.defs li:last-child{border:none}
.defs .k{font-family:var(--mono);font-size:12px;color:var(--claret);letter-spacing:.03em}
.dl{display:grid;grid-template-columns:180px 1fr;gap:4px 18px;font-size:13.5px;margin:0}
.dl dt{font-family:var(--mono);font-size:12px;color:var(--soft)}
.dl dd{margin:0;color:var(--ink2)}
.note{font-family:var(--mono);font-size:12px;color:var(--soft);border-left:2px solid var(--claret);padding:8px 0 8px 14px;margin:18px 0 0;background:var(--notebg)}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
footer{padding:34px 0 60px;color:var(--soft);font-family:var(--mono);font-size:12px}
footer .wrap{display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px}
footer a{color:var(--soft)}
footer a:hover{color:var(--claret)}
@media(max-width:640px){.grid2{grid-template-columns:1fr}.dl{grid-template-columns:1fr}.mast{padding:36px 0 22px}}
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}
@keyframes grow{from{width:0}}
.mast,section{animation:rise .6s cubic-bezier(.2,.7,.2,1) both}
@media(prefers-reduced-motion:reduce){*{animation:none!important}html{scroll-behavior:auto}}
""".replace("@@DARK@@", _DARK_VARS)


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
        '<section id="metrics"><div class="wrap"><h2 class="h2">Metrics</h2>'
        '<p class="lede">Each brief is measured on several independent axes, '
        'because no single number captures legal quality.</p>'
        '<ul class="defs">'
        '<li><span class="k">rubric score</span> — HealthBench-style checklist grading '
        '(primary when present). A strong model authors 12–20 atomic, point-weighted, '
        'case-specific criteria per case — grounded in the human reference brief or the '
        'decision text, including negative criteria for likely hallucinations — and a '
        'grader model only verifies whether each criterion is met. Score = weight met ÷ '
        'total weight. Checking a specific criterion is far more verifiable than holistic '
        'scoring, which restores discrimination at the top. Rubrics are versioned in '
        '<a href="https://github.com/okelot/LLMBenchmarkForCCL/blob/main/rubrics/rubrics.json">rubrics/rubrics.json</a>.</li>'
        f'<li><span class="k">judge score</span> — {judge_def}</li>'
        '<li><span class="k">cosine</span> — embedding similarity (all-mpnet-base-v2) '
        'between the brief and the reference. A cheap topical signal, kept as a '
        'secondary check; it cannot tell a fluent-but-wrong brief from a correct one.</li>'
        '<li><span class="k">grounded</span> — mean judge groundedness; higher means '
        'fewer fabricated facts, parties, holdings, or citations.</li>'
        '<li><span class="k">format</span> — share of responses that returned valid JSON.</li>'
        '<li><span class="k">trunc</span> — share cut off by the completion-token cap. '
        'Truncated and malformed responses are execution failures: they are excluded from '
        'the quality mean and reported here instead, so the headline score never mixes '
        'legal quality with infrastructure policy. (A strict all-rows mean is kept in '
        'data.json for anyone who prefers failures counted as zero.)</li>'
        '<li><span class="k">refuse</span> — share where the model declined ("I don\'t know"), '
        'reported separately so calibrated abstention is not confused with a wrong answer.</li>'
        '<li><span class="k">$/case</span> — mean cost per case from token usage × list price.</li>'
        '<li><span class="k">latency</span> — mean wall-clock seconds per case.</li>'
        '</ul>'
        '<p class="note">95% confidence intervals are percentile bootstraps over cases '
        '(fixed seed); the thin line beneath each score bar spans the interval. Because '
        'every model sees the same cases, ranking gaps are tested with a <b>paired</b> '
        'bootstrap on per-case differences; a ≈ marks a gap whose paired 95% CI includes '
        'zero (hover it for the interval). Sections lacking a human reference are excluded '
        'from reference-mode judging rather than scored against nothing. Sample sizes are '
        'small, so treat rankings as provisional.</p>'
        '</div></section>'
    )


def _dataset_section(tracks, meta):
    rows = "".join(
        f'<dt>{html.escape(TRACK_LABELS.get(t["key"], t["key"]))}</dt>'
        f'<dd>{t["n_models"]} models · {t["n_cases"]} cases{t.get("dates","")}</dd>'
        for t in tracks
    )
    return (
        '<section id="dataset"><div class="wrap"><h2 class="h2">Dataset</h2>'
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
        '<section id="reproducibility"><div class="wrap"><h2 class="h2">Reproducibility &amp; contamination</h2>'
        '<div class="grid2">'
        '<div><h3>How to reproduce</h3><dl class="dl">'
        '<dt>models</dt><dd>pinned in ai_models.csv (OpenRouter ids)</dd>'
        '<dt>decoding</dt><dd>temperature 0, capped max tokens</dd>'
        f'<dt>judge</dt><dd>{html.escape(meta.get("judge_model","—"))}</dd>'
        f'<dt>rubric author/grader</dt><dd>{html.escape(meta.get("rubric_model","—"))}</dd>'
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
        'run to run (served model/provider are recorded per row in newer runs); pin a '
        'provider for byte-exact reproducibility. Some reasoning models ignore the '
        'requested temperature — the decoding config actually applied is recorded, not '
        'assumed.</p>'
        '<p class="note"><b>Judge caveats.</b> Current data is graded by a judge from a '
        'family with no contestant on the board (Qwen), blind to model identity. A '
        'judge-swap experiment (OpenAI-family judge vs cross-family judge, same briefs) '
        'shifted absolute scores materially and moved mid-table order — treat scores as '
        'judge-relative, not absolute. Notably, the OpenAI judge ranked an OpenAI model '
        'above Claude Opus on closed-book; the cross-family judge reversed that. '
        'Open-book scores show a ceiling effect under both judges and rubrics (most '
        'ratings perfect), so top-of-table open-book gaps are weak evidence. Next '
        'mitigations: multi-judge panels and a human-graded calibration sample.</p>'
        '</div></div></div></section>'
    )


SITE_URL = "https://okelot.github.io/LLMBenchmarkForCCL/"

FAVICON = ("data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
           "<rect width='64' height='64' rx='10' fill='%237b1e2b'/>"
           "<text x='32' y='46' font-family='Georgia,serif' font-weight='700' font-size='40' "
           "fill='%23f3f2ec' text-anchor='middle'>L</text></svg>")

TRACK_SHORT = {"closed": "Closed-book", "open": "Open-book"}

# Runs before first paint so a stored theme choice doesn't flash the wrong scheme.
THEME_BOOT_JS = ("try{var t=localStorage.getItem('lexbench-theme');"
                 "if(t)document.documentElement.setAttribute('data-theme',t)}catch(e){}")

PAGE_JS = """
(function(){
  var K='lexbench-theme',d=document.documentElement;
  var b=document.getElementById('themetoggle');
  if(b)b.addEventListener('click',function(){
    var cur=d.getAttribute('data-theme')||
      (matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');
    var n=cur==='dark'?'light':'dark';
    d.setAttribute('data-theme',n);
    try{localStorage.setItem(K,n)}catch(e){}
  });
  document.querySelectorAll('table.board').forEach(function(tb){
    var body=tb.tBodies[0];
    tb.querySelectorAll('thead th[data-sort]').forEach(function(th){
      th.tabIndex=0;
      var act=function(){
        var idx=th.cellIndex,num=th.getAttribute('data-sort')==='num';
        var desc=num?th.getAttribute('aria-sort')!=='descending'
                    :th.getAttribute('aria-sort')==='ascending';
        tb.querySelectorAll('thead th').forEach(function(o){o.removeAttribute('aria-sort')});
        th.setAttribute('aria-sort',desc?'descending':'ascending');
        Array.prototype.slice.call(body.rows).sort(function(a,b){
          var x=a.cells[idx].getAttribute('data-v')||'',
              y=b.cells[idx].getAttribute('data-v')||'',r;
          if(num){x=parseFloat(x);y=parseFloat(y);
            if(isNaN(x))x=-Infinity;if(isNaN(y))y=-Infinity}
          r=x<y?-1:x>y?1:0;
          return desc?-r:r;
        }).forEach(function(row){body.appendChild(row)});
      };
      th.addEventListener('click',act);
      th.addEventListener('keydown',function(e){
        if(e.key==='Enter'||e.key===' '){e.preventDefault();act()}});
    });
  });
})();
"""


def render(tracks, meta):
    lead_track = next((t for t in tracks if t["agg"]["rows"]), None)
    leader = lead_track["agg"]["rows"][0] if lead_track else None
    has_judge = any(t["agg"]["primary"] in ("judge", "rubric") for t in tracks)
    primary_name = next((t["agg"]["primary_label"] for t in tracks if t["agg"]["rows"]), "cosine")
    desc = ("LexBench benchmarks frontier LLMs on Canadian case-law briefing with an "
            "LLM-judge rubric, contamination-resistant open-book and temporal-holdout "
            "tracks, and confidence intervals.")

    p = ["<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         "<title>LexBench — Frontier LLMs on Canadian case law</title>",
         f'<meta name="description" content="{desc}">',
         f'<link rel="canonical" href="{SITE_URL}">',
         f'<link rel="icon" href="{FAVICON}">',
         '<meta property="og:title" content="LexBench — Frontier LLMs on Canadian case law">',
         f'<meta property="og:description" content="{desc}">',
         '<meta property="og:type" content="website">',
         f'<meta property="og:url" content="{SITE_URL}">',
         '<meta name="twitter:card" content="summary">',
         '<meta name="theme-color" media="(prefers-color-scheme: light)" content="#e6e5df">',
         '<meta name="theme-color" media="(prefers-color-scheme: dark)" content="#181a1c">',
         '<!-- Google tag (gtag.js) -->',
         '<script async src="https://www.googletagmanager.com/gtag/js?id=G-D8P8DMTDRM"></script>',
         '<script>',
         '  window.dataLayer = window.dataLayer || [];',
         '  function gtag(){dataLayer.push(arguments);}',
         "  gtag('js', new Date());",
         '',
         "  gtag('config', 'G-D8P8DMTDRM');",
         '</script>',
         "<script>" + THEME_BOOT_JS + "</script>",
         "<style>" + CSS + "</style></head><body>"]

    # sticky nav
    live_tracks = [t for t in tracks if t["agg"]["rows"]]
    nav_links = "".join(
        f'<a href="#track-{html.escape(t["key"])}">{html.escape(TRACK_SHORT.get(t["key"], t["label"]))}</a>'
        for t in live_tracks)
    p.append('<a class="skip" href="#main">Skip to leaderboards</a>')
    p.append('<nav class="topnav"><div class="wrap">'
             '<a class="brand" href="#top">Lex<b>Bench</b></a>'
             f'<div class="links">{nav_links}'
             '<a href="#metrics">Metrics</a>'
             '<a href="#dataset">Dataset</a>'
             '<a href="#reproducibility">Reproducibility</a>'
             '<a href="lawyers.html">For lawyers</a>'
             '<a href="https://github.com/okelot/LLMBenchmarkForCCL">GitHub</a></div>'
             '<button id="themetoggle" class="tbtn" type="button" '
             'aria-label="Toggle dark mode" title="Toggle dark mode">◐</button>'
             '</div></nav>')

    # masthead
    p.append('<header class="mast" id="top"><div class="wrap">')
    p.append('<p class="eyebrow">Canadian Case-Law Benchmark</p>')
    p.append('<h1 class="wordmark"><span class="lex">Lex</span><span class="bench">Bench</span></h1>')
    p.append('<p class="thesis">Frontier language models, judged on how faithfully they brief Canadian case law.</p>')
    if leader and leader["primary_overall"] is not None:
        p.append(f'<div class="leader"><span class="lab">Leader · {html.escape(lead_track["label"])}</span>'
                 f'<span class="who">{html.escape(leader["model"])}</span>'
                 f'<span class="sc" style="color:{_rgb(_score_rgb(leader["primary_overall"]))}">'
                 f'{leader["primary_overall"]:.3f}</span></div>')
    plural = "" if meta["tracks"] == 1 else "s"
    p.append('<p class="cite">'
             f'<span>Updated <b>{meta["updated"]}</b></span>'
             f'<span><b>{meta["tracks"]}</b> track{plural}</span>'
             f'<span><b>{meta["models"]}</b> models</span>'
             f'<span>primary metric: <b>{html.escape(primary_name)}</b></span>'
             f'<span>with <b>95% CIs</b></span>'
             '<span><a href="data.json">raw data (JSON)</a></span></p>')
    p.append("</div></header>")

    # per-track leaderboards
    p.append('<main id="main">')
    for t in live_tracks:
        agg = t["agg"]
        p.append(f'<section id="track-{html.escape(t["key"])}"><div class="wrap">')
        p.append('<h2 class="h2">Leaderboard</h2>')
        p.append(f'<p class="trackname">{html.escape(t["label"])}'
                 f'<span class="nchip">{t["n_cases"]} cases · {t["n_models"]} models</span></p>')
        p.append(f'<p class="lede">{t["desc"]}</p>')
        p.append(build_leaderboard(agg, t["label"]))
        p.append(f'<p class="chartcap">{agg["primary_label"]} by brief section — hover a bar for its value</p>')
        p.append('<div class="chart">' + build_chart_svg(agg["rows"], agg["sections"], agg["primary"]) + '</div>')
        p.append('<div class="legend">' + build_legend(agg["sections"]) + '</div>')
        p.append("</div></section>")

    p.append(_methodology_section(has_judge))
    p.append(_dataset_section(tracks, meta))
    p.append(_repro_section(meta))
    p.append('</main>')

    p.append('<footer><div class="wrap">')
    p.append(f'<span>LexBench · generated {meta["generated"]}</span>')
    p.append('<span>data: <a href="https://a2aj.ca/">A2AJ</a> &amp; public case-brief wikis</span>')
    p.append('<span><a href="https://github.com/okelot/LLMBenchmarkForCCL">github.com/okelot/LLMBenchmarkForCCL</a></span>')
    p.append("</div></footer>")
    p.append("<script>" + PAGE_JS + "</script></body></html>")
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

    def _first_val(col):
        for _, f in frames:
            if col in f.columns and f[col].notna().any():
                return str(f[col].dropna().iloc[0])
        return "—"

    meta = {
        "updated": datetime.fromtimestamp(Path(input_file).stat().st_mtime).strftime("%d %b %Y"),
        "generated": datetime.now().strftime("%d %b %Y, %H:%M"),
        "models": max((t["n_models"] for t in tracks), default=0),
        "tracks": len(tracks),
        "judge_model": _first_val("judge_model"),
        "rubric_model": _first_val("rubric_model"),
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
