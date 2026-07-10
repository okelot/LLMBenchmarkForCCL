"""Small statistics helpers for reporting rigor.

A benchmark number without an uncertainty band is a point estimate masquerading
as a fact. These helpers give every headline score a bootstrap confidence
interval and let us flag when a gap between two models is not distinguishable
from noise.
"""

from typing import List, Optional, Tuple

import numpy as np

# Fixed seed so confidence intervals are reproducible run to run.
_SEED = 12345


def _clean(values: List[float]) -> np.ndarray:
    arr = np.array([v for v in values if v is not None and not _isnan(v)], dtype=float)
    return arr


def _isnan(v) -> bool:
    try:
        return np.isnan(v)
    except TypeError:
        return False


def bootstrap_ci(values: List[float], n_boot: int = 2000,
                 alpha: float = 0.05) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (mean, lo, hi) for a (1-alpha) percentile bootstrap CI of the mean.

    Returns (None, None, None) if there are no usable values; a degenerate
    (mean, mean, mean) if there is only one.
    """
    arr = _clean(values)
    if arr.size == 0:
        return None, None, None
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(_SEED)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return round(mean, 4), round(lo, 4), round(hi, 4)


def paired_bootstrap_diff(pairs: List[Tuple[float, float]], n_boot: int = 2000,
                          alpha: float = 0.05) -> dict:
    """Paired bootstrap of the mean difference (a - b) over shared cases.

    Every model sees the same cases, so the correct comparison resamples
    per-case differences, not two independent marginals — this respects the
    paired design and gives much tighter, honest intervals.

    `pairs` is a list of (a_score, b_score) tuples for the SAME case. Returns
    dict with diff, lo, hi, n and `significant` (CI excludes zero).
    """
    clean = [(a, b) for a, b in pairs
             if a is not None and b is not None and not _isnan(a) and not _isnan(b)]
    if not clean:
        return {"diff": None, "lo": None, "hi": None, "n": 0, "significant": False}
    d = np.array([a - b for a, b in clean], dtype=float)
    if d.size == 1:
        v = round(float(d[0]), 4)
        return {"diff": v, "lo": v, "hi": v, "n": 1, "significant": False}
    rng = np.random.default_rng(_SEED)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boot = d[idx].mean(axis=1)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return {"diff": round(float(d.mean()), 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "n": int(d.size), "significant": bool(lo > 0 or hi < 0)}


def bootstrap_diff(a: List[float], b: List[float], n_boot: int = 2000,
                   alpha: float = 0.05) -> dict:
    """Bootstrap the difference in means (a - b).

    Returns dict with diff, lo, hi, and `significant` (True when the CI excludes
    zero). Used to flag whether a leaderboard gap is real.
    """
    aa, bb = _clean(a), _clean(b)
    if aa.size == 0 or bb.size == 0:
        return {"diff": None, "lo": None, "hi": None, "significant": False}
    rng = np.random.default_rng(_SEED)
    da = aa[rng.integers(0, aa.size, size=(n_boot, aa.size))].mean(axis=1)
    db = bb[rng.integers(0, bb.size, size=(n_boot, bb.size))].mean(axis=1)
    diff = da - db
    lo = float(np.percentile(diff, 100 * alpha / 2))
    hi = float(np.percentile(diff, 100 * (1 - alpha / 2)))
    return {
        "diff": round(float(aa.mean() - bb.mean()), 4),
        "lo": round(lo, 4),
        "hi": round(hi, 4),
        "significant": bool(lo > 0 or hi < 0),
    }
