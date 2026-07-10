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
