"""Reproducible recomputation of the representation-geometry numbers cited in the
revised manuscript (Results 3.5) and in the Response to Reviewers (A8).

Everything printed here is derived from the raw probe outputs of the two
EfficientNetV2B0 models (CE only vs. CE+SupCon):

    cosine_probe_result/raw_cosprobe_effnet_classification_only.csv
    cosine_probe_result/raw_cosprobe_effnet_supcon.csv

Columns used
------------
hcc_cosine_mean : cosine similarity between the lesion embedding and the HCC
                  mean prototype (signed; negative when the lesion is far).
hem_cosine_mean : same against the hemangioma prototype.
delta_mean      : hcc_cosine_mean - hem_cosine_mean.
The two files share row order and labels (verified below), so cases are paired
by row index.

Statistics
----------
* location shift  : Wilcoxon signed-rank test (paired, two-sided) and paired t-test
* variance shrink : Pitman-Morgan test for correlated variances
                    t = (sx^2 - sy^2) * sqrt(n - 2) / (2 * sx * sy * sqrt(1 - r^2)), df = n - 2
Run with the Windows interpreter that has scipy:
    /mnt/c/Python314/python.exe recompute_representation_geometry.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "cosine_probe_result"
CE_CSV = RES / "raw_cosprobe_effnet_classification_only.csv"
SP_CSV = RES / "raw_cosprobe_effnet_supcon.csv"

SPLITS = ("Val", "Test")


def load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def values(rows: list[dict], key: str, label: str) -> list[float]:
    return [float(r[key]) for r in rows if r["split"] in SPLITS and r["true_label"] == label]


def describe(x: list[float]) -> dict:
    xs = sorted(x)
    n = len(xs)
    mean = sum(xs) / n
    var = sum((v - mean) ** 2 for v in xs) / (n - 1)
    q = lambda p: xs[min(n - 1, int(p * n))]  # noqa: E731
    return {
        "n": n,
        "mean": mean,
        "sd": math.sqrt(var),
        "var": var,
        "median": statistics_median(xs),
        "p10": q(0.10),
        "p90": q(0.90),
        "min": xs[0],
    }


def statistics_median(xs_sorted: list[float]) -> float:
    n = len(xs_sorted)
    mid = n // 2
    return xs_sorted[mid] if n % 2 else (xs_sorted[mid - 1] + xs_sorted[mid]) / 2


def pitman_morgan(x: list[float], y: list[float]) -> tuple[float, float]:
    """Pitman-Morgan test for the equality of two correlated variances."""
    n = len(x)
    sx, sy = stats.tstd(x, ddof=1), stats.tstd(y, ddof=1)
    r = stats.pearsonr(x, y).statistic
    t = (sx**2 - sy**2) * math.sqrt(n - 2) / (2 * sx * sy * math.sqrt(1 - r**2))
    return t, 2 * stats.t.sf(abs(t), df=n - 2)


def report(tag: str, base: list[float], new: list[float]) -> dict:
    a, b = describe(base), describe(new)
    n = len(base)
    w = stats.wilcoxon(base, new)
    pt = stats.ttest_rel(base, new)
    pm_t, pm_p = pitman_morgan(base, new)
    gains = [y - x for x, y in zip(base, new)]
    up = sum(1 for g in gains if g > 0)
    r_gain = stats.pearsonr(base, gains).statistic
    var_change = (b["var"] - a["var"]) / a["var"] * 100

    print(f"\n=== {tag} (Val+Test, n={n}) ===")
    print(f"  median      {a['median']:+.4f} -> {b['median']:+.4f}")
    print(f"  mean ± SD   {a['mean']:+.4f} ± {a['sd']:.4f}  ->  {b['mean']:+.4f} ± {b['sd']:.4f}")
    print(f"  variance    {a['var']:.5f} -> {b['var']:.5f}   ({var_change:+.1f}%)")
    print(f"  p10 / p90   {a['p10']:+.4f} / {a['p90']:+.4f}  ->  {b['p10']:+.4f} / {b['p90']:+.4f}")
    print(f"  min         {a['min']:+.4f} -> {b['min']:+.4f}")
    print(f"  per-case up {up}/{n} = {up / n * 100:.1f}%   (down {n - up})")
    print(f"  r(baseline, gain) = {r_gain:+.3f}")
    print(f"  Wilcoxon    W={w.statistic:.1f}  p={w.pvalue:.3e}   (paired t p={pt.pvalue:.3e})")
    print(f"  Pitman-Morgan t={pm_t:.2f}  p={pm_p:.3e}")
    return {"describe": a, "new": b, "var_change": var_change, "up_pct": up / n * 100, "r_gain": r_gain}


def main() -> None:
    ce, sp = load(CE_CSV), load(SP_CSV)
    assert len(ce) == len(sp), "row counts differ"
    assert all(
        (a["split"], a["true_label"]) == (b["split"], b["true_label"]) for a, b in zip(ce, sp)
    ), "row order / labels are not paired - do not pair by index"
    print(f"paired rows verified: {len(ce)}  (source: {CE_CSV.name} vs {SP_CSV.name})")

    hcc_before = values(ce, "hcc_cosine_mean", "1")
    hcc_after = values(sp, "hcc_cosine_mean", "1")
    hem_before = values(ce, "hem_cosine_mean", "0")
    hem_after = values(sp, "hem_cosine_mean", "0")
    hcc = report("true HCC - HCC cosine (same class)", hcc_before, hcc_after)
    hem = report("true hemangioma - hemangioma cosine (same class)", hem_before, hem_after)

    print("\n=== cross-class drift (what the same SupCon did to the OTHER prototype) ===")
    for tag, key, label in (
        ("true HCC - hemangioma cosine", "hem_cosine_mean", "1"),
        ("true hemangioma - HCC cosine", "hcc_cosine_mean", "0"),
    ):
        before = statistics_median(sorted(values(ce, key, label)))
        after = statistics_median(sorted(values(sp, key, label)))
        print(f"  {tag:34} median {before:+.3f} -> {after:+.3f}  ({after - before:+.3f})")

    print("\n=== Delta-score class gap (median difference between classes) ===")
    for name, rows in (("CE only", ce), ("CE+SupCon", sp)):
        hem_cases = sorted(values(rows, "delta_mean", "0"))
        hcc_cases = sorted(values(rows, "delta_mean", "1"))
        gap = statistics_median(hcc_cases) - statistics_median(hem_cases)
        print(f"  {name:10} hem {statistics_median(hem_cases):+.3f}  hcc {statistics_median(hcc_cases):+.3f}  gap {gap:.3f}")
    gap_before = statistics_median(sorted(values(ce, "delta_mean", "1"))) - statistics_median(sorted(values(ce, "delta_mean", "0")))
    gap_after = statistics_median(sorted(values(sp, "delta_mean", "1"))) - statistics_median(sorted(values(sp, "delta_mean", "0")))
    print(f"  gap change: {gap_before:.3f} -> {gap_after:.3f}  ({(gap_after - gap_before) / gap_before * 100:+.1f}%)")

    # --- assertions against the numbers printed in the manuscript / slides ---
    ok = (
        abs(hcc["describe"]["median"] - 0.882) < 5e-4
        and abs(hcc["new"]["median"] - 0.970) < 5e-4
        and abs(hcc["var_change"] + 60.7) < 0.15
        and abs(hem["var_change"] + 98.7) < 0.15
        and abs(hcc["up_pct"] - 94.7) < 0.05
        and abs(hcc["r_gain"] + 0.802) < 0.01
        and abs(gap_before - 3.163) < 5e-3
        and abs(gap_after - 2.289) < 5e-3
    )
    print(f"\nall cited numbers reproduced: {ok}")
    if not ok:
        raise SystemExit("one or more cited numbers could not be reproduced")


if __name__ == "__main__":
    main()
