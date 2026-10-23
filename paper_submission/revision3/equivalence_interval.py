"""Reproduce the reported cutoffs and their equivalence intervals.

Definition used in the manuscript
---------------------------------
For a cutoff c, the equivalence interval is the set of thresholds that assign
exactly the same label to every case of the derivation (validation) set, and
therefore reproduce the same confusion matrix:

    I(c) = ( max{ s : s < c },  min{ s : s >= c } ]

The prediction rule is  positive  <=>  score >= cutoff, so the interval is open
on the left (the highest negative must stay below the threshold) and closed on
the right (a case exactly at the threshold is still called positive).

Because AUROC reaches 1.000 on the validation set, the interval is bounded by
observed scores on both sides and is wide. For the two similarity outputs the
chosen cutoff coincides with the upper bound, i.e. with the lowest-scoring HCC
case -- the number should therefore be read as an interval, not as a unique
optimum.

Run:  python3 equivalence_interval.py
"""

from __future__ import annotations

import csv
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
RAW = REPO / "cosine_probe_result" / "raw_cosprobe_effnet_supcon.csv"

# cutoffs as applied (validation-derived; see Table 3 and Table 3B)
CUTOFFS = {
    "confidence_score": 0.50,                 # prespecified conventional value
    "hcc_cosine_mean": -0.004843093454837799,  # validation Youden plateau, upper edge
    "delta_mean": -0.9584707617759705,         # validation Youden plateau, upper edge
}


def equivalence_interval(scores: list[float], cutoff: float) -> tuple[float, float]:
    """Smallest closed-on-the-right interval of thresholds with identical labels."""
    negatives = [s for s in scores if s < cutoff]
    positives = [s for s in scores if s >= cutoff]
    if not negatives or not positives:
        raise ValueError("interval unbounded: one side has no cases")
    return max(negatives), min(positives)


def main() -> None:
    with RAW.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "Val"]

    labels = [int(float(r["true_label"])) for r in rows]
    print(f"validation n={len(rows)} (HCC {sum(labels)}, hemangioma {len(labels) - sum(labels)})")
    print()

    for column, cutoff in CUTOFFS.items():
        scores = [float(r[column]) for r in rows]
        lo, hi = equivalence_interval(scores, cutoff)
        preds = [1 if s >= cutoff else 0 for s in scores]
        tp = sum(1 for t, p in zip(labels, preds) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(labels, preds) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(labels, preds) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(labels, preds) if t == 0 and p == 0)
        at_upper = "cutoff == upper bound" if abs(cutoff - hi) < 1e-12 else ""
        print(f"{column}")
        print(f"  cutoff           {cutoff:+.6f}")
        print(f"  interval         ({lo:+.6f}, {hi:+.6f}]   width {hi - lo:.6f}")
        print(f"  rounded          ({lo:+.4f}, {hi:+.4f}]")
        print(f"  confusion (Val)  TP {tp}  FP {fp}  FN {fn}  TN {tn}   {at_upper}")
        print()


if __name__ == "__main__":
    main()
