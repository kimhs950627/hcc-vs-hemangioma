"""Recompute every number that will appear on the v2 deck, straight from the
per-case score CSVs. Prints a compact verification report."""
import csv
import statistics as st

BASE = "cosine_probe_result"


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def youden(rows, col):
    """Youden's J optimum on a split, same rule as the paper scripts."""
    ys = [(float(r[col]), r["true_label"] == "1") for r in rows]
    uniq = sorted({s for s, _ in ys}, reverse=True)
    best = (-1.0, None)
    for t in uniq:
        tp = sum(1 for s, y in ys if s >= t and y)
        fn = sum(1 for s, y in ys if s < t and y)
        fp = sum(1 for s, y in ys if s >= t and not y)
        tn = sum(1 for s, y in ys if s < t and not y)
        sens = tp / (tp + fn) if tp + fn else 0
        spec = tn / (tn + fp) if tn + fp else 0
        j = sens + spec - 1
        if j > best[0]:
            best = (j, t)
    return best[1]


def block(label, col, cut):
    y = [(float(r[col]), r["true_label"] == "1") for r in rows]
    pred = [(s >= cut) for s, _ in y]
    tp = sum(1 for (s, t), p in zip(y, pred) if p and t)
    fp = sum(1 for (s, t), p in zip(y, pred) if p and not t)
    fn = sum(1 for (s, t), p in zip(y, pred) if not p and t)
    tn = sum(1 for (s, t), p in zip(y, pred) if not p and not t)
    n = len(y)
    acc = (tp + tn) / n
    sens = tp / (tp + fn) if tp + fn else 0
    spec = tn / (tn + fp) if tn + fp else 0
    ppv = tp / (tp + fp) if tp + fp else 0
    npv = tn / (tn + fn) if tn + fn else 0
    f1 = 2 * ppv * sens / (ppv + sens) if ppv + sens else 0
    print(f"  {label:<22} cutoff={cut:<10.4f} n={n:<4} Acc={acc*100:.2f}% "
          f"Sens={sens*100:.2f}% Spec={spec*100:.2f}% PPV={ppv*100:.2f}% "
          f"NPV={npv*100:.2f}% F1={f1:.4f}  TP={tp} FP={fp} FN={fn} TN={tn}")


def dist(tag, path, col, want_hcc):
    """Mean / median / variance of a score column for one true class."""
    rows = load(path)
    sel = [float(r[col]) for r in rows
           if r["split"] in ("Val", "Test") and (r["true_label"] == "1") == want_hcc]
    med = st.median(sel)
    mean = st.fmean(sel)
    var = st.pvariance(sel)
    print(f"  {tag:<34} n={len(sel):<4} mean={mean:.3f}  median={med:.3f}  var={var:.4f}")
    return mean, med, var


sup = f"{BASE}/effnet_supcon/all_scores.csv"
ceo = f"{BASE}/raw_cosprobe_effnet_classification_only.csv"

print("=" * 96)
print("1) PRIMARY MODEL (EfficientNetV2B0 + CE+SupCon) - thresholds from Val Youden's J")
print("=" * 96)
val = [r for r in load(sup) if r["split"] == "Val"]
test = [r for r in load(sup) if r["split"] == "Test"]
cuts = {}
for col, name in (("confidence_score", "Confidence"), ("hcc_cosine_mean", "HCC cosine"), ("delta_mean", "Delta score")):
    cuts[col] = youden(val, col)
    print(f"  Youden cutoff on Val [{name:<12}] = {cuts[col]:.4f}")

print("\n  --- Validation ---")
for col, name in (("confidence_score", "Confidence"), ("hcc_cosine_mean", "HCC cosine"), ("delta_mean", "Delta score")):
    globals().update(rows=val)
    block(name, col, cuts[col])
print("  --- Test ---")
for col, name in (("confidence_score", "Confidence"), ("hcc_cosine_mean", "HCC cosine"), ("delta_mean", "Delta score")):
    globals().update(rows=test)
    block(name, col, cuts[col])

print("\n" + "=" * 96)
print("2) SupCon EFFECT ON SCORE DISTRIBUTIONS  (Val + Test)")
print("=" * 96)
for cls, col, hcc in (("True HCC - cosine", "hcc_cosine_mean", True),
                      ("True HCC - delta ", "delta_mean", True),
                      ("True hem  - cosine", "hem_cosine_mean", False)):
    print(f"  [{cls}]")
    print("    CE only      ", end="")
    m0, md0, v0 = dist("CE only", ceo, col, hcc)
    print("    CE + SupCon  ", end="")
    m1, md1, v1 = dist("CE+SupCon", sup, col, hcc)
    print(f"    shift: mean {m1-m0:+.3f} | median {md1-md0:+.3f} | variance {100*(v0-v1)/v0:+.1f}%")

print("\n" + "=" * 96)
print("3) AMBIGUOUS CONFIDENCE BAND 0.40-0.60  (Val + Test)")
print("=" * 96)
for tag, path, conf, cutcol in (
    ("EfficientNet CE+SupCon", sup, "confidence_score", "hcc_cosine_mean"),
    ("EfficientNet CE only  ", ceo, "confidence_score", None),
):
    rows2 = [r for r in load(path)
             if r["split"] in ("Val", "Test") and 0.40 <= float(r[conf]) <= 0.60]
    hcc = sum(1 for r in rows2 if r["true_label"] == "1")
    print(f"  {tag}: n={len(rows2)}  (HCC {hcc}, hemangioma {len(rows2)-hcc})")
    if cutcol and rows2:
        c = youden([r for r in load(path) if r["split"] == "Val"], cutcol)
        good = sum(1 for r in rows2 if (float(r[cutcol]) >= c) == (r["true_label"] == "1"))
        print(f"    -> HCC cosine (cutoff {c:.4f}) gets {good}/{len(rows2)} right")
    if cutcol is None and rows2:
        good = sum(1 for r in rows2 if (float(r[conf]) >= 0.5) == (r["true_label"] == "1"))
        print(f"    -> raw confidence at 0.5 gets {good}/{len(rows2)} right")

print("\n" + "=" * 96)
print("4) MIMIC SET - hyperechoic (hemangioma-like) HCC, published figures, n=5")
print("=" * 96)
mim = load("scratch/mimic_reports/mimic_evaluation_summary.csv")
for col in ("P_HCC", "HCC_Cosine", "Delta_Score", "Decision_Conf", "Decision_HCC_Cos", "Decision_Delta"):
    vals = [r[col] for r in mim]
    if col.startswith("Decision"):
        print(f"  {col:<18} HCC called {sum(1 for v in vals if v == 'HCC')}/{len(vals)}")
    else:
        nums = [float(v) for v in vals]
        print(f"  {col:<18} min={min(nums):.3f}  max={max(nums):.3f}")
