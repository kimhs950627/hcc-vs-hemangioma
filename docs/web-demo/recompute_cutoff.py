#!/usr/bin/env python3
"""외부(held-out) 데이터에서 세 지표의 컷오프를 다시 계산한다.

SMC-LUD validation cohort 의 Youden 지점은 그 코호트 안에서 AUC 1.0 이 나왔고,
그 값(confidence 0.0011 / cosine -0.0048 / delta -0.958)을 다른 데이터에 그대로 쓰면
사실상 "전부 HCC" 로 판정한다. 그래서 데모는 관행값(0.5 / 0 / 0)을 쓰고, 외부 코호트가
확보되면 이 스크립트로 다시 뽑아 `config.json` 의 `cutoffs` 를 갱신한다.

입력: `examples/expected.json` 과 같은 모양의 JSON
    {"<file>.jpg": {"probs": [p_hem, p_hcc], "hccCosine": ..., "hemCosine": ..., "delta": ...}, ...}
라벨은 파일명 접두사(hcc / hem / hemangioma)로 읽고, 아니면 `--labels labels.json`
({"<file>.jpg": "HCC"|"Hemangioma"}) 로 넘긴다.

    python3 recompute_cutoff.py examples/expected.json
    python3 recompute_cutoff.py external.json --labels external_labels.json --apply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ("confidence", "hccCosine", "delta")


def label_of(name: str, labels: dict[str, str] | None) -> int:
    if labels and name in labels:
        value = labels[name].strip().lower()
    else:
        value = name.strip().lower()
        value = "hcc" if "hcc" in value else "hem" if "hem" in value else ""
    if value.startswith("hcc") or value in {"1", "malignant", "positive"}:
        return 1
    if value.startswith("hem") or value in {"0", "benign", "negative"}:
        return 0
    raise ValueError(f"라벨을 알 수 없음: {name} — --labels 로 지정할 것")


def value_of(entry: dict, metric: str) -> float:
    if metric == "confidence":
        return float(entry["probs"][1])
    if metric == "hccCosine":
        return float(entry["hccCosine"])
    return float(entry["delta"])


def best_cutoff(scores: list[float], truth: list[int]) -> tuple[float, float, float, float]:
    """Youden J 를 최대화하는 컷오프.

    J 는 인접한 두 점수 사이에서만 일정하므로, 컷오프를 그 구간의 중앙값으로 잡는다.
    (cutoff, J, 구간하한, 구간상한) 반환 — 구간 안 어느 값이든 성능이 같다.
    """
    order = sorted(set(scores))
    positives = sum(truth)
    negatives = len(truth) - positives
    candidates = [(order[0] - 1e-6, order[0] - 1e-6, order[0])]
    candidates += [((a + b) / 2, a, b) for a, b in zip(order, order[1:])]
    candidates.append((order[-1] + 1e-6, order[-1], order[-1] + 1e-6))

    best_j = -1.0
    best: tuple[float, float, float] = (0.0, 0.0, 0.0)
    for cutoff, low, high in candidates:
        tp = sum(1 for s, t in zip(scores, truth) if s >= cutoff and t == 1)
        fp = sum(1 for s, t in zip(scores, truth) if s >= cutoff and t == 0)
        j = tp / positives + (negatives - fp) / negatives - 1.0
        if j > best_j + 1e-12:
            best_j, best = j, (cutoff, low, high)
    cutoff, low, high = best
    return cutoff, best_j, low, high


def auc(scores: list[float], truth: list[int]) -> float:
    pos = [s for s, t in zip(scores, truth) if t == 1]
    neg = [s for s, t in zip(scores, truth) if t == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scores", type=Path, help="expected.json 형식의 예측값 JSON")
    parser.add_argument("--labels", type=Path, help='{"file.jpg": "HCC"|"Hemangioma"}')
    parser.add_argument("--apply", action="store_true", help="같은 폴더 config.json 의 cutoffs 를 갱신")
    args = parser.parse_args()

    entries = json.loads(args.scores.read_text(encoding="utf-8"))
    labels = json.loads(args.labels.read_text(encoding="utf-8")) if args.labels else None
    truth = [label_of(name, labels) for name in entries]
    if len(set(truth)) < 2:
        raise SystemExit("두 클래스가 모두 있어야 함")

    cutoffs: dict[str, float] = {}
    for metric in METRICS:
        scores = [value_of(entry, metric) for entry in entries.values()]
        cutoff, j, low, high = best_cutoff(scores, truth)
        cutoffs[metric] = round(cutoff, 6)
        print(f"{metric:11s} AUC={auc(scores, truth):.3f}  J={j:.2f}  "
              f"cutoff={cutoff:.4f}  (최적 구간 {low:.4f} ~ {high:.4f}, n={len(scores)})")

    exact = all(auc([value_of(e, m) for e in entries.values()], truth) == 1.0 for m in METRICS)
    if exact:
        print("경고: 이 데이터에서 세 지표 모두 AUC 1.0 — 최적 컷오프가 유일하지 않다(구간 전체가 동률).")
        print("      표본이 분리되어 있으므로 0.5 를 포함한 구간 중앙값을 쓰거나, 표본을 늘려 다시 뽑을 것.")

    if args.apply:
        config_path = args.scores.parent / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["cutoffs"] = {
            "confidence_score": cutoffs["confidence"],
            "hcc_cosine_score": cutoffs["hccCosine"],
            "delta_score": cutoffs["delta"],
        }
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"갱신: {config_path} → {config['cutoffs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
