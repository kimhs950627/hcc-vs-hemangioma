"""
scratch/run_mimic_eval.py
========================
Evaluate 5 atypical hyperechoic HCC cases (mimicking hemangioma) using KaggleExplainablePredictor.
"""

from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import json
import pandas as pd
from inference.kaggle_explainable_predictor import KaggleExplainablePredictor


def main():
    root = Path(__file__).resolve().parent.parent
    weights_path = root / "cosine_probe_result" / "effnet_supcon" / "classifier_y0taaeki_BM.weights.h5"
    img_dir = root / "example_images_hcc_case" / "hcc_mimicking_hemangioma_web" / "images"
    out_dir = root / "scratch" / "mimic_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing predictor...")
    predictor = KaggleExplainablePredictor(weights_path=weights_path)

    images = sorted(list(img_dir.glob("*.png")))
    print(f"Found {len(images)} images to evaluate.")

    results_summary = []

    for idx, img_path in enumerate(images, start=1):
        case_name = img_path.stem
        out_plot = out_dir / f"{case_name}_report.png"
        print(f"\n[{idx}/5] Evaluating {case_name}...")

        res = predictor.predict_and_explain(
            image_path=img_path,
            save_plot_path=out_plot,
            show_inline=False,
            title_prefix=f"Mimic Case {idx}: {case_name[:30]}"
        )

        results_summary.append({
            "index": idx,
            "filename": img_path.name,
            "P_HCC": res["confidence_score"],
            "P_Hem": res["p_hemangioma"],
            "HCC_Cosine": res["hcc_cosine_score"],
            "Hem_Cosine": res["hem_cosine_score"],
            "Delta_Score": res["delta_score"],
            "Decision_Conf": res["predictions"]["by_confidence"],
            "Decision_HCC_Cos": res["predictions"]["by_hcc_cosine"],
            "Decision_Delta": res["predictions"]["by_delta"],
            "report_path": str(out_plot),
            "summary_text": res["summary"]
        })

    df = pd.DataFrame(results_summary)
    csv_path = out_dir / "mimic_evaluation_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nAll 5 evaluations completed. Summary saved to: {csv_path}")

    # Output JSON summary for quick inspection
    json_path = out_dir / "mimic_evaluation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
