"""Ablation KS table: YOLOv8x baseline vs Soccana, both compared against GT.

Reads:
    outputs/pitch_control.parquet           (track in {pipeline, gt})
    outputs/pitch_control_soccana.parquet   (track = soccana)

Writes:
    outputs/ablation_ks_summary.parquet     (row per detector x metric)
    outputs/figures/12_ablation_ks_table.png  (rendered table figure)

Uses the same locked nb04 parameters: KS alpha 0.05, 12 histogram bins, 5 metrics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

KS_ALPHA = 0.05
HIST_BINS = 12
METRICS = ["pc_mean", "pc_at_ball", "pc_in_box", "pc_in_third", "pc_area_gt_0p5"]


def histogram_overlap(a, b, bins=HIST_BINS, lo=0.0, hi=1.0):
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)
    pa = pa / pa.sum() if pa.sum() else pa
    pb = pb / pb.sum() if pb.sum() else pb
    return float(np.minimum(pa, pb).sum())


def compare(a, b, label, metric):
    if len(a) < 3 or len(b) < 3:
        return {"detector": label, "metric": metric, "n": len(a), "n_gt": len(b),
                "ks_stat": np.nan, "ks_p": np.nan, "hist_overlap": np.nan,
                "mean": np.nan, "mean_gt": np.nan, "delta": np.nan, "passes_ks": False}
    ks = stats.ks_2samp(a, b)
    return {
        "detector": label, "metric": metric,
        "n": len(a), "n_gt": len(b),
        "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
        "hist_overlap": histogram_overlap(a, b),
        "mean": float(a.mean()), "mean_gt": float(b.mean()),
        "delta": float(a.mean() - b.mean()),
        "passes_ks": bool(ks.pvalue >= KS_ALPHA),
    }


def main():
    pc_base = pd.read_parquet(OUTPUTS_DIR / "pitch_control.parquet")
    pc_soc = pd.read_parquet(OUTPUTS_DIR / "pitch_control_soccana.parquet")

    gt = pc_base[pc_base["track"] == "gt"]
    pipe = pc_base[pc_base["track"] == "pipeline"]
    print(f"gt frames      : {len(gt)}")
    print(f"yolov8x frames : {len(pipe)}")
    print(f"soccana frames : {len(pc_soc)}")

    rows = []
    for metric in METRICS:
        gt_a = gt[metric].dropna().to_numpy()
        rows.append(compare(pipe[metric].dropna().to_numpy(), gt_a, "yolov8x", metric))
        rows.append(compare(pc_soc[metric].dropna().to_numpy(), gt_a, "soccana", metric))

    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUTS_DIR / "ablation_ks_summary.parquet", index=False)

    # Pretty print
    pivot_cols = ["mean", "delta", "ks_stat", "ks_p", "hist_overlap", "passes_ks"]
    out_cols = ["detector", "metric"] + pivot_cols
    print()
    print(df[out_cols].round(4).to_string(index=False))

    # Render compact table figure
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(df) + 1.2))
    ax.axis("off")
    cell_text = df[out_cols].round(3).astype(str).values
    table = ax.table(cellText=cell_text, colLabels=out_cols,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    ax.set_title("Detector ablation: YOLOv8x vs Soccana (vs GT, KS at alpha=0.05)",
                 fontsize=11, pad=12)
    out_fig = FIGURES_DIR / "12_ablation_ks_table.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_fig}")

    # Headline summary
    base_pass = df[(df["detector"] == "yolov8x") & df["passes_ks"]]["metric"].tolist()
    soc_pass = df[(df["detector"] == "soccana") & df["passes_ks"]]["metric"].tolist()
    print(f"\nYOLOv8x passes KS (p>={KS_ALPHA}): {base_pass or 'none'}")
    print(f"Soccana passes KS (p>={KS_ALPHA}): {soc_pass or 'none'}")


if __name__ == "__main__":
    main()
