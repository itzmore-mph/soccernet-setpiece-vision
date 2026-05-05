"""Detector ablation: compare YOLOv8x vs Soccana (YOLOv11n football-finetuned).

Reads:
    outputs/detections_pipeline.parquet   (YOLOv8x, baseline)
    outputs/detections_soccana.parquet    (Soccana, ablation - run run_soccana_ablation.py first)
    outputs/detections_gt.parquet         (SoccerNet GSR ground truth)

Produces:
    outputs/ablation_detector_summary.parquet
    outputs/figures/11_ablation_detector_counts.png

Comparison axes:
    - mean detections per frame, overall and inside penalty box (where bias originates)
    - delta vs GT (signed)
    - per-clip distribution overlap
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Penalty-box bounds in metric pitch (origin top-left, 105x68)
PB_X_LEFT = 16.5
PB_X_RIGHT = 105.0 - 16.5
PB_Y_TOP = 34.0 - 40.32 / 2
PB_Y_BOT = 34.0 + 40.32 / 2


def in_box(df: pd.DataFrame) -> pd.Series:
    in_left = (df["x_m"] <= PB_X_LEFT) & df["y_m"].between(PB_Y_TOP, PB_Y_BOT)
    in_right = (df["x_m"] >= PB_X_RIGHT) & df["y_m"].between(PB_Y_TOP, PB_Y_BOT)
    return in_left | in_right


def per_frame_counts(df: pd.DataFrame, label: str) -> pd.DataFrame:
    g = df.groupby(["clip_id", "frame_idx"]).size().rename("n_total").reset_index()
    box = df[in_box(df)].groupby(["clip_id", "frame_idx"]).size().rename("n_box").reset_index()
    out = g.merge(box, on=["clip_id", "frame_idx"], how="left").fillna({"n_box": 0})
    out["detector"] = label
    return out


def summarize(counts: pd.DataFrame) -> dict:
    return {
        "n_frames": len(counts),
        "mean_total": counts["n_total"].mean(),
        "median_total": counts["n_total"].median(),
        "mean_box": counts["n_box"].mean(),
        "median_box": counts["n_box"].median(),
    }


def main():
    base = pd.read_parquet(OUTPUTS_DIR / "detections_pipeline.parquet")
    soc = pd.read_parquet(OUTPUTS_DIR / "detections_soccana.parquet")
    gt = pd.read_parquet(OUTPUTS_DIR / "detections_gt.parquet")

    counts_base = per_frame_counts(base, "yolov8x")
    counts_soc = per_frame_counts(soc, "soccana")
    counts_gt = per_frame_counts(gt, "gt")

    rows = []
    for label, c in [("yolov8x", counts_base), ("soccana", counts_soc), ("gt", counts_gt)]:
        s = summarize(c)
        s["detector"] = label
        rows.append(s)
    summary = pd.DataFrame(rows).set_index("detector")
    summary["delta_total_vs_gt"] = summary["mean_total"] - summary.loc["gt", "mean_total"]
    summary["delta_box_vs_gt"] = summary["mean_box"] - summary.loc["gt", "mean_box"]
    print(summary.round(3))

    summary.to_parquet(OUTPUTS_DIR / "ablation_detector_summary.parquet")

    # Figure: distribution of n_box per frame
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.arange(0, max(counts_base["n_total"].max(),
                            counts_soc["n_total"].max(),
                            counts_gt["n_total"].max()) + 2)
    for ax, col, title in [(axes[0], "n_total", "Detections per frame (full pitch)"),
                           (axes[1], "n_box", "Detections per frame (penalty boxes)")]:
        ax.hist(counts_gt[col], bins=bins, alpha=0.5, label="GT", color="#2ca02c")
        ax.hist(counts_base[col], bins=bins, alpha=0.5, label="YOLOv8x", color="#1f77b4")
        ax.hist(counts_soc[col], bins=bins, alpha=0.5, label="Soccana", color="#d62728")
        ax.set_xlabel(col)
        ax.set_ylabel("frames")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    out_fig = FIGURES_DIR / "11_ablation_detector_counts.png"
    fig.savefig(out_fig, dpi=150)
    print(f"Saved figure: {out_fig}")


if __name__ == "__main__":
    main()
