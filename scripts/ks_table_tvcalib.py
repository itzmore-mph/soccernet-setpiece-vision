"""Validation table: Soccana + TVCalib pipeline vs GT (33-clip cohort).

Compares the primary pipeline (Soccana detector, TVCalib autonomous H) against
SoccerNet GSR ground-truth pitch control over the full 33-clip set-piece cohort.

Reads:
    outputs/pitch_control_soccana_tvcalib.parquet  (pipeline, 33 clips)
    outputs/pitch_control_gt_full.parquet           (GT reference, 33 clips)

Writes:
    outputs/validation_summary_tvcalib.parquet
    outputs/figures/14_ks_table_tvcalib.png
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


def hist_overlap(a, b, bins=HIST_BINS, lo=0.0, hi=1.0):
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)
    pa = pa / pa.sum() if pa.sum() else pa
    pb = pb / pb.sum() if pb.sum() else pb
    return float(np.minimum(pa, pb).sum())


def compare(a, b, metric):
    if len(a) < 3 or len(b) < 3:
        return {"metric": metric, "n": len(a), "n_gt": len(b),
                "ks_stat": np.nan, "ks_p": np.nan, "hist_overlap": np.nan,
                "mean": np.nan, "mean_gt": np.nan, "delta": np.nan, "passes_ks": False}
    ks = stats.ks_2samp(a, b)
    return {
        "metric": metric,
        "n": len(a), "n_gt": len(b),
        "mean": float(a.mean()), "mean_gt": float(b.mean()),
        "delta": float(a.mean() - b.mean()),
        "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
        "hist_overlap": hist_overlap(a, b),
        "passes_ks": bool(ks.pvalue >= KS_ALPHA),
    }


def main():
    pc_pipe = pd.read_parquet(OUTPUTS_DIR / "pitch_control_soccana_tvcalib.parquet")
    pc_gt = pd.read_parquet(OUTPUTS_DIR / "pitch_control_gt_full.parquet")

    print(f"pipeline frames: {len(pc_pipe)}  ({pc_pipe['clip_id'].nunique()} clips)")
    print(f"GT frames      : {len(pc_gt)}  ({pc_gt['clip_id'].nunique()} clips)")

    rows = []
    for metric in METRICS:
        pipe_a = pc_pipe[metric].dropna().to_numpy()
        gt_a = pc_gt[metric].dropna().to_numpy()
        rows.append(compare(pipe_a, gt_a, metric))

    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUTS_DIR / "validation_summary_tvcalib.parquet", index=False)

    cols = ["metric", "mean", "mean_gt", "delta", "ks_stat", "ks_p", "hist_overlap", "passes_ks"]
    print()
    print(df[cols].round(4).to_string(index=False))

    # Render table figure
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(df) + 1.4))
    ax.axis("off")
    cell = df[cols].round(3).astype(str).values
    table = ax.table(cellText=cell, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    ax.set_title(
        f"Soccana + TVCalib vs GT (33 clips, KS alpha={KS_ALPHA})",
        fontsize=11, pad=12,
    )
    out_fig = FIGURES_DIR / "14_ks_table_tvcalib.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_fig}")

    passes = df[df["passes_ks"]]["metric"].tolist()
    print(f"\nPasses KS (p>={KS_ALPHA}): {passes or 'none'}")
    print(f"Mean |delta|: {df['delta'].abs().mean():.4f}")
    print(f"Mean hist overlap: {df['hist_overlap'].mean():.3f}")


if __name__ == "__main__":
    main()
