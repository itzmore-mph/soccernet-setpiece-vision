"""Supplementary validation analyses for the Pitch Control pipeline.

Adds the agreement and context diagnostics that the core validation lacked:
  - Bland-Altman bias and 95% limits of agreement per metric,
  - skill score vs a constant GT-mean baseline (does the pipeline beat trivial?),
  - absolute error binned by detected-defender shortfall (the recall story),
  - box-control confusion matrix (the pc_in_box sign inversion as a classifier),
  - frame-to-frame temporal step size, pipeline vs GT.

All statistics are deterministic (no resampling), so they reproduce exactly.

Reads:
    outputs/pitch_control_soccana_tvcalib.parquet  (pipeline)
    outputs/pitch_control_gt_full.parquet           (GT reference)

Writes:
    outputs/validation_extras.parquet               (tidy long table)
    outputs/figures/15_bland_altman.png
    outputs/figures/16_validation_context.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

KEYS = ["split", "clip_id", "frame_idx"]
CLIP_KEYS = ["split", "clip_id"]
METRICS = ["pc_mean", "pc_at_ball", "pc_in_box", "pc_in_third", "pc_area_gt_0p5"]
DENSITY_METRIC = "pc_mean"
TEMPORAL_METRIC = "pc_mean"
DENSITY_BINS = [-99, -1, 0, 2, 4, 99]
DENSITY_LABELS = ["over", "exact", "1-2 short", "3-4 short", "5+ short"]


def load_paired(pipe: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Inner-join pipeline and GT on frame keys, suffixing metric columns."""
    cols = KEYS + METRICS + ["n_defenders"]
    return pipe[cols].merge(gt[cols], on=KEYS, suffixes=("_pipe", "_gt"))


def bland_altman_stats(paired: pd.DataFrame) -> pd.DataFrame:
    """Bias and 95% limits of agreement (bias +/- 1.96 SD of differences) per metric."""
    rows = []
    for m in METRICS:
        a = paired[f"{m}_pipe"].to_numpy(float)
        b = paired[f"{m}_gt"].to_numpy(float)
        mask = np.isfinite(a) & np.isfinite(b)
        diff = a[mask] - b[mask]
        n = diff.size
        bias = float(diff.mean()) if n else np.nan
        sd = float(diff.std(ddof=1)) if n > 1 else np.nan
        rows.append(
            {
                "metric": m,
                "n": n,
                "bias": bias,
                "loa_lo": bias - 1.96 * sd if n > 1 else np.nan,
                "loa_hi": bias + 1.96 * sd if n > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def skill_scores(paired: pd.DataFrame) -> pd.DataFrame:
    """Pipeline MAE vs a constant GT-mean predictor; skill = 1 - MAE_pipe / MAE_base."""
    rows = []
    for m in METRICS:
        a = paired[f"{m}_pipe"].to_numpy(float)
        b = paired[f"{m}_gt"].to_numpy(float)
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        mae_pipe = float(np.mean(np.abs(a - b))) if a.size else np.nan
        mae_base = float(np.mean(np.abs(b.mean() - b))) if b.size else np.nan
        skill = 1.0 - mae_pipe / mae_base if mae_base and mae_base > 0 else np.nan
        rows.append({"metric": m, "mae_pipe": mae_pipe, "mae_baseline": mae_base, "skill": skill})
    return pd.DataFrame(rows)


def error_by_density(paired: pd.DataFrame, metric: str = DENSITY_METRIC) -> pd.DataFrame:
    """Absolute error of one metric binned by detected-defender shortfall."""
    d = paired.copy()
    d["abs_err"] = (d[f"{metric}_pipe"] - d[f"{metric}_gt"]).abs()
    d["def_shortfall"] = d["n_defenders_gt"] - d["n_defenders_pipe"]
    d["bin"] = pd.cut(d["def_shortfall"], bins=DENSITY_BINS, labels=DENSITY_LABELS)
    g = d.groupby("bin", observed=False)["abs_err"].agg(["mean", "count"]).reset_index()
    g = g.rename(columns={"bin": "shortfall_bin"})
    g["shortfall_bin"] = g["shortfall_bin"].astype(str)
    return g


def box_control_confusion(paired: pd.DataFrame) -> pd.DataFrame:
    """2x2 counts: does each side judge the attacker to control the box (pc_in_box > 0.5)?"""
    gt_atk = paired["pc_in_box_gt"] > 0.5
    pipe_atk = paired["pc_in_box_pipe"] > 0.5
    rows = []
    for gt_label, gt_mask in [("GT_atk", gt_atk), ("GT_def", ~gt_atk)]:
        for pipe_label, pipe_mask in [("pipe_atk", pipe_atk), ("pipe_def", ~pipe_atk)]:
            rows.append({"gt": gt_label, "pipe": pipe_label, "count": int((gt_mask & pipe_mask).sum())})
    return pd.DataFrame(rows)


def temporal_summary(pipe: pd.DataFrame, gt: pd.DataFrame, metric: str = TEMPORAL_METRIC) -> pd.DataFrame:
    """Mean absolute frame-to-frame change of one metric, pipeline vs GT (stability)."""

    def step(df: pd.DataFrame) -> float:
        d = df.sort_values(KEYS).copy()
        d["delta"] = d.groupby(CLIP_KEYS)[metric].diff().abs()
        per_clip = d.groupby(CLIP_KEYS)["delta"].mean()
        return float(per_clip.mean())

    return pd.DataFrame(
        [
            {"source": "pipeline", "metric": metric, "mean_abs_step": step(pipe)},
            {"source": "gt", "metric": metric, "mean_abs_step": step(gt)},
        ]
    )


def build_extras_table(pipe: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Combine all extras into one deterministic tidy long table for the committed artifact."""
    paired = load_paired(pipe, gt)
    rows = []

    for _, r in bland_altman_stats(paired).iterrows():
        for col in ("bias", "loa_lo", "loa_hi", "n"):
            rows.append({"analysis": "bland_altman", "row": r["metric"], "col": col, "value": float(r[col])})
    for _, r in skill_scores(paired).iterrows():
        for col in ("mae_pipe", "mae_baseline", "skill"):
            rows.append({"analysis": "skill", "row": r["metric"], "col": col, "value": float(r[col])})
    for _, r in error_by_density(paired).iterrows():
        rows.append(
            {"analysis": "density", "row": r["shortfall_bin"], "col": "abs_err_mean", "value": float(r["mean"])}
        )
        rows.append({"analysis": "density", "row": r["shortfall_bin"], "col": "count", "value": float(r["count"])})
    for _, r in box_control_confusion(paired).iterrows():
        rows.append({"analysis": "box_confusion", "row": r["gt"], "col": r["pipe"], "value": float(r["count"])})
    for _, r in temporal_summary(pipe, gt).iterrows():
        rows.append(
            {"analysis": "temporal", "row": r["source"], "col": "mean_abs_step", "value": float(r["mean_abs_step"])}
        )

    return pd.DataFrame(rows).sort_values(["analysis", "row", "col"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def render_bland_altman(paired: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for i, m in enumerate(METRICS):
        ax = axes.flat[i]
        a = paired[f"{m}_pipe"].to_numpy(float)
        b = paired[f"{m}_gt"].to_numpy(float)
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        mean, diff = (a + b) / 2.0, a - b
        bias, sd = diff.mean(), diff.std(ddof=1)
        ax.scatter(mean, diff, s=12, alpha=0.35)
        ax.axhline(bias, color="k", lw=1.2, label=f"bias {bias:+.3f}")
        ax.axhline(bias + 1.96 * sd, color="r", ls="--", lw=1, label=f"+1.96SD {bias + 1.96 * sd:+.3f}")
        ax.axhline(bias - 1.96 * sd, color="r", ls="--", lw=1, label=f"-1.96SD {bias - 1.96 * sd:+.3f}")
        ax.set_title(m)
        ax.set_xlabel("mean(pipe, GT)")
        ax.set_ylabel("pipe - GT")
        ax.legend(fontsize=7)
    axes.flat[-1].set_visible(False)
    fig.suptitle("Bland-Altman agreement, pipeline vs GT (frame level)", fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "15_bland_altman.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def render_context(paired: pd.DataFrame) -> None:
    skill = skill_scores(paired)
    dens = error_by_density(paired)
    conf = box_control_confusion(paired)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    ax[0].bar(skill["metric"], skill["skill"], color="#1f77b4")
    ax[0].axhline(0, color="k", lw=1)
    ax[0].set_title("Skill score vs GT-mean baseline\n(>0 beats trivial predictor)")
    ax[0].set_ylabel("skill = 1 - MAE_pipe / MAE_base")
    ax[0].tick_params(axis="x", rotation=45)

    ax[1].bar(dens["shortfall_bin"], dens["mean"], color="#ff7f0e")
    ax[1].set_title(f"Mean |error| ({DENSITY_METRIC}) by defender shortfall")
    ax[1].set_xlabel("GT minus pipeline defenders")
    ax[1].set_ylabel("mean abs error")
    ax[1].tick_params(axis="x", rotation=45)

    mat = conf.pivot(index="gt", columns="pipe", values="count")
    im = ax[2].imshow(mat.to_numpy(), cmap="Blues")
    ax[2].set_xticks(range(mat.shape[1]))
    ax[2].set_xticklabels(mat.columns)
    ax[2].set_yticks(range(mat.shape[0]))
    ax[2].set_yticklabels(mat.index)
    for (r, c), v in np.ndenumerate(mat.to_numpy()):
        ax[2].text(c, r, str(int(v)), ha="center", va="center")
    ax[2].set_title("Box control confusion\n(pc_in_box > 0.5 = attacker)")
    fig.colorbar(im, ax=ax[2], fraction=0.046)

    fig.tight_layout()
    out = FIG_DIR / "16_validation_context.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    pipe = pd.read_parquet(OUTPUTS_DIR / "pitch_control_soccana_tvcalib.parquet")
    gt = pd.read_parquet(OUTPUTS_DIR / "pitch_control_gt_full.parquet")

    table = build_extras_table(pipe, gt)
    out = OUTPUTS_DIR / "validation_extras.parquet"
    table.to_parquet(out, index=False)

    pd.set_option("display.width", 200, "display.max_rows", 100)
    print(table.round(4).to_string(index=False))
    print(f"\nSaved: {out}")

    paired = load_paired(pipe, gt)
    render_bland_altman(paired)
    render_context(paired)


if __name__ == "__main__":
    main()
