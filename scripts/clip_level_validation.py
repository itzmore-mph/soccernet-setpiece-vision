"""Clip-level validation: the inferential unit the ICC/n_eff analysis demands.

The frame-level tables in ``ks_table_tvcalib.py`` treat 651 pipeline frames as
independent, but ICC(2,1) = 0.89-0.93 shows the ~31 frames of each set-piece clip
are near-replicates (n_eff ~ 22-24, roughly one observation per clip). Testing on
frames is therefore pseudoreplication. This script collapses each clip to a single
value per metric (the within-clip mean), pairs the 21 clips common to the pipeline
and GT cohorts, and runs the validation at the clip level:

  - paired bias (mean of per-clip pipeline-minus-GT differences),
  - Wilcoxon signed-rank test (paired, distribution-free; appropriate at n=21),
  - Pearson and Spearman correlation on the 21 clip means,
  - a percentile bootstrap 95% CI on the bias (deterministic seed for reproducibility).

Reads:
    outputs/pitch_control_soccana_tvcalib.parquet  (pipeline, 21 clips)
    outputs/pitch_control_gt_full.parquet           (GT reference, 31 clips)

Writes:
    outputs/clip_level_validation.parquet
    outputs/figures/13_clip_level_validation.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Reuse the canonical metric list so frame-level and clip-level tables stay aligned.
from ks_table_tvcalib import METRICS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CLIP_KEYS = ["split", "clip_id"]
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260602  # fixed so the CI reproduces byte-stably in CI
CI_ALPHA = 0.05


def clip_means(pc: pd.DataFrame) -> pd.DataFrame:
    """Collapse a per-frame PC table to one mean value per clip and metric.

    Parameters
    ----------
    pc : pd.DataFrame
        Per-frame pitch-control table with ``split``, ``clip_id`` and the
        columns listed in :data:`METRICS`.

    Returns
    -------
    pd.DataFrame
        One row per clip, indexed by ``(split, clip_id)``, with a column per metric.
    """
    return pc.groupby(CLIP_KEYS)[METRICS].mean()


def bootstrap_bias_ci(
    diffs: np.ndarray, n_boot: int = BOOTSTRAP_N, alpha: float = CI_ALPHA, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean paired difference (bias).

    Parameters
    ----------
    diffs : np.ndarray
        Per-clip pipeline-minus-GT differences for one metric.
    n_boot : int
        Number of bootstrap resamples.
    alpha : float
        Two-sided significance level; the CI spans ``[alpha/2, 1-alpha/2]``.
    seed : int
        RNG seed; fixed for reproducibility.

    Returns
    -------
    tuple[float, float]
        Lower and upper percentile-bootstrap bounds on the bias.
    """
    rng = np.random.default_rng(seed)
    n = diffs.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def compare_clip_level(pipe: pd.Series, gt: pd.Series, metric: str) -> dict:
    """Clip-level paired comparison for a single metric.

    Parameters
    ----------
    pipe, gt : pd.Series
        Per-clip mean values for the pipeline and GT cohorts, indexed identically.
    metric : str
        Metric name (carried into the output row).

    Returns
    -------
    dict
        Paired statistics: bias, bootstrap CI, Wilcoxon p, Pearson, Spearman.
    """
    diffs = (pipe - gt).to_numpy()
    n = diffs.shape[0]
    bias = float(diffs.mean())
    ci_lo, ci_hi = bootstrap_bias_ci(diffs)

    # Wilcoxon needs at least one non-zero difference; guard the degenerate case.
    if np.allclose(diffs, 0.0):
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = float(stats.wilcoxon(pipe.to_numpy(), gt.to_numpy()).pvalue)

    pearson = float(stats.pearsonr(pipe.to_numpy(), gt.to_numpy()).statistic)
    spearman = float(stats.spearmanr(pipe.to_numpy(), gt.to_numpy()).statistic)

    return {
        "metric": metric,
        "n_clips": n,
        "mean_pipe": float(pipe.mean()),
        "mean_gt": float(gt.mean()),
        "bias": bias,
        "bias_ci_lo": ci_lo,
        "bias_ci_hi": ci_hi,
        "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
        "wilcoxon_p": wilcoxon_p,
        "pearson": pearson,
        "spearman": spearman,
    }


def main() -> None:
    pc_pipe = pd.read_parquet(OUTPUTS_DIR / "pitch_control_soccana_tvcalib.parquet")
    pc_gt = pd.read_parquet(OUTPUTS_DIR / "pitch_control_gt_full.parquet")

    pipe_c = clip_means(pc_pipe)
    gt_c = clip_means(pc_gt)

    # Pair on the clips common to both cohorts (pipeline is the limiting subset).
    common = pipe_c.index.intersection(gt_c.index)
    pipe_c = pipe_c.loc[common].sort_index()
    gt_c = gt_c.loc[common].sort_index()
    print(f"pipeline clips: {len(pipe_c)}  |  GT clips: {len(gt_c)}  |  matched: {len(common)}")

    rows = [compare_clip_level(pipe_c[m], gt_c[m], m) for m in METRICS]
    df = pd.DataFrame(rows)
    out = OUTPUTS_DIR / "clip_level_validation.parquet"
    df.to_parquet(out, index=False)

    cols = [
        "metric",
        "n_clips",
        "mean_pipe",
        "mean_gt",
        "bias",
        "bias_ci_lo",
        "bias_ci_hi",
        "wilcoxon_p",
        "pearson",
        "spearman",
    ]
    print()
    print(df[cols].round(4).to_string(index=False))
    print(f"\nSaved: {out}")

    _render_forest(df)


def _render_forest(df: pd.DataFrame) -> None:
    """Forest plot of clip-level bias with bootstrap 95% CIs per metric."""
    metric_display = {
        "pc_mean": "PC Mean",
        "pc_at_ball": "PC at Ball",
        "pc_in_box": "PC in Box",
        "pc_in_third": "PC in Third",
        "pc_area_gt_0p5": "PC Area > 0.5",
    }
    order = df.iloc[::-1].reset_index(drop=True)  # top metric at the top of the plot
    y = np.arange(len(order))
    bias = order["bias"].to_numpy()
    lo = order["bias_ci_lo"].to_numpy()
    hi = order["bias_ci_hi"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 0.7 * len(order) + 1.5))
    ax.axvline(0.0, color="0.5", linestyle="--", linewidth=1, zorder=1)
    ax.errorbar(
        bias,
        y,
        xerr=[bias - lo, hi - bias],
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        capsize=4,
        markersize=7,
        zorder=2,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([metric_display.get(m, m) for m in order["metric"]])
    ax.set_xlabel("Clip-level bias (pipeline minus GT), with percentile bootstrap 95% CI")
    ax.set_title(f"Clip-level validation (n={int(df['n_clips'].iloc[0])} matched clips)", fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    out_fig = FIGURES_DIR / "13_clip_level_validation.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_fig}")


if __name__ == "__main__":
    main()
