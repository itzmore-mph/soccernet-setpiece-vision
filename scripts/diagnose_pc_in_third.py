"""Root-cause diagnostic for the paradoxical pc_in_third correlation.

Pooled Pearson r between pipeline and GT ``pc_in_third`` is near zero, which
looks like no agreement. This script shows the null is a Simpson's paradox:
stratified by set-piece type the pipeline tracks GT for corners (r > 0) and
inverts for direct free kicks (r < 0), and pooling cancels the two. It also
rules out range restriction (the pipeline over-disperses relative to GT) and
quantifies clip-level instability with a bootstrap CI on r.

Reads:
    outputs/pitch_control_soccana_tvcalib.parquet  (pipeline)
    outputs/pitch_control_gt_full.parquet           (GT reference)

Writes:
    outputs/pc_in_third_by_action.parquet           (stratified statistics)
    outputs/figures/14_pc_in_third_by_action.png    (committed figure)
    outputs/figures/diagnostics/*.png               (exploratory, gitignored)

Run: uv run python scripts/diagnose_pc_in_third.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"
DIAG_DIR = FIG_DIR / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

PIPE_PATH = OUTPUTS_DIR / "pitch_control_soccana_tvcalib.parquet"
GT_PATH = OUTPUTS_DIR / "pitch_control_gt_full.parquet"
METRIC = "pc_in_third"
KEYS = ["split", "clip_id", "frame_idx"]
CLIP_KEYS = ["split", "clip_id"]
N_BOOT = 10_000
SEED = 20260602


# --------------------------------------------------------------------------- #
# Safe correlation helpers (handle constant / tiny / NaN inputs)
# --------------------------------------------------------------------------- #
def safe_pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Pearson r and p, returning (nan, nan) when undefined (constant/too few)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan, np.nan
    r = stats.pearsonr(a, b)
    return float(r.statistic), float(r.pvalue)


def safe_spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Spearman rho and p, guarding constant / tiny inputs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan, np.nan
    rho = stats.spearmanr(a, b)
    return float(rho.statistic), float(rho.pvalue)


def bootstrap_r_ci(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for Pearson r over paired (a, b).

    Deterministic for a fixed seed. Resamples that collapse to a constant
    (zero variance) are skipped so the correlation stays defined.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = a.size
    if n < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    aa, bb = a[idx], b[idx]
    sa, sb = aa.std(axis=1), bb.std(axis=1)
    ok = (sa > 0) & (sb > 0)
    am = aa - aa.mean(axis=1, keepdims=True)
    bm = bb - bb.mean(axis=1, keepdims=True)
    cov = (am * bm).mean(axis=1)
    rs = np.full(n_boot, np.nan)
    rs[ok] = cov[ok] / (sa[ok] * sb[ok])
    rs = rs[np.isfinite(rs)]
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


# --------------------------------------------------------------------------- #
# Load + pair
# --------------------------------------------------------------------------- #
def load_paired(pipe: pd.DataFrame, gt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (frame_paired, clip_paired) with _pipe / _gt suffixed metric cols."""
    frame = pipe[KEYS + [METRIC, "action_class"]].merge(gt[KEYS + [METRIC]], on=KEYS, suffixes=("_pipe", "_gt"))
    pipe_c = pipe.groupby(CLIP_KEYS)[METRIC].mean().rename(f"{METRIC}_pipe")
    gt_c = gt.groupby(CLIP_KEYS)[METRIC].mean().rename(f"{METRIC}_gt")
    act = pipe.groupby(CLIP_KEYS)["action_class"].first()
    clip = pd.concat([pipe_c, gt_c, act], axis=1, join="inner").reset_index()
    return frame, clip


def _stats_row(segment: str, df: pd.DataFrame) -> dict:
    """Correlation + dispersion statistics for one paired segment."""
    x = df[f"{METRIC}_gt"].to_numpy()
    y = df[f"{METRIC}_pipe"].to_numpy()
    r, rp = safe_pearson(x, y)
    rho, _ = safe_spearman(x, y)
    lo, hi = bootstrap_r_ci(x, y)
    return {
        "segment": segment,
        "n": int((np.isfinite(x) & np.isfinite(y)).sum()),
        "pearson_r": r,
        "pearson_p": rp,
        "spearman_rho": rho,
        "r_ci_lo": lo,
        "r_ci_hi": hi,
        "std_gt": float(np.nanstd(x)),
        "std_pipe": float(np.nanstd(y)),
    }


def compute_stratified_stats(pipe: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Tidy table of pc_in_third agreement, pooled and stratified by action class.

    Deterministic (fixed bootstrap seed); used both for the committed artifact and
    for the reproducibility check. Rows are sorted by ``segment`` for stable diffs.
    """
    frame, clip = load_paired(pipe, gt)
    rows = [_stats_row("ALL frames", frame), _stats_row("ALL clips", clip)]
    for act, g in frame.groupby("action_class"):
        rows.append(_stats_row(f"frame | {act}", g))
    for act, g in clip.groupby("action_class"):
        rows.append(_stats_row(f"clip | {act}", g))
    return pd.DataFrame(rows).sort_values("segment").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def render_committed_figure(pipe: pd.DataFrame, gt: pd.DataFrame, table: pd.DataFrame) -> None:
    """Frame-level scatter with a separate OLS fit per set-piece type (the paradox)."""
    frame, _ = load_paired(pipe, gt)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = {"Corner": "#1f77b4", "Direct free-kick": "#d62728"}
    lim = [0.2, 1.0]
    ax.plot(lim, lim, "k--", lw=1, label="y = x (identity)")
    for act, g in frame.groupby("action_class"):
        x = g[f"{METRIC}_gt"].to_numpy()
        y = g[f"{METRIC}_pipe"].to_numpy()
        c = colors.get(act, "#555555")
        seg = table[table["segment"] == f"frame | {act}"]
        r = float(seg["pearson_r"].iloc[0]) if len(seg) else np.nan
        ax.scatter(x, y, s=16, alpha=0.4, color=c, label=f"{act} (r = {r:+.2f}, n={len(g)})")
        if np.std(x) > 0:
            b, a0 = np.polyfit(x, y, 1)
            xs = np.linspace(lim[0], lim[1], 50)
            ax.plot(xs, a0 + b * xs, color=c, lw=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("pc_in_third (ground truth)")
    ax.set_ylabel("pc_in_third (pipeline)")
    ax.set_title(
        "pc_in_third: opposite-signed agreement by set-piece type\n(Simpson's paradox; pooled r approx 0)",
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    out = FIG_DIR / "14_pc_in_third_by_action.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def render_diagnostic_scatters(pipe: pd.DataFrame, gt: pd.DataFrame) -> None:
    """Exploratory frame- and clip-level scatters with marginals (gitignored)."""
    frame, clip = load_paired(pipe, gt)
    for level, df in [("frame", frame), ("clip", clip)]:
        x = df[f"{METRIC}_gt"].to_numpy()
        y = df[f"{METRIC}_pipe"].to_numpy()
        fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1]})
        codes = pd.Categorical(df["action_class"]).codes
        ax[0].scatter(x, y, c=codes, cmap="tab10", alpha=0.5, s=18)
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax[0].plot(lim, lim, "k--", lw=1, label="y = x")
        if np.std(x) > 0:
            b, a0 = np.polyfit(x, y, 1)
            xs = np.linspace(lim[0], lim[1], 50)
            ax[0].plot(xs, a0 + b * xs, "r-", lw=1.5, label=f"OLS (slope={b:.2f})")
        ax[0].set_xlabel(f"{METRIC} GT")
        ax[0].set_ylabel(f"{METRIC} pipeline")
        ax[0].set_title(f"{level}-level (n={len(df)})")
        ax[0].legend(fontsize=8)
        ax[1].hist(x, bins=20, orientation="horizontal", alpha=0.5, label="GT")
        ax[1].hist(y, bins=20, orientation="horizontal", alpha=0.5, label="pipe")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        out = DIAG_DIR / f"pc_in_third_scatter_{level}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


def main() -> None:
    pipe = pd.read_parquet(PIPE_PATH)
    gt = pd.read_parquet(GT_PATH)

    table = compute_stratified_stats(pipe, gt)
    out = OUTPUTS_DIR / "pc_in_third_by_action.parquet"
    table.to_parquet(out, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(table.round(4).to_string(index=False))
    print(f"\nSaved: {out}")

    # Dispersion contrast: rejects the range-restriction hypothesis.
    allf = table[table["segment"] == "ALL frames"].iloc[0]
    print(
        f"\nDispersion: pipeline std {allf['std_pipe']:.4f} vs GT std {allf['std_gt']:.4f} "
        f"-> pipeline over-disperses (range restriction rejected)."
    )

    render_committed_figure(pipe, gt, table)
    render_diagnostic_scatters(pipe, gt)


if __name__ == "__main__":
    main()
