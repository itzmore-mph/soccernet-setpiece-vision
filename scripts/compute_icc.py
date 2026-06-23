"""Compute ICC(2,1) per Pitch Control metric and effective sample size.

Reads outputs/pitch_control_soccana_tvcalib.parquet, computes ICC(2,1) for each
PC metric using pingouin, derives n_eff, and saves outputs/icc_per_metric.parquet.

This script reads only committed parquets (no SoccerNet raw video needed).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg

# PC metrics to compute ICC for
PC_METRICS = ["pc_mean", "pc_at_ball", "pc_in_box", "pc_in_third", "pc_area_gt_0p5"]

# Paths
ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "outputs" / "pitch_control_soccana_tvcalib.parquet"
OUTPUT_PATH = ROOT / "outputs" / "icc_per_metric.parquet"
FIGURE_PATH = ROOT / "outputs" / "figures" / "icc_effective_sample_size.png"


def compute_icc_per_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ICC(2,1) and n_eff for each PC metric.

    Parameters
    ----------
    df : pd.DataFrame
        Pitch control data with columns: clip_id, frame_idx, and PC metric columns.

    Returns
    -------
    pd.DataFrame
        Results with columns: metric, icc_value, ci_lower, ci_upper, n_eff.
    """
    n_clips = df["clip_id"].nunique()

    results = []
    for metric in PC_METRICS:
        # Build long-format dataframe for pingouin
        long_df = df[["clip_id", "frame_idx", metric]].dropna(subset=[metric])

        # Compute ICC(2,1) — nan_policy='omit' handles unbalanced panels
        icc_table = pg.intraclass_corr(
            data=long_df,
            targets="clip_id",
            raters="frame_idx",
            ratings=metric,
            nan_policy="omit",
        )
        # ICC(2,1) corresponds to ICC(A,1) in pingouin: two-way random, absolute agreement
        icc_row = icc_table[icc_table["Type"] == "ICC(A,1)"].iloc[0]
        icc_value = icc_row["ICC"]
        ci_lower = icc_row["CI95"][0]
        ci_upper = icc_row["CI95"][1]

        # Standard design-effect formula: DEFF = 1 + (m_avg - 1) * ICC
        # n_eff = N_total / DEFF, where N_total = total paired frame observations
        N_total = len(long_df)
        m_avg = N_total / n_clips
        n_eff = N_total / (1 + (m_avg - 1) * icc_value)

        results.append(
            {
                "metric": metric,
                "icc_value": icc_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_eff": n_eff,
            }
        )

    return pd.DataFrame(results)


def plot_icc_figure(results: pd.DataFrame, output_path: Path) -> None:
    """Generate publication-quality horizontal bar chart of ICC values with CI and n_eff.

    Parameters
    ----------
    results : pd.DataFrame
        ICC results with columns: metric, icc_value, ci_lower, ci_upper, n_eff.
    output_path : Path
        File path for the saved figure (PNG).
    """
    # Sort by ICC value for visual clarity
    df = results.sort_values("icc_value", ascending=True).reset_index(drop=True)

    # Compute asymmetric error bars from CI
    xerr_lower = df["icc_value"] - df["ci_lower"]
    xerr_upper = df["ci_upper"] - df["icc_value"]
    xerr = [xerr_lower.values, xerr_upper.values]

    metric_labels = {
        "pc_mean": "PC Mean",
        "pc_at_ball": "PC at Ball",
        "pc_in_box": "PC in Box",
        "pc_in_third": "PC in Third",
        "pc_area_gt_0p5": "PC Area > 0.5",
    }
    labels = df["metric"].map(metric_labels).fillna(df["metric"])

    fig, ax = plt.subplots(figsize=(8, 4))

    # Horizontal bar chart
    y_pos = range(len(df))
    ax.barh(
        y_pos,
        df["icc_value"],
        xerr=xerr,
        height=0.6,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
        error_kw={"elinewidth": 1.2, "capthick": 1.2, "color": "#2d2d2d"},
    )

    # Annotate each bar with n_eff
    for i, (icc_val, n_eff, ci_hi) in enumerate(
        zip(df["icc_value"], df["n_eff"], df["ci_upper"])
    ):
        ax.text(
            ci_hi + 0.02,
            i,
            f"n_eff = {n_eff:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )

    # Formatting
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("ICC(2,1)", fontsize=11)
    ax.set_title("Intraclass Correlation & Effective Sample Size per Metric", fontsize=12, pad=10)
    ax.set_xlim(0, min(1.05, df["ci_upper"].max() + 0.18))
    ax.axvline(x=0.75, color="#999999", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(x=0.50, color="#cccccc", linestyle="--", linewidth=0.8, alpha=0.5)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def main() -> None:
    """Main entry point: read data, compute ICC, save results."""
    print(f"Reading pitch control data from {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    n_clips = df["clip_id"].nunique()
    N_total = len(df)
    m_avg = N_total / n_clips
    print(f"  Shape: {df.shape} | Clips: {n_clips} | N_total frames: {N_total} | m_avg: {m_avg:.2f}")

    print("\nComputing ICC(2,1) per metric...")
    results = compute_icc_per_metric(df)

    # Save to parquet
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved ICC results to {OUTPUT_PATH}")

    # Generate publication-quality figure
    plot_icc_figure(results, FIGURE_PATH)

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Metric':<20} {'ICC(2,1)':>10} {'CI Lower':>10} {'CI Upper':>10} {'n_eff':>10}")
    print("-" * 70)
    for _, row in results.iterrows():
        print(
            f"{row['metric']:<20} {row['icc_value']:>10.4f} {row['ci_lower']:>10.4f} "
            f"{row['ci_upper']:>10.4f} {row['n_eff']:>10.2f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
