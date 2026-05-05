"""Insert detector ablation section into nb04 between section 5 (verdict) and
the final Conclusion markdown, and extend the Conclusion to reference the
ablation findings. Idempotent on both edits.
"""

from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "04_evaluation_and_validation.ipynb"
MARKER = "## 6. Detector Ablation"

MD_SRC = '''## 6. Detector Ablation — YOLOv8x vs Soccana

Replaces the COCO-pretrained YOLOv8x detector with **Soccana** (`Adit-jain/soccana` on HuggingFace), a YOLOv11n model fine-tuned on SoccerNet plus additional football footage. All other pipeline stages (homography, ByteTrack, KMeans-HSV team assignment, Laurie Shaw pitch-control model) are held constant.

Computed offline by `scripts/run_soccana_ablation.py` and `scripts/run_pc_soccana.py`. This cell loads the resulting Parquet and reports the same KS / overlap diagnostics already used for the YOLOv8x baseline.
'''

CODE_SRC = '''pc_soc_path = OUTPUTS_DIR / "pitch_control_soccana.parquet"
if not pc_soc_path.is_file():
    print("pitch_control_soccana.parquet not found.")
    print("Run scripts/run_soccana_ablation.py then scripts/run_pc_soccana.py to generate it.")
else:
    pc_soc = pd.read_parquet(pc_soc_path)
    gt_only = pc[pc["track"] == "gt"]
    pipe_only = pc[pc["track"] == "pipeline"]
    print(f"gt frames      : {len(gt_only)}")
    print(f"yolov8x frames : {len(pipe_only)}")
    print(f"soccana frames : {len(pc_soc)}")

    def _row(arr_a, arr_gt, label, metric):
        if len(arr_a) < 3 or len(arr_gt) < 3:
            return None
        ks = stats.ks_2samp(arr_a, arr_gt)
        return {
            "detector": label, "metric": metric,
            "n": len(arr_a), "n_gt": len(arr_gt),
            "mean": float(arr_a.mean()),
            "delta": float(arr_a.mean() - arr_gt.mean()),
            "ks_stat": float(ks.statistic),
            "ks_p": float(ks.pvalue),
            "hist_overlap": histogram_overlap(arr_a, arr_gt),
            "passes_ks": bool(ks.pvalue >= KS_ALPHA),
        }

    rows = []
    for metric in METRICS:
        gt_a = gt_only[metric].dropna().to_numpy()
        for label, df_d in [("yolov8x", pipe_only), ("soccana", pc_soc)]:
            r = _row(df_d[metric].dropna().to_numpy(), gt_a, label, metric)
            if r is not None:
                rows.append(r)
    ablation = pd.DataFrame(rows)

    # Side-by-side per metric
    pivot_delta = ablation.pivot(index="metric", columns="detector", values="delta").round(4)
    pivot_ks = ablation.pivot(index="metric", columns="detector", values="ks_p").round(4)
    pivot_overlap = ablation.pivot(index="metric", columns="detector", values="hist_overlap").round(4)
    print("\\nBias (mean - GT mean):")
    print(pivot_delta)
    print("\\nKS p-value:")
    print(pivot_ks)
    print("\\nHistogram overlap:")
    print(pivot_overlap)

    # Bias-reduction summary
    bias_red = (pivot_delta["yolov8x"].abs() - pivot_delta["soccana"].abs()) / pivot_delta["yolov8x"].abs()
    print("\\nRelative |bias| reduction (yolov8x -> soccana):")
    print((bias_red * 100).round(1).astype(str) + " %")

    # Save and figure
    ablation.to_parquet(OUTPUTS_DIR / "ablation_ks_summary.parquet", index=False)

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 4), sharey=True)
    for ax, metric in zip(axes, METRICS):
        edges = np.linspace(0, 1, HIST_BINS + 1)
        ax.hist(gt_only[metric].dropna(), bins=edges, alpha=0.5, label="GT", color="#2ca02c")
        ax.hist(pipe_only[metric].dropna(), bins=edges, alpha=0.5, label="YOLOv8x", color="#1f77b4")
        ax.hist(pc_soc[metric].dropna(), bins=edges, alpha=0.5, label="Soccana", color="#d62728")
        ax.set_title(metric)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
    fig.suptitle("Detector ablation: GT vs YOLOv8x vs Soccana")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "12_ablation_histograms.png", dpi=150)
    plt.show()
'''


def main():
    nb = json.loads(NB.read_text())
    cells = nb["cells"]
    if any(MARKER in "".join(c.get("source", [])) for c in cells):
        print("Ablation section already present, skipping.")
        return

    # Insert before the final Conclusion markdown (search by header)
    insert_at = None
    for i, c in enumerate(cells):
        if c["cell_type"] == "markdown" and "### Conclusion" in "".join(c["source"]):
            insert_at = i
            break
    if insert_at is None:
        insert_at = len(cells)
        print("No Conclusion cell found, appending at end.")

    new_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": MD_SRC.splitlines(keepends=True),
    }
    new_code = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": CODE_SRC.splitlines(keepends=True),
    }
    cells.insert(insert_at, new_md)
    cells.insert(insert_at + 1, new_code)

    NB.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"Inserted ablation cells at position {insert_at} in {NB}")


if __name__ == "__main__":
    main()
