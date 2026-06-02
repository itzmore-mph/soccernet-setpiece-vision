"""Unit tests for the supplementary validation extras."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from validation_extras import (  # noqa: E402
    bland_altman_stats,
    box_control_confusion,
    build_extras_table,
    error_by_density,
    skill_scores,
    temporal_summary,
)

METRICS = ["pc_mean", "pc_at_ball", "pc_in_box", "pc_in_third", "pc_area_gt_0p5"]


def _toy(n_per_clip: int = 6, n_clips: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(3)
    pipe_rows, gt_rows = [], []
    for c in range(n_clips):
        for f in range(n_per_clip):
            key = {"split": "train", "clip_id": f"c{c}", "frame_idx": f}
            gt_vals = {m: float(rng.uniform(0.3, 0.7)) for m in METRICS}
            pipe_vals = {m: float(np.clip(gt_vals[m] + rng.normal(scale=0.05), 0, 1)) for m in METRICS}
            gt_rows.append({**key, **gt_vals, "n_defenders": 10})
            pipe_rows.append({**key, **pipe_vals, "n_defenders": int(rng.integers(6, 11))})
    return pd.DataFrame(pipe_rows), pd.DataFrame(gt_rows)


def _paired(pipe, gt):
    cols = ["split", "clip_id", "frame_idx"] + METRICS + ["n_defenders"]
    return pipe[cols].merge(gt[cols], on=["split", "clip_id", "frame_idx"], suffixes=("_pipe", "_gt"))


def test_bland_altman_loa_brackets_bias():
    pipe, gt = _toy()
    df = bland_altman_stats(_paired(pipe, gt))
    assert set(df["metric"]) == set(METRICS)
    for _, r in df.iterrows():
        assert r["loa_lo"] <= r["bias"] <= r["loa_hi"]


def test_skill_score_zero_when_pipeline_equals_gt():
    """If pipeline equals GT, MAE_pipe is 0 so skill is 1.0 (perfect)."""
    pipe, gt = _toy()
    for m in METRICS:
        pipe[m] = gt[m]
    df = skill_scores(_paired(pipe, gt))
    assert np.allclose(df["skill"].to_numpy(), 1.0)


def test_error_by_density_bins_present_and_counts_sum():
    pipe, gt = _toy()
    paired = _paired(pipe, gt)
    df = error_by_density(paired)
    assert df["count"].sum() == len(paired)


def test_box_control_confusion_counts_sum_to_n():
    pipe, gt = _toy()
    paired = _paired(pipe, gt)
    df = box_control_confusion(paired)
    assert df["count"].sum() == len(paired)
    assert set(df["gt"]) == {"GT_atk", "GT_def"}


def test_temporal_summary_non_negative_steps():
    pipe, gt = _toy()
    df = temporal_summary(pipe, gt)
    assert (df["mean_abs_step"] >= 0).all()
    assert set(df["source"]) == {"pipeline", "gt"}


def test_build_extras_table_is_deterministic_and_long():
    pipe, gt = _toy()
    t1 = build_extras_table(pipe, gt)
    t2 = build_extras_table(pipe, gt)
    pd.testing.assert_frame_equal(t1, t2)
    assert set(t1["analysis"]) == {"bland_altman", "skill", "density", "box_confusion", "temporal"}
    assert list(t1.columns) == ["analysis", "row", "col", "value"]
