"""Unit tests for clip-level validation transforms."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from clip_level_validation import bootstrap_bias_ci, clip_means, compare_clip_level  # noqa: E402


def test_clip_means_collapses_frames_to_clip_mean():
    """Each clip's frames collapse to their per-metric mean."""
    df = pd.DataFrame(
        {
            "split": ["train"] * 4,
            "clip_id": ["a", "a", "b", "b"],
            "frame_idx": [1, 2, 1, 2],
            "pc_mean": [0.2, 0.4, 0.8, 1.0],
            "pc_at_ball": [0.5, 0.5, 0.0, 1.0],
            "pc_in_box": [0.1, 0.1, 0.1, 0.1],
            "pc_in_third": [0.3, 0.3, 0.6, 0.6],
            "pc_area_gt_0p5": [0.0, 0.0, 0.5, 0.5],
        }
    )
    out = clip_means(df)
    assert out.loc[("train", "a"), "pc_mean"] == pytest.approx(0.3)
    assert out.loc[("train", "b"), "pc_mean"] == pytest.approx(0.9)
    assert out.loc[("train", "a"), "pc_at_ball"] == pytest.approx(0.5)


def test_bootstrap_bias_ci_is_deterministic():
    """A fixed seed yields identical bounds across calls (CI reproducibility)."""
    diffs = np.array([0.1, -0.2, 0.05, 0.3, -0.1, 0.0, 0.15])
    assert bootstrap_bias_ci(diffs, seed=42) == bootstrap_bias_ci(diffs, seed=42)


def test_bootstrap_bias_ci_brackets_mean_and_is_ordered():
    """Lower bound <= mean <= upper bound for a moderate sample."""
    diffs = np.array([0.1, -0.2, 0.05, 0.3, -0.1, 0.0, 0.15, 0.2, -0.05, 0.1])
    lo, hi = bootstrap_bias_ci(diffs)
    assert lo <= diffs.mean() <= hi


def test_compare_clip_level_flags_clear_positive_bias():
    """A consistent positive offset is detected: CI excludes zero, bias positive."""
    gt = pd.Series(np.linspace(0.1, 0.5, 12))
    pipe = gt + 0.3
    row = compare_clip_level(pipe, gt, "pc_in_box")
    assert row["bias"] == pytest.approx(0.3, abs=1e-9)
    assert row["ci_excludes_zero"] is True
    assert row["n_clips"] == 12
    # A pure constant shift preserves rank order, so Pearson is ~1.
    assert row["pearson"] == pytest.approx(1.0, abs=1e-9)


def test_compare_clip_level_no_bias_includes_zero():
    """Identical inputs give zero bias and a CI that contains zero."""
    gt = pd.Series(np.linspace(0.2, 0.8, 15))
    row = compare_clip_level(gt.copy(), gt.copy(), "pc_mean")
    assert row["bias"] == pytest.approx(0.0, abs=1e-12)
    assert row["ci_excludes_zero"] is False
    assert row["wilcoxon_p"] == pytest.approx(1.0)
