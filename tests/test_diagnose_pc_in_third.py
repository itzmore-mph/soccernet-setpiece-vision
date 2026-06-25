"""Unit tests for the pc_in_third diagnostic helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from diagnose_pc_in_third import (  # noqa: E402
    bootstrap_r_ci,
    compute_stratified_stats,
    safe_pearson,
    safe_spearman,
)


def test_safe_pearson_perfect_positive():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    r, p = safe_pearson(a, 2 * a + 1)
    assert r == pytest.approx(1.0)


def test_safe_pearson_constant_input_returns_nan():
    """A zero-variance input must not raise; it returns NaN, not a divide error."""
    r, p = safe_pearson(np.array([0.5, 0.5, 0.5, 0.5]), np.array([0.1, 0.2, 0.3, 0.4]))
    assert np.isnan(r) and np.isnan(p)


def test_safe_pearson_too_few_points_returns_nan():
    r, _ = safe_pearson(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert np.isnan(r)


def test_safe_spearman_handles_nan_pairs():
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    b = np.array([2.0, 1.0, 3.0, 4.0, 5.0])
    rho, _ = safe_spearman(a, b)
    assert np.isfinite(rho)


def test_bootstrap_r_ci_deterministic_and_ordered():
    rng = np.random.default_rng(0)
    a = rng.normal(size=40)
    b = a + rng.normal(scale=0.5, size=40)
    lo1, hi1 = bootstrap_r_ci(a, b, seed=123)
    lo2, hi2 = bootstrap_r_ci(a, b, seed=123)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= hi1


def test_bootstrap_r_ci_degenerate_returns_nan():
    lo, hi = bootstrap_r_ci(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert np.isnan(lo) and np.isnan(hi)


def _toy_pc(action_signs: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build pipeline/GT frames where each action class has a prescribed r sign.

    Pipeline = 0.5 + sign * (gt - 0.5) + small noise, so the frame-level Pearson
    correlation within each action class carries the prescribed sign.
    """
    rng = np.random.default_rng(7)
    pipe_rows, gt_rows = [], []
    clip = 0
    for action, sign in action_signs.items():
        for _ in range(5):
            clip += 1
            for frame in range(6):
                gt_val = rng.uniform(0.3, 0.7)
                pipe_val = 0.5 + sign * (gt_val - 0.5) + rng.normal(scale=0.01)
                key = {"split": "train", "clip_id": f"c{clip}", "frame_idx": frame, "action_class": action}
                gt_rows.append({**key, "pc_in_third": gt_val})
                pipe_rows.append({**key, "pc_in_third": pipe_val})
    return pd.DataFrame(pipe_rows), pd.DataFrame(gt_rows)


def test_compute_stratified_stats_recovers_opposite_signs():
    """Simpson check: opposite per-class signs are surfaced by stratification."""
    pipe, gt = _toy_pc({"Corner": +1, "Direct free-kick": -1})
    table = compute_stratified_stats(pipe, gt)
    corner = table.loc[table["segment"] == "frame | Corner", "pearson_r"].iloc[0]
    dfk = table.loc[table["segment"] == "frame | Direct free-kick", "pearson_r"].iloc[0]
    assert corner > 0
    assert dfk < 0
    # Expected segments present and sorted.
    assert set(table["segment"]) >= {"ALL frames", "ALL clips", "frame | Corner", "frame | Direct free-kick"}
