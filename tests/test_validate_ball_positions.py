"""Unit tests for validate_ball_positions in _pipeline_core.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add scripts directory to path so we can import _pipeline_core
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from _pipeline_core import validate_ball_positions


# ---------------------------------------------------------------------------
# Helper to build DataFrames
# ---------------------------------------------------------------------------

def _make_df(rows: list[tuple], columns: list[str] = None) -> pd.DataFrame:
    """Create a DataFrame from tuples with standard ball position columns."""
    if columns is None:
        columns = ["clip_id", "x_pitch", "y_pitch"]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Tests: Basic merge and error computation
# ---------------------------------------------------------------------------

class TestBasicValidation:
    """Core merge and Euclidean distance computation."""

    def test_identical_positions_give_zero_error(self):
        auto = _make_df([("clip_a", 50.0, 34.0), ("clip_b", 20.0, 10.0)])
        gt = _make_df([("clip_a", 50.0, 34.0), ("clip_b", 20.0, 10.0)])
        result = validate_ball_positions(auto, gt)
        assert len(result) == 2
        assert all(result["euclidean_error"] == 0.0)
        assert all(~result["flagged"])

    def test_known_euclidean_distance(self):
        # Distance between (0, 0) and (3, 4) = 5.0
        auto = _make_df([("clip_a", 0.0, 0.0)])
        gt = _make_df([("clip_a", 3.0, 4.0)])
        result = validate_ball_positions(auto, gt)
        assert len(result) == 1
        assert np.isclose(result["euclidean_error"].iloc[0], 5.0)

    def test_euclidean_distance_symmetric(self):
        auto = _make_df([("clip_a", 10.0, 20.0)])
        gt = _make_df([("clip_a", 13.0, 24.0)])
        result = validate_ball_positions(auto, gt)
        expected = np.sqrt(9.0 + 16.0)  # 5.0
        assert np.isclose(result["euclidean_error"].iloc[0], expected)


# ---------------------------------------------------------------------------
# Tests: Flagging logic
# ---------------------------------------------------------------------------

class TestFlagging:
    """Clips with error > 5m should be flagged."""

    def test_error_exactly_5m_not_flagged(self):
        # Distance = 5.0 exactly (not > 5)
        auto = _make_df([("clip_a", 0.0, 0.0)])
        gt = _make_df([("clip_a", 3.0, 4.0)])
        result = validate_ball_positions(auto, gt)
        assert not result["flagged"].iloc[0]

    def test_error_above_5m_flagged(self):
        # Distance between (0, 0) and (5, 5) = sqrt(50) ≈ 7.07
        auto = _make_df([("clip_a", 0.0, 0.0)])
        gt = _make_df([("clip_a", 5.0, 5.0)])
        result = validate_ball_positions(auto, gt)
        assert result["flagged"].iloc[0]

    def test_mixed_flagged_and_unflagged(self):
        auto = _make_df([
            ("clip_a", 50.0, 34.0),  # close to GT
            ("clip_b", 0.0, 0.0),    # far from GT
        ])
        gt = _make_df([
            ("clip_a", 51.0, 34.0),  # error = 1.0
            ("clip_b", 10.0, 10.0),  # error = sqrt(200) ≈ 14.14
        ])
        result = validate_ball_positions(auto, gt)
        assert not result.loc[result["clip_id"] == "clip_a", "flagged"].iloc[0]
        assert result.loc[result["clip_id"] == "clip_b", "flagged"].iloc[0]


# ---------------------------------------------------------------------------
# Tests: Merge behavior
# ---------------------------------------------------------------------------

class TestMergeBehavior:
    """Only clips present in both DataFrames should appear in output."""

    def test_only_matching_clips_included(self):
        auto = _make_df([("clip_a", 50.0, 34.0), ("clip_c", 10.0, 10.0)])
        gt = _make_df([("clip_a", 50.0, 34.0), ("clip_b", 20.0, 20.0)])
        result = validate_ball_positions(auto, gt)
        assert len(result) == 1
        assert result["clip_id"].iloc[0] == "clip_a"

    def test_no_matching_clips_returns_empty(self):
        auto = _make_df([("clip_a", 50.0, 34.0)])
        gt = _make_df([("clip_b", 20.0, 20.0)])
        result = validate_ball_positions(auto, gt)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: Output schema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Validate the output DataFrame has the expected columns."""

    def test_output_columns(self):
        auto = _make_df([("clip_a", 50.0, 34.0)])
        gt = _make_df([("clip_a", 51.0, 35.0)])
        result = validate_ball_positions(auto, gt)
        expected_cols = {"clip_id", "x_auto", "y_auto", "x_gt", "y_gt", "euclidean_error", "flagged"}
        assert set(result.columns) == expected_cols

    def test_output_dtypes(self):
        auto = _make_df([("clip_a", 50.0, 34.0)])
        gt = _make_df([("clip_a", 51.0, 35.0)])
        result = validate_ball_positions(auto, gt)
        assert result["euclidean_error"].dtype == np.float64
        assert result["flagged"].dtype == bool
