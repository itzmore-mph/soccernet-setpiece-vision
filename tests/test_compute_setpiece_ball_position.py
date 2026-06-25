"""Unit tests for compute_setpiece_ball_position in _pipeline_core.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add scripts directory to path so we can import _pipeline_core
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from _pipeline_core import compute_setpiece_ball_position

# ---------------------------------------------------------------------------
# Helper to build a ball detections DataFrame
# ---------------------------------------------------------------------------


def _make_ball_df(rows: list[tuple[int, float, float]]) -> pd.DataFrame:
    """Create a ball_detections DataFrame from (frame_idx, x_pitch, y_pitch) tuples."""
    return pd.DataFrame(rows, columns=["frame_idx", "x_pitch", "y_pitch"])


# ---------------------------------------------------------------------------
# Tests: Frame-1 priority
# ---------------------------------------------------------------------------


class TestFrame1Priority:
    """When frame_idx == 1 has a valid detection, it should be used."""

    def test_frame1_present_returns_frame1_position(self):
        df = _make_ball_df(
            [
                (1, 50.0, 34.0),
                (2, 51.0, 35.0),
                (3, 52.0, 36.0),
            ]
        )
        result = compute_setpiece_ball_position(df)
        assert result == (50.0, 34.0)

    def test_frame1_only_detection(self):
        df = _make_ball_df([(1, 10.0, 20.0)])
        result = compute_setpiece_ball_position(df)
        assert result == (10.0, 20.0)

    def test_frame1_takes_priority_over_many_other_frames(self):
        rows = [(1, 0.0, 0.0)] + [(i, 100.0, 68.0) for i in range(2, 50)]
        df = _make_ball_df(rows)
        result = compute_setpiece_ball_position(df)
        assert result == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Tests: Median fallback for frames 1–5
# ---------------------------------------------------------------------------


class TestMedianFrames1To5:
    """When frame-1 is missing but frames 1–5 have detections, use their median."""

    def test_no_frame1_uses_median_of_frames_2_to_5(self):
        df = _make_ball_df(
            [
                (2, 40.0, 30.0),
                (3, 50.0, 34.0),
                (4, 60.0, 38.0),
            ]
        )
        result = compute_setpiece_ball_position(df)
        assert result == (50.0, 34.0)

    def test_single_frame_in_1_to_5_range(self):
        df = _make_ball_df(
            [
                (3, 25.0, 12.0),
                (10, 80.0, 60.0),
            ]
        )
        result = compute_setpiece_ball_position(df)
        assert result == (25.0, 12.0)

    def test_even_number_of_frames_1_to_5(self):
        df = _make_ball_df(
            [
                (2, 40.0, 20.0),
                (4, 60.0, 40.0),
            ]
        )
        result = compute_setpiece_ball_position(df)
        # Median of [40, 60] = 50, median of [20, 40] = 30
        assert result == (50.0, 30.0)


# ---------------------------------------------------------------------------
# Tests: Median fallback for all detections
# ---------------------------------------------------------------------------


class TestMedianAllDetections:
    """When no frames 1–5 exist, use median of all valid detections."""

    def test_only_frames_after_5(self):
        df = _make_ball_df(
            [
                (6, 20.0, 10.0),
                (7, 30.0, 20.0),
                (8, 40.0, 30.0),
            ]
        )
        result = compute_setpiece_ball_position(df)
        assert result == (30.0, 20.0)

    def test_single_frame_after_5(self):
        df = _make_ball_df([(10, 75.0, 50.0)])
        result = compute_setpiece_ball_position(df)
        assert result == (75.0, 50.0)

    def test_many_frames_after_5(self):
        df = _make_ball_df(
            [
                (10, 10.0, 10.0),
                (20, 20.0, 20.0),
                (30, 30.0, 30.0),
                (40, 40.0, 40.0),
                (50, 50.0, 50.0),
            ]
        )
        result = compute_setpiece_ball_position(df)
        # Median of [10, 20, 30, 40, 50] = 30
        assert result == (30.0, 30.0)


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty DataFrame, boundary values."""

    def test_empty_dataframe_raises_value_error(self):
        df = pd.DataFrame(columns=["frame_idx", "x_pitch", "y_pitch"])
        with pytest.raises(ValueError, match="No valid ball detections"):
            compute_setpiece_ball_position(df)

    def test_frame1_at_pitch_boundary(self):
        df = _make_ball_df([(1, 105.0, 68.0)])
        result = compute_setpiece_ball_position(df)
        assert result == (105.0, 68.0)

    def test_frame1_at_origin(self):
        df = _make_ball_df([(1, 0.0, 0.0)])
        result = compute_setpiece_ball_position(df)
        assert result == (0.0, 0.0)

    def test_frame5_is_included_in_1_to_5_range(self):
        """Frame 5 should be included in the 1-5 range (inclusive)."""
        df = _make_ball_df(
            [
                (5, 80.0, 50.0),
                (10, 20.0, 10.0),
            ]
        )
        result = compute_setpiece_ball_position(df)
        assert result == (80.0, 50.0)
