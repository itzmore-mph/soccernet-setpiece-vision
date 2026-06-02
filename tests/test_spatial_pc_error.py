"""Unit tests for the spatial PC error helpers (orientation + ball expansion)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from _pipeline_core import GRID_NX, GRID_NY, PITCH_LENGTH_M  # noqa: E402
from spatial_pc_error import _expand_balls, frame_surfaces  # noqa: E402


def test_expand_balls_broadcasts_clip_ball_to_all_frames():
    det = pd.DataFrame(
        {
            "split": ["train"] * 3,
            "clip_id": ["a", "a", "a"],
            "frame_idx": [0, 1, 2],
        }
    )
    balls = pd.DataFrame({"split": ["train"], "clip_id": ["a"], "x_pitch": [50.0], "y_pitch": [34.0]})
    out = _expand_balls(balls, det)
    assert len(out) == 3
    assert set(out["ball_x_m"]) == {50.0}


def _frame(team_col: str, xs, ys, teams, ball, clip="a") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["train"] * len(xs),
            "clip_id": [clip] * len(xs),
            "frame_idx": [0] * len(xs),
            "x_m": xs,
            "y_m": ys,
            team_col: teams,
        }
    ), pd.DataFrame({"split": ["train"], "clip_id": [clip], "x_pitch": [ball[0]], "y_pitch": [ball[1]]})


def test_frame_surface_shape():
    det, balls = _frame("team", [30.0, 35.0, 70.0, 75.0], [30.0, 38.0, 30.0, 38.0], [0, 0, 1, 1], (32.0, 34.0))
    surf = frame_surfaces(det, "team", balls)
    (key,) = surf.keys()
    assert surf[key].shape == (GRID_NY, GRID_NX)


def test_orientation_makes_mirror_configs_identical():
    """A left-attacking frame and its right-attacking mirror orient to the same surface."""
    # Left config: ball on the left, attackers near the left goal.
    left_det, left_balls = _frame(
        "team", [20.0, 25.0, 60.0, 65.0], [30.0, 38.0, 30.0, 38.0], [0, 0, 1, 1], (18.0, 34.0), clip="L"
    )
    # Mirror across the halfway line: x -> 105 - x, same teams and ball mirrored.
    mx = [PITCH_LENGTH_M - x for x in [20.0, 25.0, 60.0, 65.0]]
    right_det, right_balls = _frame(
        "team", mx, [30.0, 38.0, 30.0, 38.0], [0, 0, 1, 1], (PITCH_LENGTH_M - 18.0, 34.0), clip="R"
    )
    sl = list(frame_surfaces(left_det, "team", left_balls).values())[0]
    sr = list(frame_surfaces(right_det, "team", right_balls).values())[0]
    # Both oriented attack-to-right, so they must match closely.
    assert np.allclose(sl, sr, atol=1e-9)


def test_empty_team_frame_is_skipped():
    """A frame with only one team yields no surface (matches process_track)."""
    det, balls = _frame("team", [30.0, 35.0], [30.0, 38.0], [0, 0], (32.0, 34.0))
    assert frame_surfaces(det, "team", balls) == {}
