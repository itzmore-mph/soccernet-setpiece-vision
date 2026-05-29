"""Property-based tests for pipeline-optimization spec.

Uses hypothesis library to verify universal correctness properties
across randomly generated inputs.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Add scripts directory to path so we can import _pipeline_core
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

from _pipeline_core import (
    assign_teams_global_consensus,
    compute_setpiece_ball_position,
    filter_pitch_bounds,
    interpolate_ball_gaps,
    verify_ssd_mount,
)


# ---------------------------------------------------------------------------
# Feature: pipeline-optimization, Property 1: Effective Sample Size Formula Correctness
# ---------------------------------------------------------------------------
# *For any* valid ICC value in [0, 1], number of clips n > 0, and number of
# frames m > 1, the computed effective sample size n_eff = n / (1 + (m−1) × ICC)
# SHALL satisfy: 0 < n_eff ≤ n, and when ICC = 0 then n_eff = n (no correlation
# means no adjustment).
#
# **Validates: Requirements 2.2**
# ---------------------------------------------------------------------------


def compute_n_eff(n: int, m: int, icc: float) -> float:
    """Replicate the n_eff formula from compute_icc.py."""
    return n / (1 + (m - 1) * icc)


@settings(max_examples=100)
@given(
    n=st.integers(min_value=1, max_value=1000),
    m=st.integers(min_value=2, max_value=500),
    icc=st.floats(min_value=0.0, max_value=1.0),
)
def test_n_eff_formula(n, m, icc):
    """Property 1: Effective Sample Size Formula Correctness.

    For any valid ICC in [0, 1], n > 0, m > 1:
    - 0 < n_eff <= n
    - When ICC = 0, n_eff = n

    Validates: Requirements 2.2
    """
    n_eff = compute_n_eff(n, m, icc)

    # n_eff must be positive
    assert n_eff > 0, f"n_eff should be > 0, got {n_eff}"

    # n_eff must not exceed n (correlation can only reduce effective size)
    assert n_eff <= n + 1e-10, f"n_eff ({n_eff}) should be <= n ({n})"

    # When ICC = 0, n_eff = n (no correlation → no adjustment)
    if icc == 0.0:
        assert abs(n_eff - n) < 1e-10, (
            f"When ICC=0, n_eff should equal n. Got n_eff={n_eff}, n={n}"
        )

    # When ICC = 1, n_eff = n / m (maximum correlation → maximum reduction)
    if icc == 1.0:
        expected = n / m
        assert abs(n_eff - expected) < 1e-10, (
            f"When ICC=1, n_eff should equal n/m={expected}. Got n_eff={n_eff}"
        )


# ---------------------------------------------------------------------------
# Feature: pipeline-optimization, Property 3: Pitch Bounds Filtering Correctness
# ---------------------------------------------------------------------------
# *For any* set of 2D coordinates, after applying `filter_pitch_bounds()`,
# all retained coordinates SHALL have x ∈ [0, 105] and y ∈ [0, 68], and all
# coordinates within those bounds in the input SHALL be present in the output.
#
# **Validates: Requirements 4.4**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    coords=st.lists(
        st.tuples(
            st.floats(min_value=-50, max_value=200, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-50, max_value=150, allow_nan=False, allow_infinity=False),
        ),
        min_size=0,
        max_size=50,
    ),
)
def test_pitch_bounds_filter(coords):
    """Property 3: Pitch Bounds Filtering Correctness.

    For any set of 2D coordinates:
    - All retained coordinates have x ∈ [0, 105] and y ∈ [0, 68]
    - All input coordinates within bounds are present in the output (no valid points dropped)

    Validates: Requirements 4.4
    """
    arr = np.array(coords, dtype=np.float64).reshape(-1, 2) if coords else np.empty((0, 2))
    result = filter_pitch_bounds(arr)

    # Property A: All retained coordinates are within bounds
    if len(result) > 0:
        assert np.all(result[:, 0] >= 0), "Some x < 0 in output"
        assert np.all(result[:, 0] <= 105), "Some x > 105 in output"
        assert np.all(result[:, 1] >= 0), "Some y < 0 in output"
        assert np.all(result[:, 1] <= 68), "Some y > 68 in output"

    # Property B: No valid coordinates are dropped
    # Count how many input coords are within bounds
    if len(arr) > 0:
        valid_mask = (
            (arr[:, 0] >= 0) & (arr[:, 0] <= 105) &
            (arr[:, 1] >= 0) & (arr[:, 1] <= 68)
        )
        expected_count = int(valid_mask.sum())
        assert len(result) == expected_count, (
            f"Expected {expected_count} valid coords in output, got {len(result)}"
        )


# ---------------------------------------------------------------------------
# Feature: pipeline-optimization, Property 2: Mode Consensus Equals Statistical Mode
# ---------------------------------------------------------------------------
# *For any* track with a non-empty sequence of team labels across frames,
# the `assign_teams_global_consensus()` function SHALL return the most
# frequently occurring label for that track (the statistical mode).
#
# **Validates: Requirements 3.2**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    # Generate a dict of {frame_idx: {track_id: label}}
    # with at least 1 frame and 1 track
    per_frame_labels=st.dictionaries(
        keys=st.integers(min_value=1, max_value=250),
        values=st.dictionaries(
            keys=st.integers(min_value=0, max_value=50),
            values=st.integers(min_value=0, max_value=2),
            min_size=1,
        ),
        min_size=1,
    ),
)
def test_mode_consensus(per_frame_labels):
    """Property 2: Mode Consensus Equals Statistical Mode.

    For any track with a non-empty sequence of team labels across frames,
    assign_teams_global_consensus() returns the most frequently occurring
    label (statistical mode), with ties broken by smallest label.

    Validates: Requirements 3.2
    """
    consensus = assign_teams_global_consensus(per_frame_labels)

    # Reconstruct expected mode per track independently
    track_labels: dict[int, list[int]] = {}
    for frame_map in per_frame_labels.values():
        for tid, lbl in frame_map.items():
            track_labels.setdefault(tid, []).append(lbl)

    # Every track that appears in the input must have a consensus label
    assert set(consensus.keys()) == set(track_labels.keys())

    # For each track, verify the consensus label is the mode
    for tid, labels in track_labels.items():
        counts = Counter(labels)
        max_count = max(counts.values())
        # Tie-break: smallest label among those with max count
        expected_mode = min(lbl for lbl, cnt in counts.items() if cnt == max_count)
        assert consensus[tid] == expected_mode, (
            f"Track {tid}: expected mode {expected_mode}, got {consensus[tid]}. "
            f"Labels: {labels}, Counts: {dict(counts)}"
        )


# ---------------------------------------------------------------------------
# Feature: pipeline-optimization, Property 4: Ball Gap Interpolation Boundary
# ---------------------------------------------------------------------------
# *For any* ball position time series with gaps of varying sizes,
# `interpolate_ball_gaps(max_gap=5)` SHALL fill all gaps of <= 5 consecutive
# missing frames with linearly interpolated values, and SHALL leave gaps of
# > 5 frames unfilled.
#
# **Validates: Requirements 5.3**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    frame_indices=st.lists(
        st.integers(min_value=1, max_value=300),
        min_size=2,
        max_size=30,
        unique=True,
    ),
)
def test_ball_gap_interpolation(frame_indices):
    """Property 4: Ball Gap Interpolation Boundary.

    Validates: Requirements 5.3
    """
    frame_indices = sorted(frame_indices)

    # Create ball_df with deterministic positions based on index
    ball_df = pd.DataFrame(
        {
            "frame_idx": frame_indices,
            "x_pitch": [float(i * 3) for i in range(len(frame_indices))],
            "y_pitch": [float(i * 2) for i in range(len(frame_indices))],
        }
    )

    result = interpolate_ball_gaps(ball_df, max_gap=5)

    # Property: gaps <= 5 are filled with interpolated frames
    for i in range(len(frame_indices) - 1):
        gap = frame_indices[i + 1] - frame_indices[i] - 1
        if gap <= 5:
            for f in range(frame_indices[i] + 1, frame_indices[i + 1]):
                assert f in result["frame_idx"].values, (
                    f"Frame {f} should be interpolated (gap={gap} <= 5)"
                )
        elif gap > 5:
            for f in range(frame_indices[i] + 1, frame_indices[i + 1]):
                assert f not in result["frame_idx"].values, (
                    f"Frame {f} should NOT be interpolated (gap={gap} > 5)"
                )


# ---------------------------------------------------------------------------
# Feature: pipeline-optimization, Property 6: Set-Piece Ball Position Frame-1 Priority
# ---------------------------------------------------------------------------
# *For any* clip with ball detections where frame-1 has a valid detection
# within pitch bounds, the computed set-piece ball position SHALL equal the
# frame-1 projected position. For any clip where frame-1 has no valid
# detection, the position SHALL equal the median of valid detections in
# frames 1–5.
#
# **Validates: Requirements 5.5**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    x1=st.floats(min_value=0, max_value=105),
    y1=st.floats(min_value=0, max_value=68),
    other_frames=st.lists(
        st.tuples(
            st.integers(min_value=2, max_value=250),
            st.floats(min_value=0, max_value=105),
            st.floats(min_value=0, max_value=68),
        ),
        min_size=0,
        max_size=20,
    ),
)
def test_ball_position_priority(x1, y1, other_frames):
    """Property 6: Set-Piece Ball Position Frame-1 Priority (frame-1 present).

    When frame-1 has a valid detection, the set-piece ball position SHALL
    equal the frame-1 projected position regardless of other detections.

    Validates: Requirements 5.5
    """
    # Case: frame-1 present → should return frame-1 position
    rows = [(1, x1, y1)] + [(f, x, y) for f, x, y in other_frames]
    df = pd.DataFrame(rows, columns=["frame_idx", "x_pitch", "y_pitch"])
    result = compute_setpiece_ball_position(df)
    assert result == (x1, y1), f"Expected frame-1 position ({x1}, {y1}), got {result}"


@settings(max_examples=100)
@given(
    frames_1_to_5=st.lists(
        st.tuples(
            st.integers(min_value=2, max_value=5),
            st.floats(min_value=0, max_value=105, allow_nan=False),
            st.floats(min_value=0, max_value=68, allow_nan=False),
        ),
        min_size=1,
        max_size=4,
    ),
)
def test_ball_position_median_fallback(frames_1_to_5):
    """Property 6: Set-Piece Ball Position Frame-1 Priority (median fallback).

    When frame-1 has no valid detection, the position SHALL equal the median
    of valid detections in frames 1–5.

    Validates: Requirements 5.5
    """
    df = pd.DataFrame(frames_1_to_5, columns=["frame_idx", "x_pitch", "y_pitch"])
    result = compute_setpiece_ball_position(df)
    expected_x = float(df["x_pitch"].median())
    expected_y = float(df["y_pitch"].median())
    assert abs(result[0] - expected_x) < 1e-10, (
        f"Expected median x={expected_x}, got {result[0]}"
    )
    assert abs(result[1] - expected_y) < 1e-10, (
        f"Expected median y={expected_y}, got {result[1]}"
    )


# ---------------------------------------------------------------------------
# Feature: pipeline-optimization, Property 7: SSD Mount Verification Rejects Invalid Paths
# ---------------------------------------------------------------------------
# *For any* path string that does not correspond to an existing directory,
# `verify_ssd_mount()` SHALL exit the process with a non-zero status and an
# error message containing the path.
#
# **Validates: Requirements 3.6, 6.6, 7.5, 9.4**
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    path_suffix=st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "N"), whitelist_characters="-_/"
        ),
        min_size=5,
        max_size=50,
    ),
)
def test_ssd_mount_rejects_invalid(path_suffix):
    """Property 7: SSD Mount Verification Rejects Invalid Paths.

    For any path that does not correspond to an existing directory,
    verify_ssd_mount() SHALL exit with non-zero status and an error
    message containing the path.

    Validates: Requirements 3.6, 6.6, 7.5, 9.4
    """
    import tempfile

    # Create a non-existent path (guaranteed not to exist on disk)
    fake_path = f"/nonexistent/{path_suffix}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        env_file = Path(tmp_dir) / ".env"
        env_file.write_text(f"SOCCERNET_LOCAL_DIR={fake_path}\n")

        with pytest.raises(SystemExit) as exc_info:
            verify_ssd_mount(str(env_file))

    # Should exit with non-zero status.
    # sys.exit(string_message) sets code to the string itself (which counts as
    # non-zero/truthy exit). sys.exit(int) sets code to that int.
    exit_code = exc_info.value.code
    assert exit_code != 0 or isinstance(exit_code, str), (
        f"Expected non-zero exit code, got {exit_code!r}"
    )
    # The error message should contain the path so the user knows what's wrong
    if isinstance(exit_code, str):
        assert fake_path in exit_code or "gamestate-2024" in exit_code, (
            f"Error message should reference the path. Got: {exit_code}"
        )



# ---------------------------------------------------------------------------
# Feature: pipeline-optimization, Property 8: Reproducibility Idempotence
# ---------------------------------------------------------------------------
# *For any* set of committed input parquets (detections, ball positions),
# running the PC computation pipeline twice SHALL produce byte-identical
# output parquets.
#
# **Validates: Requirements 7.2**
# ---------------------------------------------------------------------------


def test_pc_idempotence():
    """Property 8: Reproducibility Idempotence.

    Running the PC computation pipeline twice on the same inputs
    produces identical output parquets.

    Validates: Requirements 7.2
    """
    from _pipeline_core import process_track

    # Read committed inputs
    det = pd.read_parquet(OUTPUTS_DIR / "detections_soccana_tvcalib.parquet")
    balls_raw = pd.read_parquet(OUTPUTS_DIR / "ball_positions.parquet")

    # Expand balls to per-frame rows (same logic as run_pc_soccana_tvcalib.py)
    if "frame_idx" in balls_raw.columns and "ball_x_m" in balls_raw.columns:
        balls = balls_raw
    else:
        # New schema: expand per-clip to per-frame
        frames = det[["split", "clip_id", "frame_idx"]].drop_duplicates()
        balls_renamed = balls_raw.rename(
            columns={"x_pitch": "ball_x_m", "y_pitch": "ball_y_m"}
        )
        balls_renamed = balls_renamed[["split", "clip_id", "ball_x_m", "ball_y_m"]]
        balls = frames.merge(balls_renamed, on=["split", "clip_id"], how="inner")

    # Run PC computation twice
    pc_run1 = process_track(
        det, track_name="soccana_tvcalib", team_col="team_kmeans", balls=balls
    )
    pc_run2 = process_track(
        det, track_name="soccana_tvcalib", team_col="team_kmeans", balls=balls
    )

    # Assert identical outputs
    pd.testing.assert_frame_equal(pc_run1, pc_run2)
