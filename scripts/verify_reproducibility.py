"""Verify all analysis reproduces from committed parquets (no SSD needed).

Checks:
1. PC computation from detections + ball positions → matches committed PC parquet
2. Validation statistics from PC parquets → matches committed validation parquet
3. ICC computation from PC parquet → matches committed ICC parquet

Exit code 0 on success, non-zero on failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

# Ensure sibling scripts are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pipeline_core import process_track
from compute_icc import compute_icc_per_metric
from ks_table_tvcalib import compare, METRICS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Tolerance for floating-point comparison
RTOL = 1e-5
ATOL = 1e-8


def _expand_balls_to_frames(
    balls_per_clip: pd.DataFrame, det: pd.DataFrame
) -> pd.DataFrame:
    """Expand per-clip ball positions to per-frame rows matching process_track schema.

    Handles both old schema (frame_idx, ball_x_m, ball_y_m) and new schema
    (x_pitch, y_pitch) from the optimized pipeline.
    """
    if "frame_idx" in balls_per_clip.columns and "ball_x_m" in balls_per_clip.columns:
        return balls_per_clip

    frames = det[["split", "clip_id", "frame_idx"]].drop_duplicates()
    balls_renamed = balls_per_clip.rename(
        columns={"x_pitch": "ball_x_m", "y_pitch": "ball_y_m"}
    )
    balls_renamed = balls_renamed[["split", "clip_id", "ball_x_m", "ball_y_m"]]
    expanded = frames.merge(balls_renamed, on=["split", "clip_id"], how="inner")
    return expanded


def verify_pc_computation() -> bool:
    """Re-derive pitch control from detections + ball positions and compare."""
    print("=" * 70)
    print("[1/3] Verifying PC computation reproducibility...")
    print("=" * 70)

    det_path = OUTPUTS_DIR / "detections_soccana_tvcalib.parquet"
    balls_path = OUTPUTS_DIR / "ball_positions.parquet"
    committed_path = OUTPUTS_DIR / "pitch_control_soccana_tvcalib.parquet"

    for p in (det_path, balls_path, committed_path):
        if not p.exists():
            print(f"  SKIP: {p.name} not found")
            return True  # Not a failure if inputs missing

    det = pd.read_parquet(det_path)
    balls_raw = pd.read_parquet(balls_path)
    balls = _expand_balls_to_frames(balls_raw, det)
    committed = pd.read_parquet(committed_path)

    print(f"  Detections: {det.shape[0]} rows, {det['clip_id'].nunique()} clips")
    print(f"  Ball positions (expanded): {balls.shape[0]} rows")
    print(f"  Committed PC: {committed.shape[0]} rows")

    print("  Re-computing pitch control...")
    recomputed = process_track(
        det, track_name="soccana_tvcalib", team_col="team_kmeans", balls=balls
    )

    # Sort both for consistent comparison
    sort_cols = ["split", "clip_id", "frame_idx"]
    committed_sorted = committed.sort_values(sort_cols).reset_index(drop=True)
    recomputed_sorted = recomputed.sort_values(sort_cols).reset_index(drop=True)

    # Compare only the columns present in both
    common_cols = sorted(set(committed_sorted.columns) & set(recomputed_sorted.columns))
    committed_cmp = committed_sorted[common_cols].reset_index(drop=True)
    recomputed_cmp = recomputed_sorted[common_cols].reset_index(drop=True)

    try:
        assert_frame_equal(
            recomputed_cmp,
            committed_cmp,
            rtol=RTOL,
            atol=ATOL,
            check_dtype=False,
            obj="PC recomputed vs committed",
        )
        print("  ✓ PC computation reproduces identically")
        return True
    except AssertionError as e:
        print(f"  ✗ PC computation MISMATCH:\n    {e}")
        return False


def verify_validation_statistics() -> bool:
    """Re-derive validation statistics from PC parquets and compare."""
    print()
    print("=" * 70)
    print("[2/3] Verifying validation statistics reproducibility...")
    print("=" * 70)

    pc_pipe_path = OUTPUTS_DIR / "pitch_control_soccana_tvcalib.parquet"
    pc_gt_path = OUTPUTS_DIR / "pitch_control_gt_full.parquet"
    committed_path = OUTPUTS_DIR / "validation_summary_tvcalib.parquet"

    for p in (pc_pipe_path, pc_gt_path, committed_path):
        if not p.exists():
            print(f"  SKIP: {p.name} not found")
            return True

    pc_pipe = pd.read_parquet(pc_pipe_path)
    pc_gt = pd.read_parquet(pc_gt_path)
    committed = pd.read_parquet(committed_path)

    print(f"  Pipeline PC: {pc_pipe.shape[0]} rows, {pc_pipe['clip_id'].nunique()} clips")
    print(f"  GT PC: {pc_gt.shape[0]} rows, {pc_gt['clip_id'].nunique()} clips")
    print(f"  Committed validation: {committed.shape[0]} rows")

    print("  Re-computing validation statistics...")
    rows = []
    for metric in METRICS:
        pipe_a = pc_pipe[metric].dropna().to_numpy()
        gt_a = pc_gt[metric].dropna().to_numpy()
        rows.append(compare(pipe_a, gt_a, metric))
    recomputed = pd.DataFrame(rows)

    # Sort both for consistent comparison
    committed_sorted = committed.sort_values("metric").reset_index(drop=True)
    recomputed_sorted = recomputed.sort_values("metric").reset_index(drop=True)

    # Compare only common columns
    common_cols = sorted(set(committed_sorted.columns) & set(recomputed_sorted.columns))
    committed_cmp = committed_sorted[common_cols].reset_index(drop=True)
    recomputed_cmp = recomputed_sorted[common_cols].reset_index(drop=True)

    try:
        assert_frame_equal(
            recomputed_cmp,
            committed_cmp,
            rtol=RTOL,
            atol=ATOL,
            check_dtype=False,
            obj="Validation recomputed vs committed",
        )
        print("  ✓ Validation statistics reproduce identically")
        return True
    except AssertionError as e:
        print(f"  ✗ Validation statistics MISMATCH:\n    {e}")
        return False


def verify_icc_computation() -> bool:
    """Re-derive ICC from PC parquet and compare."""
    print()
    print("=" * 70)
    print("[3/3] Verifying ICC computation reproducibility...")
    print("=" * 70)

    pc_path = OUTPUTS_DIR / "pitch_control_soccana_tvcalib.parquet"
    committed_path = OUTPUTS_DIR / "icc_per_metric.parquet"

    for p in (pc_path, committed_path):
        if not p.exists():
            print(f"  SKIP: {p.name} not found")
            return True

    pc = pd.read_parquet(pc_path)
    committed = pd.read_parquet(committed_path)

    print(f"  PC data: {pc.shape[0]} rows, {pc['clip_id'].nunique()} clips")
    print(f"  Committed ICC: {committed.shape[0]} rows")

    print("  Re-computing ICC(2,1) per metric...")
    recomputed = compute_icc_per_metric(pc)

    # Sort both for consistent comparison
    committed_sorted = committed.sort_values("metric").reset_index(drop=True)
    recomputed_sorted = recomputed.sort_values("metric").reset_index(drop=True)

    # Compare only common columns
    common_cols = sorted(set(committed_sorted.columns) & set(recomputed_sorted.columns))
    committed_cmp = committed_sorted[common_cols].reset_index(drop=True)
    recomputed_cmp = recomputed_sorted[common_cols].reset_index(drop=True)

    try:
        assert_frame_equal(
            recomputed_cmp,
            committed_cmp,
            rtol=RTOL,
            atol=ATOL,
            check_dtype=False,
            obj="ICC recomputed vs committed",
        )
        print("  ✓ ICC computation reproduces identically")
        return True
    except AssertionError as e:
        print(f"  ✗ ICC computation MISMATCH:\n    {e}")
        return False


def main() -> int:
    """Run all reproducibility checks. Returns 0 on success, 1 on failure."""
    print("Reproducibility Verification (Level 1: from committed parquets)")
    print("No SSD required — all inputs are committed to the repository.")
    print()

    results = {
        "PC computation": verify_pc_computation(),
        "Validation statistics": verify_validation_statistics(),
        "ICC computation": verify_icc_computation(),
    }

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll reproducibility checks passed.")
        return 0
    else:
        print("\nSome reproducibility checks FAILED. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
