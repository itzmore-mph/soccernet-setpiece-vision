"""Verify all analysis reproduces from committed parquets (no SSD needed).

Checks:
1. PC computation from detections + ball positions → matches committed PC parquet
2. Validation statistics from PC parquets → matches committed validation parquet
3. ICC computation from PC parquet → matches committed ICC parquet
4. Clip-level validation from PC parquets → matches committed clip-level parquet
5. pc_in_third stratified stats from PC parquets → matches committed by-action parquet
6. Supplementary validation extras from PC parquets → matches committed extras parquet
7. Spatial PC error map from detections + ball → matches committed per-cell parquet

Each check is tri-state:
  PASS  — re-derived result matches the committed parquet.
  FAIL  — re-derived result diverges, OR a *public* (committed) input is missing.
  SKIP  — a *private*, NDA-restricted input is absent. This is expected in public
          CI: the raw Soccana detections, ball positions, and homographies are
          fit to NDA video frames and ship only in the closed university
          submission. A SKIP never counts as a PASS and never masks a regression.

Checks 1 and 7 depend on the private detections + ball parquets, so they SKIP in
public CI and only run locally (or in the university submission) where those exist.
Checks 2-6 run from committed public PC parquets, so they are real in CI.

Exit code 0 when nothing FAILED (PASS/SKIP only), non-zero on any FAIL.
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
from clip_level_validation import clip_means, compare_clip_level
from diagnose_pc_in_third import compute_stratified_stats
from validation_extras import build_extras_table
from spatial_pc_error import compute_cell_error

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Tolerance for floating-point comparison
RTOL = 1e-5
ATOL = 1e-8

# Tri-state check results.
PASS, SKIP, FAIL = "PASS", "SKIP", "FAIL"

# NDA video-derived parquets: gitignored, absent in public CI. Missing → SKIP.
PRIVATE_INPUTS = {
    "detections_soccana_tvcalib.parquet",
    "ball_positions.parquet",
    "homographies_tvcalib.parquet",
}


def _classify_missing(path: Path) -> str:
    """Return SKIP if a missing input is a known private parquet, else FAIL."""
    return SKIP if path.name in PRIVATE_INPUTS else FAIL


def _expand_balls_to_frames(balls_per_clip: pd.DataFrame, det: pd.DataFrame) -> pd.DataFrame:
    """Expand per-clip ball positions to per-frame rows matching process_track schema.

    Handles both old schema (frame_idx, ball_x_m, ball_y_m) and new schema
    (x_pitch, y_pitch) from the optimized pipeline.
    """
    if "frame_idx" in balls_per_clip.columns and "ball_x_m" in balls_per_clip.columns:
        return balls_per_clip

    frames = det[["split", "clip_id", "frame_idx"]].drop_duplicates()
    balls_renamed = balls_per_clip.rename(columns={"x_pitch": "ball_x_m", "y_pitch": "ball_y_m"})
    balls_renamed = balls_renamed[["split", "clip_id", "ball_x_m", "ball_y_m"]]
    expanded = frames.merge(balls_renamed, on=["split", "clip_id"], how="inner")
    return expanded


def verify_pc_computation() -> str:
    """Re-derive pitch control from detections + ball positions and compare."""
    print("=" * 70)
    print("[1/7] Verifying PC computation reproducibility...")
    print("=" * 70)

    det_path = OUTPUTS_DIR / "detections_soccana_tvcalib.parquet"
    balls_path = OUTPUTS_DIR / "ball_positions.parquet"
    committed_path = OUTPUTS_DIR / "pitch_control_soccana_tvcalib.parquet"

    for p in (det_path, balls_path, committed_path):
        if not p.exists():
            verdict = _classify_missing(p)
            print(f"  {verdict}: {p.name} not found")
            return verdict

    det = pd.read_parquet(det_path)
    balls_raw = pd.read_parquet(balls_path)
    balls = _expand_balls_to_frames(balls_raw, det)
    committed = pd.read_parquet(committed_path)

    print(f"  Detections: {det.shape[0]} rows, {det['clip_id'].nunique()} clips")
    print(f"  Ball positions (expanded): {balls.shape[0]} rows")
    print(f"  Committed PC: {committed.shape[0]} rows")

    print("  Re-computing pitch control...")
    recomputed = process_track(det, track_name="soccana_tvcalib", team_col="team_kmeans", balls=balls)

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
        return PASS
    except AssertionError as e:
        print(f"  ✗ PC computation MISMATCH:\n    {e}")
        return FAIL


def verify_validation_statistics() -> str:
    """Re-derive validation statistics from PC parquets and compare."""
    print()
    print("=" * 70)
    print("[2/7] Verifying validation statistics reproducibility...")
    print("=" * 70)

    pc_pipe_path = OUTPUTS_DIR / "pitch_control_soccana_tvcalib.parquet"
    pc_gt_path = OUTPUTS_DIR / "pitch_control_gt_full.parquet"
    committed_path = OUTPUTS_DIR / "validation_summary_tvcalib.parquet"

    for p in (pc_pipe_path, pc_gt_path, committed_path):
        if not p.exists():
            verdict = _classify_missing(p)
            print(f"  {verdict}: {p.name} not found")
            return verdict

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
        return PASS
    except AssertionError as e:
        print(f"  ✗ Validation statistics MISMATCH:\n    {e}")
        return FAIL


def verify_icc_computation() -> str:
    """Re-derive ICC from PC parquet and compare."""
    print()
    print("=" * 70)
    print("[3/7] Verifying ICC computation reproducibility...")
    print("=" * 70)

    pc_path = OUTPUTS_DIR / "pitch_control_soccana_tvcalib.parquet"
    committed_path = OUTPUTS_DIR / "icc_per_metric.parquet"

    for p in (pc_path, committed_path):
        if not p.exists():
            verdict = _classify_missing(p)
            print(f"  {verdict}: {p.name} not found")
            return verdict

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
        return PASS
    except AssertionError as e:
        print(f"  ✗ ICC computation MISMATCH:\n    {e}")
        return FAIL


def verify_clip_level_validation() -> str:
    """Re-derive clip-level paired validation from PC parquets and compare."""
    print()
    print("=" * 70)
    print("[4/7] Verifying clip-level validation reproducibility...")
    print("=" * 70)

    pc_pipe_path = OUTPUTS_DIR / "pitch_control_soccana_tvcalib.parquet"
    pc_gt_path = OUTPUTS_DIR / "pitch_control_gt_full.parquet"
    committed_path = OUTPUTS_DIR / "clip_level_validation.parquet"

    for p in (pc_pipe_path, pc_gt_path, committed_path):
        if not p.exists():
            verdict = _classify_missing(p)
            print(f"  {verdict}: {p.name} not found")
            return verdict

    pc_pipe = pd.read_parquet(pc_pipe_path)
    pc_gt = pd.read_parquet(pc_gt_path)
    committed = pd.read_parquet(committed_path)

    pipe_c = clip_means(pc_pipe)
    gt_c = clip_means(pc_gt)
    common = pipe_c.index.intersection(gt_c.index)
    pipe_c = pipe_c.loc[common].sort_index()
    gt_c = gt_c.loc[common].sort_index()
    print(f"  Matched clips: {len(common)}")
    print(f"  Committed clip-level: {committed.shape[0]} rows")

    print("  Re-computing clip-level paired validation (deterministic bootstrap)...")
    recomputed = pd.DataFrame([compare_clip_level(pipe_c[m], gt_c[m], m) for m in METRICS])

    committed_sorted = committed.sort_values("metric").reset_index(drop=True)
    recomputed_sorted = recomputed.sort_values("metric").reset_index(drop=True)

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
            obj="Clip-level recomputed vs committed",
        )
        print("  ✓ Clip-level validation reproduces identically")
        return PASS
    except AssertionError as e:
        print(f"  ✗ Clip-level validation MISMATCH:\n    {e}")
        return FAIL


def verify_pc_in_third_stratified() -> str:
    """Re-derive the pc_in_third by-action stratified stats and compare."""
    print()
    print("=" * 70)
    print("[5/7] Verifying pc_in_third stratified stats reproducibility...")
    print("=" * 70)

    pc_pipe_path = OUTPUTS_DIR / "pitch_control_soccana_tvcalib.parquet"
    pc_gt_path = OUTPUTS_DIR / "pitch_control_gt_full.parquet"
    committed_path = OUTPUTS_DIR / "pc_in_third_by_action.parquet"

    for p in (pc_pipe_path, pc_gt_path, committed_path):
        if not p.exists():
            verdict = _classify_missing(p)
            print(f"  {verdict}: {p.name} not found")
            return verdict

    pc_pipe = pd.read_parquet(pc_pipe_path)
    pc_gt = pd.read_parquet(pc_gt_path)
    committed = pd.read_parquet(committed_path)
    print(f"  Committed stratified stats: {committed.shape[0]} rows")

    print("  Re-computing stratified stats (deterministic bootstrap)...")
    recomputed = compute_stratified_stats(pc_pipe, pc_gt)

    committed_sorted = committed.sort_values("segment").reset_index(drop=True)
    recomputed_sorted = recomputed.sort_values("segment").reset_index(drop=True)

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
            obj="pc_in_third stratified recomputed vs committed",
        )
        print("  ✓ pc_in_third stratified stats reproduce identically")
        return PASS
    except AssertionError as e:
        print(f"  ✗ pc_in_third stratified stats MISMATCH:\n    {e}")
        return FAIL


def verify_validation_extras() -> str:
    """Re-derive the supplementary validation extras table and compare."""
    print()
    print("=" * 70)
    print("[6/7] Verifying supplementary validation extras reproducibility...")
    print("=" * 70)

    pc_pipe_path = OUTPUTS_DIR / "pitch_control_soccana_tvcalib.parquet"
    pc_gt_path = OUTPUTS_DIR / "pitch_control_gt_full.parquet"
    committed_path = OUTPUTS_DIR / "validation_extras.parquet"

    for p in (pc_pipe_path, pc_gt_path, committed_path):
        if not p.exists():
            verdict = _classify_missing(p)
            print(f"  {verdict}: {p.name} not found")
            return verdict

    pc_pipe = pd.read_parquet(pc_pipe_path)
    pc_gt = pd.read_parquet(pc_gt_path)
    committed = pd.read_parquet(committed_path)
    print(f"  Committed extras: {committed.shape[0]} rows")

    print("  Re-computing validation extras...")
    recomputed = build_extras_table(pc_pipe, pc_gt)

    sort_cols = ["analysis", "row", "col"]
    committed_sorted = committed.sort_values(sort_cols).reset_index(drop=True)
    recomputed_sorted = recomputed.sort_values(sort_cols).reset_index(drop=True)

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
            obj="Validation extras recomputed vs committed",
        )
        print("  ✓ Validation extras reproduce identically")
        return PASS
    except AssertionError as e:
        print(f"  ✗ Validation extras MISMATCH:\n    {e}")
        return FAIL


def verify_spatial_pc_error() -> str:
    """Re-derive the per-cell spatial PC error map and compare (needs private inputs)."""
    print()
    print("=" * 70)
    print("[7/7] Verifying spatial PC error reproducibility...")
    print("=" * 70)

    pipe_det_path = OUTPUTS_DIR / "detections_soccana_tvcalib.parquet"
    gt_det_path = OUTPUTS_DIR / "detections_gt_full.parquet"
    balls_path = OUTPUTS_DIR / "ball_positions.parquet"
    committed_path = OUTPUTS_DIR / "spatial_pc_error.parquet"

    for p in (pipe_det_path, gt_det_path, balls_path, committed_path):
        if not p.exists():
            verdict = _classify_missing(p)
            print(f"  {verdict}: {p.name} not found")
            return verdict

    pipe_det = pd.read_parquet(pipe_det_path)
    gt_det = pd.read_parquet(gt_det_path)
    balls = pd.read_parquet(balls_path)
    committed = pd.read_parquet(committed_path)
    print(f"  Committed spatial cells: {committed.shape[0]} rows")

    print("  Re-computing per-cell spatial error...")
    recomputed = compute_cell_error(pipe_det, gt_det, balls)

    sort_cols = ["iy", "ix"]
    committed_sorted = committed.sort_values(sort_cols).reset_index(drop=True)
    recomputed_sorted = recomputed.sort_values(sort_cols).reset_index(drop=True)

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
            obj="Spatial PC error recomputed vs committed",
        )
        print("  ✓ Spatial PC error reproduces identically")
        return PASS
    except AssertionError as e:
        print(f"  ✗ Spatial PC error MISMATCH:\n    {e}")
        return FAIL


def main() -> int:
    """Run all reproducibility checks. Returns 0 unless a check FAILED."""
    print("Reproducibility Verification (Level 1: from committed parquets)")
    print("Public PC/validation/ICC parquets are committed; the raw video-derived")
    print("inputs (Soccana detections, ball positions, homographies) are NDA-private")
    print("and absent in public CI, so checks 1 and 7 are expected to SKIP there.")
    print()

    results = {
        "PC computation": verify_pc_computation(),
        "Validation statistics": verify_validation_statistics(),
        "ICC computation": verify_icc_computation(),
        "Clip-level validation": verify_clip_level_validation(),
        "pc_in_third stratified": verify_pc_in_third_stratified(),
        "Validation extras": verify_validation_extras(),
        "Spatial PC error": verify_spatial_pc_error(),
    }

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    symbol = {PASS: "✓ PASS", SKIP: "– SKIP", FAIL: "✗ FAIL"}
    any_failed = False
    for name, verdict in results.items():
        print(f"  {symbol[verdict]}: {name}")
        if verdict == FAIL:
            any_failed = True

    n_pass = sum(v == PASS for v in results.values())
    n_skip = sum(v == SKIP for v in results.values())
    n_fail = sum(v == FAIL for v in results.values())
    print(f"\n{n_pass} passed, {n_skip} skipped, {n_fail} failed.")

    if any_failed:
        print("Some reproducibility checks FAILED. See details above.")
        return 1
    if n_pass == 0:
        print("No checks could run (all inputs absent). Treating as failure.")
        return 1
    print("All runnable reproducibility checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
