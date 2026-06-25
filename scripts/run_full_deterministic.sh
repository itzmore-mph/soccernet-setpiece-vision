#!/usr/bin/env bash
# Full deterministic regeneration of all outputs from scratch.
#
# Forces CPU + fixed seeds at every non-deterministic stage (YOLO detection,
# TVCalib homography) so the run is bit-reproducible: deleting all outputs and
# re-running this script yields byte-identical parquets every time, on any
# machine. mps/cuda float reductions are NOT bit-deterministic and drift
# downstream stats (see CLAUDE.md / determinism notes).
#
# Requires: SoccerNet GSR dataset (SOCCERNET_LOCAL_DIR), TVCalib sibling env.
# Runtime: long (YOLO + TVCalib on CPU). One-time canonical regen.
#
# Usage:  bash scripts/run_full_deterministic.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "project root: $ROOT"

# --- 0. Safety: back up the current canonical outputs before deleting ---
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="/tmp/outputs_canonical_${STAMP}"
echo ">>> backing up current outputs/ -> $BACKUP"
cp -r outputs "$BACKUP"

# --- 1. Clean slate ---
echo ">>> removing outputs parquets + figures"
rm -f outputs/*.parquet
rm -rf outputs/figures/*
rm -f /tmp/tvcalib_batch_out/calib.json   # force TVCalib re-inference

# --- 2. TVCalib homographies (CPU, deterministic) [needs video + tvcalib env] ---
# MUST run before detection: run_optimized_pipeline projects detections to pitch
# via these homographies (load_tvcalib_lookup), so they must exist first.
echo ">>> [1/4] tvcalib homographies"
uv run python scripts/run_tvcalib_batch.py

# --- 3. Detection + ball (CPU, deterministic) [needs video + homographies] ---
echo ">>> [2/4] detection + ball"
uv run python scripts/run_optimized_pipeline.py

# --- 4. Ground truth from GSR labels [needs dataset] ---
echo ">>> [3/4] ground truth"
uv run python scripts/dump_gt_setpieces.py
uv run python scripts/dump_gt_ball.py

# --- 5. Pitch control + analysis chain (pure numpy/pandas, deterministic) ---
echo ">>> [4/4] pitch control + analysis"
uv run python scripts/run_pc_soccana_tvcalib.py
uv run python scripts/run_pc_gt_full.py
for s in ks_table_tvcalib compute_icc clip_level_validation \
         diagnose_pc_in_third validation_extras spatial_pc_error \
         render_cohort_funnel; do
    echo "    - $s"
    uv run python "scripts/$s.py"
done

# --- 6. Notebooks (regenerate notebook-only parquets + figures) ---
echo ">>> notebooks (setpieces, gt_spatial_benchmarks, pitch_control, validation_paired)"
for n in 01_business_and_data_understanding 02_modeling_pitch_control \
         03_evaluation_and_validation 04_deployment_visualizations; do
    echo "    - $n"
    uv run jupyter nbconvert --to notebook --execute "notebooks/$n.ipynb" --inplace
done

echo ">>> DONE. canonical backup at $BACKUP"
echo ">>> verify determinism: diff regenerated outputs against the backup,"
echo "    or run this script a second time and diff the two output sets."
