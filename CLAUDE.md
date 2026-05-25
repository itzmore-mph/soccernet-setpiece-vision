# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MSc final project: Pitch Control from broadcast video for set-piece analysis. Pipeline: Soccana (YOLOv11n) detection → ByteTrack tracking → KMeans team assignment → TVCalib homography → Laurie Shaw pitch control model. Validation against SoccerNet GSR ground-truth on 33 clips.

## Environment

```bash
conda activate py311-dev  # Python 3.11
pip install -r requirements.txt
```

`.env` required for full reproduction:
```
SOCCERNET_PASSWORD=...
SOCCERNET_LOCAL_DIR=/Volumes/MPH-ExternalStorage/soccernet-gsr
```

## Common Commands

```bash
# Lint and format
ruff check scripts/
black scripts/

# Run a single pipeline script
python scripts/run_soccana_tvcalib.py        # ~30 min, SSD required
python scripts/run_pc_soccana_tvcalib.py     # SSD-free
python scripts/ks_table_tvcalib.py           # validation table + figure

# Execute notebooks non-interactively (order matters: 02 → 03 → 04)
jupyter nbconvert --to notebook --execute notebooks/02_pitch_control.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/03_evaluation_and_validation.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/04_visualizations.ipynb --inplace
```

## Architecture

**`scripts/_pipeline_core.py`** — only shared module. All pipeline scripts import from it. Do not import it from notebooks.

Key functions in `_pipeline_core.py`:
- `discover_setpiece_clips(gsr_root)` — scans SSD for Corner/Direct free-kick clips
- `load_tvcalib_lookup(outputs_dir)` — reads `homographies_tvcalib.parquet`, returns `(split, clip_id, frame_idx) → H` where H maps image → pitch (top-left origin, metres)
- `run_clip(clip, yolo, H_lookup)` — processes one clip end-to-end: detect → track → HSV sample → KMeans team assignment → project to pitch
- `pitch_control_surface(att_xy, def_xy, ball_xy)` — Laurie Shaw TTI sigmoid model, 60×40 grid, 105m × 68m pitch
- `process_track(df, track_name, team_col, balls)` — computes PC for every frame in a detections DataFrame

**Data flow:**

```
SSD frames → run_soccana_tvcalib.py → detections_soccana_tvcalib.parquet
                                     ↓
homographies_tvcalib.parquet ────────┘
ball_positions.parquet ──────→ run_pc_soccana_tvcalib.py → pitch_control_soccana_tvcalib.parquet
detections_gt_full.parquet ──→ run_pc_gt_full.py         → pitch_control_gt_full.parquet
                                                                    ↓
                                                    ks_table_tvcalib.py → validation_summary_tvcalib.parquet
```

All intermediate state lives in `outputs/` as Parquet. `homographies_tvcalib.parquet` is committed (TVCalib is a sibling repo, not a Python package — see `scripts/run_tvcalib_batch.py` docstring for rebuild instructions).

## Coordinate Systems

| System | Convention |
|---|---|
| Pipeline / mplsoccer | 105m × 68m, origin top-left |
| SoccerNet GSR `bbox_pitch` | centred origin (±52.5m, ±34m) |
| StatsBomb | 120×80 yards, origin top-left |

Conversions: GSR → pipeline: `x = x_gsr + 52.5`, `y = y_gsr + 34`. StatsBomb → pipeline: `x = x_sb × (105/120)`, `y = y_sb × (68/80)`.

## Key Constants (`_pipeline_core.py`)

- `FRAME_WINDOW = 15` — frames before/after action position to process per clip
- `YOLO_CONF = 0.40` — detection confidence threshold
- `DEVICE = os.getenv("TORCH_DEVICE", "mps")` — override with `TORCH_DEVICE=cpu` if needed
- `GRID_NX, GRID_NY = 60, 40` — pitch control grid resolution
- `MAX_SPEED = 5.0` m/s, `REACTION_TIME = 0.7` s — Spearman model parameters

## Notebooks

Four CRISP-DM phases; notebooks 02–04 run from committed Parquet and require no SSD:

| Notebook | Phase | Key outputs |
|---|---|---|
| 01 | Business + Data Understanding | `setpieces.parquet`, `gt_spatial_benchmarks.parquet` |
| 02 | Pitch Control | figures 07–10 |
| 03 | Evaluation + Validation | figures 08–09, `validation_paired.parquet` |
| 04 | Visualizations | annotated stills, `pitch_control.parquet` |

## Report

`report.md` is pandoc markdown (thesis source). Build to PDF with pandoc if needed.
