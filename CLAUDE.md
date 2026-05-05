# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# soccernet-setpiece-vision — Master's Final Project

## Candidate
Moritz Philipp Haaf, MSc AI Applied to Sports, Sports Data Campus
Deadline: 30 June 2026

## Data Storage
External SSD: `/Volumes/MPH-ExternalStorage/soccernet-gsr` (Mac) — path on Windows will differ.
All SoccerNet GSR video clips and annotations stored here.
Never store video data inside the project repo.

## Goal
Computer Vision pipeline for set-piece analysis in football.
Derive Pitch Control from SoccerNet GSR broadcast video.
Validate distributionally against StatsBomb 360 Euro 2024.

## Five Notebooks (CRISP-DM) — all run locally on Mac
1. notebooks/01_business_and_data_understanding.ipynb
2. notebooks/02_data_preparation_and_pipeline.ipynb
3. notebooks/03_pitch_control.ipynb
4. notebooks/04_evaluation_and_validation.ipynb
5. notebooks/05_visualizations.ipynb

## Stack (keep it simple)
- Python 3.11, conda env py311-dev
- No Poetry, no CI, no pre-commit (project rule overrides global CLAUDE.md; `pre-commit` may appear in `pip freeze` from global env but is not used here)
- `requirements.txt` is full `pip freeze` of `py311-dev`, not curated. Treat as lock snapshot, not source of truth.
- statsbombpy: Euro 2024 (competition_id=55, season_id=282)
- YOLOv8 (ultralytics): player detection on SoccerNet GSR frames
- OpenCV: homography, coordinate transformation
- KMeans (scikit-learn): team assignment via jersey HSV colour
- Pitch Control: Laurie Shaw Friends of Tracking implementation
- mplsoccer: all pitch visualisations
- Parquet: all intermediate outputs saved to outputs/

## Coordinate Systems

Three systems are in play — always be explicit about which one you are in:

| System | Convention |
|---|---|
| StatsBomb | 120 yards × 80 yards, origin top-left |
| Pipeline / mplsoccer | 105 m × 68 m, origin top-left |
| SoccerNet GSR `bbox_pitch` | centred origin (±52.5 m, ±34 m) |

StatsBomb → Pipeline: `x_m = x_sb * (105/120)`, `y_m = y_sb * (68/80)`
GSR centred → Pipeline: `x_m = x_gsr + 52.5`, `y_m = y_gsr + 34`

## Validation Strategy
Distributional, not frame-matched.
SoccerNet GSR clips != StatsBomb Euro 2024 matches.
Compare Pitch Control distributions across comparable set-piece types.

## Key Paths
| Resource | Path |
|---|---|
| Notebooks | ./notebooks |
| Outputs (Parquet) | ./outputs |
| Figures | ./outputs/figures |
| Scripts | ./scripts |
| Thesis source | ./report.md |
| SoccerNet data (Mac) | /Volumes/MPH-ExternalStorage/soccernet-gsr |

## Repo State (as of 2026-05-05)
All five notebooks exist and have been executed. Outputs directory is populated. Pipeline is functionally complete through validation.

**Notebooks:**
- `notebooks/01_business_and_data_understanding.ipynb` — complete
- `notebooks/02_data_preparation_and_pipeline.ipynb` — complete
- `notebooks/03_pitch_control.ipynb` — complete
- `notebooks/04_evaluation_and_validation.ipynb` — complete
- `notebooks/05_visualizations.ipynb` — complete (animated PC + minimap + broadcast stills)

**Scripts:**
- `scripts/download_soccernet.py` — SoccerNet GSR download (idempotent, needs SSD mounted)
- `scripts/dump_ball_positions.py` — reads `Labels-GameState.json` from the SSD and writes `outputs/ball_positions.parquet`; run this once after nb02 so that nb03/nb04 can execute offline without the SSD

**Outputs (all Parquet):**
- `outputs/ball_positions.parquet`
- `outputs/detections_gt.parquet`
- `outputs/detections_pipeline.parquet`
- `outputs/pipeline_diagnostics.parquet`
- `outputs/pitch_control.parquet`
- `outputs/setpieces.parquet`
- `outputs/validation_paired.parquet`
- `outputs/validation_summary.parquet`

**Figures** (`outputs/figures/`): 9 static PNGs + 2 animated GIFs (corner, direct free-kick).

**Thesis source:** `report.md` — Markdown with LaTeX/pandoc front-matter, renders to PDF via `pandoc report.md -o report.pdf`.

**Key validated result (for thesis context):** `pc_at_ball` passes distributional validation at the pooled level (KS p=0.061, histogram overlap 0.857). Global surface metrics (`pc_mean`, `pc_area_gt_0p5`) underestimate by ~−0.13 to −0.15 due to structural YOLO under-detection of defenders in crowded penalty-area crops — bias is explainable, not random.

**Next milestone:** written thesis/report ahead of 30 June 2026 deadline.

`scripts/download_soccernet.py` downloads task `gamestate-2024`, splits `[train, valid, test, challenge]`. Idempotent: skips splits already on disk (zip >0B or extracted dir non-empty). Honors `SOCCERNET_LOCAL_DIR` env var; defaults to SSD path.

## Commands

```bash
# Activate env
conda activate py311-dev

# Launch JupyterLab
jupyter lab

# Run a notebook non-interactively
jupyter nbconvert --to notebook --execute notebooks/01_business_and_data_understanding.ipynb --inplace

# Lint / format
ruff check .
black .

# Download SoccerNet GSR data (idempotent, needs .env with SOCCERNET_PASSWORD)
python scripts/download_soccernet.py
```

## External Caches
- StatsBomb: `~/.cache/statsbombpy/` (statsbombpy auto-cache, offline-friendly)
- YOLOv8 weights: `~/.cache/ultralytics/` (auto-download on first use; pin checkpoint name in notebook 02)
- HuggingFace (if used): `~/.cache/huggingface/`

## Editable Packages
Two sibling repos are installed as editable packages in `py311-dev`:
- `broadcast-to-tactics` at `/Users/mph/Dev/itzmore-mph/MAIS-projects/final-master-project/broadcast-to-tactics`
- `setpiece-pipeline` (this repo itself, installed as `setpiece-pipeline==0.1.0`)

If imports from `broadcast_to_tactics` fail, check that editable install is active: `pip install -e ../broadcast-to-tactics`.

## Pipeline Architecture
Each notebook is a self-contained CRISP-DM phase:

1. **01_business_and_data_understanding** — StatsBomb Euro 2024 EDA; set-piece type distribution; establish validation benchmarks.
2. **02_data_preparation_and_pipeline** — SoccerNet GSR frame extraction (OpenCV), YOLOv8 player detection, KMeans team assignment (HSV jersey colour), homography to pitch coordinates, output Parquet to `outputs/`.
3. **03_pitch_control** — Load Parquet detections, run Laurie Shaw pitch control model, aggregate control surfaces per set-piece type, save results to `outputs/`.
4. **04_evaluation_and_validation** — Distributional comparison (KS test / histogram overlap) of pipeline pitch control vs StatsBomb 360 Euro 2024 ground truth. All mplsoccer visualisations here.

## Key Design Decisions
- **No frame-level ground truth:** validation is distributional only. Never claim per-frame accuracy.
- **Coordinate systems:** StatsBomb is yards (120x80); pipeline works in metres (105x68). Conversion is `x_m = x_sb * (105/120)`, `y_m = y_sb * (68/80)`. Apply before any geometry.
- **Team assignment:** KMeans on HSV values of bounding-box crops, not jersey numbers. Works for broadcast clips where numbers aren't reliably resolved.
- **outputs/ discipline:** only `.parquet` files land here. Any intermediate frame images or video go to the external SSD or `/tmp`.
- **Pitch Control source:** vendor or pin Laurie Shaw's "Friends of Tracking" repo + commit hash in notebook 03 to lock implementation behaviour.
- **Validation thresholds:** lock KS test alpha + histogram bin count in notebook 04 for reproducibility.

## Rules
- Everything inline in notebooks, no separate src/ modules
- All paths relative to project root except SoccerNet data on external SSD
- .env for secrets (SOCCERNET_PASSWORD), never hardcoded
- Each notebook runs independently
- All four notebooks run locally on MacBook Air M3
- outputs/ contains only Parquet files, never raw video