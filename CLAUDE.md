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
- YOLOv8x (ultralytics): player detection on SoccerNet GSR frames
- Soccana (HuggingFace `Adit-jain/soccana`, YOLOv11n football-finetuned): ablation detector for section 6
- ByteTrack (ultralytics built-in): persistent player IDs across frames
  - Requires `lap` package; `pip install lap` if `yolo.track()` fails on import
- OpenCV: homography via RANSAC, coordinate transformation
- KMeans (scikit-learn): two-pass team assignment on per-track mean HSV jersey colour (once per clip)
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
- `notebooks/04_evaluation_and_validation.ipynb` — complete (incl. section 6 detector ablation)
- `notebooks/05_visualizations.ipynb` — complete (animated PC + minimap + broadcast stills)

**Scripts:**
- `scripts/download_soccernet.py` — SoccerNet GSR download (idempotent, needs SSD mounted)
- `scripts/dump_ball_positions.py` — reads `Labels-GameState.json` from the SSD and writes `outputs/ball_positions.parquet`; run this once after nb02 so that nb03/nb04 can execute offline without the SSD
- `scripts/run_soccana_ablation.py` — re-runs nb02 detection stage with Soccana (HF: `Adit-jain/soccana`) holding all other stages constant; writes `outputs/detections_soccana.parquet`. Needs SSD.
- `scripts/run_pc_soccana.py` — mirrors nb03 PC model on Soccana detections; writes `outputs/pitch_control_soccana.parquet`. SSD-free.
- `scripts/compare_detectors.py` — detection-count ablation (YOLOv8x vs Soccana vs GT); writes `outputs/ablation_detector_summary.parquet` + figure 11.
- `scripts/ablation_ks_table.py` — standalone KS table for the PC ablation; writes `outputs/ablation_ks_summary.parquet` + figure 12.
- `scripts/repair_setpieces_freeze_frames.py` — re-fetches StatsBomb 360 freeze frames per match if nb01 produced 0% coverage. Use only if `outputs/setpieces.parquet` shows empty `freeze_frame` arrays.
- `scripts/_patch_nb01_cell15.py` — internal helper that patches a specific cell in nb01 (not for general use).
- `scripts/_patch_nb04_add_ablation.py` — internal helper that adds the detector ablation section to nb04 (not for general use).

**Outputs (all Parquet):**
- `outputs/ball_positions.parquet`
- `outputs/detections_gt.parquet`
- `outputs/detections_pipeline.parquet`
- `outputs/detections_soccana.parquet`             # ablation detections
- `outputs/pipeline_diagnostics.parquet`
- `outputs/pitch_control.parquet`
- `outputs/pitch_control_soccana.parquet`          # ablation PC surfaces
- `outputs/setpieces.parquet`
- `outputs/validation_paired.parquet`
- `outputs/validation_summary.parquet`
- `outputs/ablation_detector_summary.parquet`      # detection-count comparison
- `outputs/ablation_ks_summary.parquet`            # PC KS table per detector

**Figures** (`outputs/figures/`): 12 static PNGs + 2 animated GIFs (corner, direct free-kick). Figures 11-12 cover the detector ablation (count distributions, KS table, histogram overlays).

**Clip cohort:** 33 set-piece clips in SoccerNet GSR 2024; 20 processable end-to-end with YOLOv8x (18 paired with GT in PC validation); 13 excluded (homography failure); 2 further excluded from visualisations (annotation errors). Soccana ablation processes the same cohort.

**Thesis source:** `report.md` — Markdown with LaTeX/pandoc front-matter, renders to PDF via `pandoc report.md -o report.pdf`.

**Key validated result (for thesis context):** `pc_at_ball` passes distributional validation at the pooled level (KS p=0.202, hist overlap 0.864). Global surface metrics (`pc_mean`, `pc_area_gt_0p5`) underestimate by ~−0.17 to −0.19. Detector ablation with Soccana (YOLOv11n football-finetuned) reduces global-metric bias by ~30% but does not flip any KS-fail to KS-pass — bias decomposes ~30% detector-domain / ~70% structural occlusion.

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

# Detector ablation pipeline (after nb02 has run)
python scripts/run_soccana_ablation.py
python scripts/run_pc_soccana.py
python scripts/ablation_ks_table.py

# Register py311-dev as Jupyter kernel (one-time, after fresh env)
python -m ipykernel install --user --name py311-dev --display-name "Python (py311-dev)"
```

## External Caches
- StatsBomb: `~/.cache/statsbombpy/` (statsbombpy auto-cache, offline-friendly)
- YOLOv8 weights: `~/.cache/ultralytics/` (auto-download on first use; pin checkpoint name in notebook 02)
- HuggingFace (if used): `~/.cache/huggingface/`

## Editable Packages
- `setpiece-pipeline` — this repo itself, installed as `setpiece-pipeline==0.1.0` (`pip install -e .`)
- `broadcast-to-tactics` — sibling repo at `/Users/mph/Dev/itzmore-mph/MAIS-projects/final-master-project/broadcast-to-tactics` (Mac path). If imports from `broadcast_to_tactics` fail: `pip install -e ../broadcast-to-tactics`.

## Pipeline Architecture
Each notebook is a self-contained CRISP-DM phase:

1. **01_business_and_data_understanding** — StatsBomb Euro 2024 EDA; set-piece type distribution; establish validation benchmarks.
2. **02_data_preparation_and_pipeline** — SoccerNet GSR frame extraction (OpenCV), YOLOv8x player detection, ByteTrack for persistent IDs, two-pass KMeans team assignment on per-track mean HSV (once per clip), RANSAC homography to pitch coordinates, output Parquet to `outputs/`.
3. **03_pitch_control** — Load Parquet detections, run Laurie Shaw pitch control model, aggregate control surfaces per set-piece type, save results to `outputs/`.
4. **04_evaluation_and_validation** — Distributional comparison (KS test / histogram overlap) of pipeline pitch control vs StatsBomb 360 Euro 2024 ground truth, plus bias diagnostics.
5. **05_visualizations** — Animated pitch-control GIFs, broadcast stills overlay, minimap renders. All mplsoccer outputs land in `outputs/figures/`.

## Detector Ablation
A second detection pass swaps YOLOv8x (COCO-pretrained) for Soccana (`Adit-jain/soccana`, YOLOv11n finetuned on SoccerNet + match footage). All other pipeline stages are held constant: ByteTrack, KMeans-HSV teams, RANSAC homography, Laurie Shaw PC model, and locked nb04 thresholds. Soccana weights live at `Model/weights/best.pt` inside the HF repo and are fetched via `huggingface_hub.hf_hub_download` (cached at `~/.cache/huggingface/`). Soccana class IDs: 0=Player, 1=Ball, 2=Referee — pipeline filters to class 0 only.

Run order:
1. `python scripts/run_soccana_ablation.py`   (~30-60 min, SSD required)
2. `python scripts/run_pc_soccana.py`         (seconds)
3. `python scripts/ablation_ks_table.py`      (or re-execute nb04 section 6)

## Key Design Decisions
- **No frame-level ground truth:** validation is distributional only. Never claim per-frame accuracy.
- **Coordinate systems:** StatsBomb is yards (120x80); pipeline works in metres (105x68). Conversion is `x_m = x_sb * (105/120)`, `y_m = y_sb * (68/80)`. Apply before any geometry.
- **Team assignment:** KMeans on HSV values of bounding-box crops, not jersey numbers. Works for broadcast clips where numbers aren't reliably resolved.
- **outputs/ discipline:** only `.parquet` files land here. Any intermediate frame images or video go to the external SSD or `/tmp`.
- **Pitch Control source:** vendor or pin Laurie Shaw's "Friends of Tracking" repo + commit hash in notebook 03 to lock implementation behaviour.
- **Validation thresholds:** lock KS test alpha + histogram bin count in notebook 04 for reproducibility.

## Rules
- Everything inline in notebooks, no separate src/ modules
- Use `py311-dev` conda env exclusively. Do NOT use `.venv` — observed kernel/version drift between the two breaks ultralytics weight loading on PyTorch ≥2.6.
- On Mac: verify `which python` resolves to `/Users/mph/miniconda3/envs/py311-dev/bin/python` before running scripts.
- All paths relative to project root except SoccerNet data on external SSD
- .env for secrets (SOCCERNET_PASSWORD), never hardcoded
- Each notebook runs independently
- All five notebooks run locally on MacBook Air M3; repo is also cloned on Windows (c:\Users\morhaaf\Dev\Git\) but execution is Mac-primary
- outputs/ contains only Parquet files, never raw video