# soccernet-setpiece-vision

**Pitch Control from Broadcast Video: A Computer Vision Pipeline for Set-Piece Analysis**

Master's Final Project, MSc AI Applied to Sports, Sports Data Campus

Author: Moritz Philipp Haaf | Submission: 30 June 2026

---

## Overview

Reproducible pipeline that derives Pitch Control from broadcast video without proprietary tracking hardware or ground-truth pitch annotations. Targets set-piece situations (corners, direct free kicks) where broadcast cameras are near-static and all relevant players are in frame.

**Pipeline stages (single video pass per clip, frames 1–250):**
1. Player + Referee detection — Soccana (YOLOv11n, football-finetuned), conf=0.25, TTA, agnostic NMS; classes 0 (Player) and 2 (Referee)
2. Ball detection — Soccana class=1, conf=0.15; gap interpolation up to 5 frames; frame-1 priority for set-piece resting position
3. Multi-object tracking — ByteTrack for persistent IDs; separate tracker instances for players and ball
4. Team assignment — global KMeans (k=3) on per-track mean HSV across 250-frame fitting window; cross-frame mode consensus per `track_id`; referees assigned `team=-1`
5. Camera calibration — TVCalib (Theiner & Ewerth, WACV 2023) autonomous homography; pitch-bounds filtering [0–105 m × 0–68 m]
6. Pitch Control — Laurie Shaw time-to-intercept model (zero-velocity, static-frame); 60×40 grid on 105 m × 68 m pitch; frames 1–31 only

**Validation:** Distributional comparison (KS test, histogram overlap) plus per-frame paired statistics against SoccerNet GSR ground-truth annotations on 33 set-piece clips.

---

## Repository Structure

```
soccernet-setpiece-vision/
├── notebooks/                    # Four CRISP-DM phases
│   ├── 01_business_and_data_understanding.ipynb
│   ├── 02_modeling_pitch_control.ipynb
│   ├── 03_evaluation_and_validation.ipynb
│   └── 04_deployment_visualizations.ipynb
├── scripts/                      # Pipeline reproduction scripts
│   ├── _pipeline_core.py         # Shared module (detection, tracking, team assignment, PC model)
│   ├── download_soccernet.py     # Data download (idempotent)
│   ├── run_tvcalib_batch.py      # Homography computation (requires TVCalib sibling env)
│   ├── run_optimized_pipeline.py # Single-pass: detections + ball + team assignment (Fixes 2–4)
│   ├── dump_gt_setpieces.py      # GT detections for all 33 clips
│   ├── run_pc_soccana_tvcalib.py # Pitch control (pipeline)
│   ├── run_pc_gt_full.py         # Pitch control (GT reference)
│   ├── ks_table_tvcalib.py       # Frame-level distributional validation table + figure
│   ├── compute_icc.py            # ICC(2,1) + effective sample size per PC metric
│   ├── clip_level_validation.py  # Clip-level paired validation: bias + bootstrap CI + Wilcoxon (n=22)
│   ├── diagnose_pc_in_third.py   # pc_in_third Simpson's-paradox diagnostic: stratified r by set-piece type
│   ├── validation_extras.py      # Bland-Altman, skill vs baseline, density, box confusion, temporal stability
│   ├── spatial_pc_error.py       # Per-cell PC error heatmap (pipeline detections in, aggregate out)
│   ├── render_cohort_funnel.py   # Cohort-attrition funnel figure (33 clips → 22 ball-position → 674 PC frames)
│   ├── render_annotated_clips.py # Team-coloured player + orange referee bbox overlays to MP4
│   └── render_pc_overlay.py      # PC heatmap overlay on broadcast frames
├── tests/                        # pytest unit + property-based tests (hypothesis)
├── outputs/                      # All numeric parquets committed; only video media is gitignored
│   ├── *.parquet                 # Committed: PC surfaces, validation, ICC, GT + pipeline detections,
│   │                             #   ball positions, homographies, set-pieces
│   └── figures/                  # Committed analysis/validation figures (PNG); rendered video
│                                 #   media (annotated/, overlay/, *.mp4, *.gif) gitignored
├── pyproject.toml                # Project metadata + all dependencies (uv)
├── uv.lock                       # Fully pinned, platform-aware lockfile
├── .python-version               # Python 3.11
├── .env.example                  # Template for SOCCERNET_PASSWORD / SOCCERNET_LOCAL_DIR
├── CITATION.cff
└── LICENSE
```

> The thesis document (PDF/DOCX) and the project proposal are not in this public repo; they are delivered separately as the closed university submission. Only the TVCalib setup notes under `docs/tvcalib-setup/` are tracked here, since they document the external calibration step.

---

## Prerequisites

- **Python 3.11** — install via any Python version manager
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — fast Python package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **SoccerNet GSR data** on external SSD (~35 GB) — only needed for full reproduction
- **Internet** — first run downloads Soccana weights from HuggingFace (~5 MB, cached)

**Note on homographies:** `outputs/homographies_tvcalib.parquet` holds pre-computed camera calibration matrices for all 33 clips (33 × 31 frames = 1,023 homographies). These are numeric matrices and are committable (see Data & Licensing below), but they must be regenerated from the SSD via TVCalib — they are not bundled here yet. They are required only for the SSD-path pipeline (`run_optimized_pipeline.py`). **You do not need TVCalib to reproduce the public analysis** — notebooks 02–04 and the validation/ICC scripts read the committed PC parquets directly. TVCalib is needed only to regenerate the homographies from scratch.

---

## TVCalib Setup (only needed to regenerate homographies)

TVCalib runs in a separate sibling directory with its own Python environment. Expected layout:

```
parent-dir/
├── soccernet-setpiece-vision/   ← this repo
└── tvcalib/                     ← TVCalib repo
```

### 1. Clone TVCalib

```bash
cd ..
git clone https://github.com/MM4SPA/tvcalib
```

### 2. Create TVCalib environment

```bash
cd tvcalib
python3.11 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install torch==2.1.* torchvision kornia==0.8.2 pytorch-lightning==2.6.1
pip install SoccerNet==0.1.62 opencv-python numpy
```

### 3. Apply PyTorch 2.x compatibility patches

Two small patches are required for TVCalib to run with PyTorch 2.x:

**`tvcalib/tvcalib/sncalib_dataset.py` — line 13:**
```python
# Replace:
from torch._six import string_classes
# With:
string_classes = (str, bytes)
```

**`tvcalib/run_inference.py` — line 89:**
```python
# Replace:
checkpoint = torch.load(path)
# With:
checkpoint = torch.load(path, weights_only=False)
```

### 4. Download segmentation checkpoint

Download `train_59.pt` from the [TVCalib releases](https://github.com/MM4SPA/tvcalib/releases) and place it at:

```
tvcalib/data/segment_localization/train_59.pt
```

### 5. Verify setup

```bash
# From soccernet-setpiece-vision/
uv run python scripts/run_tvcalib_batch.py
# Expected: "discovered 33 set-piece clips" → stages frames → runs TVCalib → saves parquet
```

---

## Quick Start (from committed outputs, no SSD needed)

All four notebooks run top-to-bottom without the SSD. Notebooks 02 and 03 reproduce the full Pitch Control and validation analysis from the committed Parquet files. Notebook 02 recomputes GT Pitch Control live from committed GT detections and loads the pipeline side from the committed PC parquet when the raw pipeline detections are absent. Notebook 04 only renders broadcast overlays, so without the SSD frames and pipeline detections it runs but its render cells skip cleanly (no figures produced); mount the SSD to generate the visualizations.

```bash
# 1. Install dependencies (Python 3.11 required)
uv sync

# 2. Run analysis notebooks (order matters)
uv run jupyter nbconvert --to notebook --execute notebooks/02_modeling_pitch_control.ipynb --inplace
uv run jupyter nbconvert --to notebook --execute notebooks/03_evaluation_and_validation.ipynb --inplace
# nb04 runs SSD-free but its render cells skip without SSD frames + pipeline detections:
uv run jupyter nbconvert --to notebook --execute notebooks/04_deployment_visualizations.ipynb --inplace

# 3. Run validation table (from committed pipeline outputs)
uv run python scripts/ks_table_tvcalib.py
```

Or open each notebook in JupyterLab / VS Code and run interactively.

---

## Full Reproduction (from raw video data)

Reproduces all results end-to-end from the SoccerNet GSR video clips.

### 1. Environment setup

```bash
uv sync
```

This installs all dependencies from `pyproject.toml` into a managed `.venv` using the pinned `uv.lock` lockfile. The lockfile resolves identically on Windows and macOS; OS-specific wheels (e.g. Apple-Silicon `torch`/`hf-xet`, Windows `pywinpty`) are selected automatically via environment markers.

### 2. Configure data paths

Create `.env` in the project root:

```env
SOCCERNET_PASSWORD=your_password_here
SOCCERNET_LOCAL_DIR=/Volumes/MPH-ExternalStorage/soccernet-gsr
```

### 3. Download SoccerNet GSR data

```bash
uv run python scripts/download_soccernet.py
```

### 4. Run notebook 01 — Business & Data Understanding

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/01_business_and_data_understanding.ipynb --inplace
```

Produces: `outputs/setpieces.parquet`, `outputs/gt_spatial_benchmarks.parquet`.

### 5. Compute TVCalib homographies

```bash
# Stages frames 1–31 per clip into /tmp, runs TVCalib (~15 min, SSD required)
# Requires TVCalib set up in ../tvcalib/ — see scripts/run_tvcalib_batch.py docstring
uv run python scripts/run_tvcalib_batch.py
```

Produces: `outputs/homographies_tvcalib.parquet`. Skip this step if you want to use the committed pre-computed homographies.

### 6. Run the pipeline (Soccana + TVCalib)

```bash
# Single-pass: player detection, ball detection, team assignment (~30 min, SSD required)
# Outputs: detections_soccana_tvcalib.parquet + ball_positions.parquet
uv run python scripts/run_optimized_pipeline.py

# GT detections (no SSD needed after this point)
uv run python scripts/dump_gt_setpieces.py

# Compute pitch control surfaces
uv run python scripts/run_pc_soccana_tvcalib.py
uv run python scripts/run_pc_gt_full.py

# Generate validation tables + ICC + clip-level + pc_in_third + extras + spatial error
uv run python scripts/ks_table_tvcalib.py
uv run python scripts/compute_icc.py
uv run python scripts/clip_level_validation.py
uv run python scripts/diagnose_pc_in_third.py
uv run python scripts/validation_extras.py
uv run python scripts/spatial_pc_error.py   # needs pipeline detections; output is an aggregate
```

### 7. Run notebook 02 — Pitch Control

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/02_modeling_pitch_control.ipynb --inplace
```

### 8. Run notebook 03 — Evaluation & Validation

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/03_evaluation_and_validation.ipynb --inplace
```

### 9. Run notebook 04 — Visualizations

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/04_deployment_visualizations.ipynb --inplace
```

Requires SSD (reads broadcast frames for overlays).

### 10. (Optional) Render annotated broadcast clips

```bash
uv run python scripts/render_annotated_clips.py
uv run python scripts/render_pc_overlay.py
```

---

## Reproducibility

This project supports two levels of reproducibility:

### Level 1: From Committed Parquets (No SSD Required)

Pitch Control, validation statistics, ICC values, and figures reproduce from the committed public Parquet files. Notebooks 02 and 03 run top-to-bottom without the SSD, and the validation/ICC scripts below re-derive the published tables from the committed PC parquets. The video-derived inputs (Soccana detections, ball positions) are not required for this level.

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/02_modeling_pitch_control.ipynb --inplace
uv run jupyter nbconvert --to notebook --execute notebooks/03_evaluation_and_validation.ipynb --inplace
uv run python scripts/ks_table_tvcalib.py
uv run python scripts/compute_icc.py
uv run python scripts/clip_level_validation.py
uv run python scripts/diagnose_pc_in_third.py
uv run python scripts/validation_extras.py
```

### Level 2: Full End-to-End (SSD Required)

Complete reproduction from raw SoccerNet GSR video frames requires the external SSD mounted at the path specified in `.env`. This regenerates all intermediate parquets (detections, ball positions) and produces identical outputs to the committed versions.

```bash
uv run python scripts/run_optimized_pipeline.py
uv run python scripts/run_pc_soccana_tvcalib.py
uv run python scripts/ks_table_tvcalib.py
uv run python scripts/compute_icc.py
uv run python scripts/clip_level_validation.py
uv run python scripts/diagnose_pc_in_third.py
uv run python scripts/validation_extras.py
uv run python scripts/spatial_pc_error.py
```

---

## Coordinate Systems

| System | Convention |
|---|---|
| StatsBomb | 120 yards × 80 yards, origin top-left |
| Pipeline / mplsoccer | 105 m × 68 m, origin top-left |
| SoccerNet GSR `bbox_pitch` | centred origin (±52.5 m, ±34 m) |

Conversions: `x_m = x_sb × (105/120)`, `y_m = y_sb × (68/80)`. GSR → Pipeline: `x_m = x_gsr + 52.5`, `y_m = y_gsr + 34`.

---

## Citations

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

- Somers et al. (2024). SoccerNet Game State Reconstruction. CVPRW 2024.
- Theiner & Ewerth (2023). TVCalib: Camera Calibration for Sports Field Registration in Soccer. WACV 2023.
- Shaw, L. (2020). Pitch Control model. Friends of Tracking Data. Commit `21f4c2d`.
- Zhang et al. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. ECCV 2022.
- Jocher et al. (2023). Ultralytics YOLO.
- Adit Jain. Soccana: YOLOv11n football detector. HuggingFace: `Adit-jain/soccana`.
- StatsBomb (2024). StatsBomb Open Data.
- Spearman, W. (2018). Beyond Expected Goals. MIT Sloan Sports Analytics Conference.

---

## Data & Licensing

The code in this repository is MIT-licensed. Data redistribution follows guidance the SoccerNet team provided in writing (2026-06-01):

- **SoccerNet GSR annotations** (`Labels-GameState.json`, `bbox_pitch`) are annotated by the SoccerNet team and open source. Outputs derived solely from them — GT detections, validation, ICC, set-pieces — are committed freely.
- **Numeric pipeline outputs** derived from the video (Soccana detections, ball positions, TVCalib homographies, Pitch Control surfaces) are committable. They contain coordinates and matrices, not video. Note these were generated from league-copyrighted frames and theoretically carry the same copyright, so they are shared for academic, non-commercial use only.
- **Raw and rendered video** is **not redistributable.** Broadcast frames are never committed, and annotated clips / GIFs / MP4s rendered from those frames are gitignored. Short (≤5 s) annotated excerpts may be shared off-repo as academic fair use, but are kept out of the public repository.

Use is academic and non-commercial. See `LICENSE` for the code license.

---

## License

[MIT License](LICENSE) (code only - see Data & Licensing above for data terms).
