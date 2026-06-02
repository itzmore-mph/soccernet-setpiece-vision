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
│   ├── 02_pitch_control.ipynb
│   ├── 03_evaluation_and_validation.ipynb
│   └── 04_visualizations.ipynb
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
│   ├── verify_reproducibility.py # SSD-free: re-derives PC/validation/ICC from parquets
│   ├── render_annotated_clips.py # Team-coloured player + orange referee bbox overlays to MP4
│   └── render_pc_overlay.py      # PC heatmap overlay on broadcast frames
├── tests/                        # pytest unit + property-based tests (hypothesis)
├── outputs/                      # Public parquets committed; NDA video-derived ones gitignored
│   ├── *.parquet                 # Committed: PC surfaces, validation, ICC, GT detections, set-pieces
│   │                             # Gitignored (NDA): detections_soccana_tvcalib, ball_positions, homographies
│   └── figures/                  # Notebook + validation figures (PNG)
├── docs/                         # Proposal PDF, thesis PDF, TVCalib setup notes
├── .github/workflows/            # CI: reproducibility check on every push
├── report.md                     # Thesis source (pandoc → PDF)
├── pyproject.toml                # Project metadata + all dependencies (uv)
├── uv.lock                       # Fully pinned, platform-aware lockfile
├── .python-version               # Python 3.11
├── CITATION.cff
└── LICENSE
```

---

## Prerequisites

- **Python 3.11** — install via any Python version manager
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — fast Python package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **SoccerNet GSR data** on external SSD (~35 GB) — only needed for full reproduction
- **Internet** — first run downloads Soccana weights from HuggingFace (~5 MB, cached)

**Note on homographies:** `outputs/homographies_tvcalib.parquet` holds pre-computed camera calibration matrices for all 33 clips (33 × 31 frames = 1,023 homographies). It is **gitignored (NDA)** because the matrices are fit to NDA video frames, so it is not in the public repo; it ships only in the closed university submission and is required only for the SSD-path pipeline (`run_optimized_pipeline.py`). **You do not need TVCalib to reproduce the public analysis** — notebooks 02–04 and the validation/ICC scripts read the committed PC parquets directly. TVCalib is needed only to regenerate the homographies from scratch.

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

Notebooks 02 and 03 read entirely from committed Parquet files and require no video data.
Notebook 04 requires SSD (broadcast frames) for its visualizations — skip or run interactively and skip Section 4–8 cells if no SSD is available.

```bash
# 1. Install dependencies (Python 3.11 required)
uv sync

# 2. Run analysis notebooks (order matters)
uv run uv run jupyter nbconvert --to notebook --execute notebooks/02_pitch_control.ipynb --inplace
uv run uv run jupyter nbconvert --to notebook --execute notebooks/03_evaluation_and_validation.ipynb --inplace
# nb04 requires SSD — run interactively or only if SSD is mounted:
uv run uv run jupyter nbconvert --to notebook --execute notebooks/04_visualizations.ipynb --inplace

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

# Generate validation tables + ICC + clip-level paired validation
uv run python scripts/ks_table_tvcalib.py
uv run python scripts/compute_icc.py
uv run python scripts/clip_level_validation.py
```

### 7. Run notebook 02 — Pitch Control

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/02_pitch_control.ipynb --inplace
```

### 8. Run notebook 03 — Evaluation & Validation

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/03_evaluation_and_validation.ipynb --inplace
```

### 9. Run notebook 04 — Visualizations

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/04_visualizations.ipynb --inplace
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

Validation statistics, ICC values, and figures reproduce identically from the committed public Parquet files, and CI re-derives them on every push (`verify_reproducibility.py` checks 2 and 3). The Pitch Control surfaces themselves can also be re-derived, but only where the NDA video-derived inputs (Soccana detections, ball positions) are present — locally with the SSD or in the university submission; that check (check 1) SKIPs in public CI by design.

```bash
uv run python scripts/verify_reproducibility.py
```

### Level 2: Full End-to-End (SSD Required)

Complete reproduction from raw SoccerNet GSR video frames requires the external SSD mounted at the path specified in `.env`. This regenerates all intermediate parquets (detections, ball positions) and produces identical outputs to the committed versions.

```bash
uv run python scripts/run_optimized_pipeline.py
uv run python scripts/run_pc_soccana_tvcalib.py
uv run python scripts/ks_table_tvcalib.py
uv run python scripts/compute_icc.py
uv run python scripts/clip_level_validation.py
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

## License

[MIT License](LICENSE).
