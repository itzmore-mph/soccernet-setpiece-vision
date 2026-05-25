# soccernet-setpiece-vision

**Pitch Control from Broadcast Video: A Computer Vision Pipeline for Set-Piece Analysis**

Master's Final Project, MSc AI Applied to Sports, Sports Data Campus

Author: Moritz Philipp Haaf | Submission: 30 June 2026

---

## Overview

Reproducible pipeline that derives Pitch Control from broadcast video without proprietary tracking hardware or ground-truth pitch annotations. Targets set-piece situations (corners, direct free kicks) where broadcast cameras are near-static and all relevant players are in frame.

**Pipeline stages:**
1. Player + Referee detection — Soccana (YOLOv11n, football-finetuned), classes 0 (Player) and 2 (Referee) in a single pass
2. Multi-object tracking — ByteTrack for persistent IDs across frames
3. Team assignment — KMeans on per-track mean HSV jersey colour (players only; referees excluded with `team=-1`)
4. Camera calibration — TVCalib (Theiner & Ewerth, WACV 2023) autonomous homography
5. Pitch Control — Laurie Shaw time-to-intercept model on metric pitch (105 m × 68 m); referees excluded from computation

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
│   ├── _pipeline_core.py         # Shared module (detection, tracking, PC model)
│   ├── download_soccernet.py     # Data download (idempotent)
│   ├── run_tvcalib_batch.py      # Homography computation (requires TVCalib, see note)
│   ├── run_soccana_tvcalib.py    # Soccana detections under TVCalib H
│   ├── dump_ball_positions.py    # Cache ball positions from SSD
│   ├── dump_gt_setpieces.py      # GT detections for all 33 clips
│   ├── run_pc_soccana_tvcalib.py # Pitch control (pipeline)
│   ├── run_pc_gt_full.py         # Pitch control (GT reference)
│   ├── ks_table_tvcalib.py       # Validation table + figure
│   ├── render_annotated_clips.py # Team-coloured player + orange referee bbox overlays to MP4
│   └── render_pc_overlay.py      # PC heatmap overlay on broadcast frames
├── outputs/                      # Parquet outputs + figures/
├── docs/                         # Project proposal documents
├── report.md                     # Thesis source (pandoc → PDF)
├── requirements.txt              # Direct dependencies with version ranges
├── CITATION.cff
└── LICENSE
```

---

## Prerequisites

- **Python 3.11** — install via [Miniconda](https://docs.anaconda.com/miniconda/) or any Python version manager
- **SoccerNet GSR data** on external SSD (~35 GB) — only needed for full reproduction
- **Internet** — first run downloads Soccana weights from HuggingFace (~5 MB, cached)

**Note on homographies:** `outputs/homographies_tvcalib.parquet` is committed and contains pre-computed camera calibration matrices for all 33 clips (33 clips × 31 frames = 1,023 homographies). All downstream scripts read directly from this file. **You do not need TVCalib to run the pipeline or reproduce the analysis** — only to regenerate the homographies from scratch.

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
python scripts/run_tvcalib_batch.py
# Expected: "discovered 33 set-piece clips" → stages frames → runs TVCalib → saves parquet
```

---

## Quick Start (from committed outputs, no SSD needed)

Notebooks 02 and 03 read entirely from committed Parquet files and require no video data.
Notebook 04 requires SSD (broadcast frames) for its visualizations — skip or run interactively and skip Section 4–8 cells if no SSD is available.

```bash
# 1. Install dependencies (Python 3.11 required)
pip install -r requirements.txt

# 2. Run analysis notebooks (order matters)
jupyter nbconvert --to notebook --execute notebooks/02_pitch_control.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/03_evaluation_and_validation.ipynb --inplace
# nb04 requires SSD — run interactively or only if SSD is mounted:
jupyter nbconvert --to notebook --execute notebooks/04_visualizations.ipynb --inplace

# 3. Run validation table (from committed pipeline outputs)
python scripts/ks_table_tvcalib.py
```

Or open each notebook in JupyterLab / VS Code and run interactively.

---

## Full Reproduction (from raw video data)

Reproduces all results end-to-end from the SoccerNet GSR video clips.

### 1. Environment setup

```bash
conda create -n py311-dev python=3.11 -y
conda activate py311-dev
pip install -r requirements.txt
```

### 2. Configure data paths

Create `.env` in the project root:

```env
SOCCERNET_PASSWORD=your_password_here
SOCCERNET_LOCAL_DIR=/Volumes/MPH-ExternalStorage/soccernet-gsr
```

### 3. Download SoccerNet GSR data

```bash
python scripts/download_soccernet.py
```

### 4. Run notebook 01 — Business & Data Understanding

```bash
jupyter nbconvert --to notebook --execute notebooks/01_business_and_data_understanding.ipynb --inplace
```

Produces: `outputs/setpieces.parquet`, `outputs/gt_spatial_benchmarks.parquet`.

### 5. Compute TVCalib homographies

```bash
# Stages frames 1–31 per clip into /tmp, runs TVCalib (~15 min, SSD required)
# Requires TVCalib set up in ../tvcalib/ — see scripts/run_tvcalib_batch.py docstring
python scripts/run_tvcalib_batch.py
```

Produces: `outputs/homographies_tvcalib.parquet`. Skip this step if you want to use the committed pre-computed homographies.

### 6. Run the pipeline (Soccana + TVCalib)

```bash
# Soccana detector + ByteTrack + team assignment (~30 min, SSD required)
python scripts/run_soccana_tvcalib.py

# GT detections must run before ball positions (ball positions reads both parquets)
python scripts/dump_gt_setpieces.py
python scripts/dump_ball_positions.py

# Compute pitch control surfaces (SSD-free from here)
python scripts/run_pc_soccana_tvcalib.py
python scripts/run_pc_gt_full.py

# Generate validation table
python scripts/ks_table_tvcalib.py
```

### 7. Run notebook 02 — Pitch Control

```bash
jupyter nbconvert --to notebook --execute notebooks/02_pitch_control.ipynb --inplace
```

### 8. Run notebook 03 — Evaluation & Validation

```bash
jupyter nbconvert --to notebook --execute notebooks/03_evaluation_and_validation.ipynb --inplace
```

### 9. Run notebook 04 — Visualizations

```bash
jupyter nbconvert --to notebook --execute notebooks/04_visualizations.ipynb --inplace
```

Requires SSD (reads broadcast frames for overlays).

### 10. (Optional) Render annotated broadcast clips

```bash
python scripts/render_annotated_clips.py
python scripts/render_pc_overlay.py
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
