# soccernet-setpiece-vision

**Pitch Control from Broadcast Video: A Computer Vision Pipeline for Set-Piece Analysis**

Master's Final Project, MSc AI Applied to Sports, Sports Data Campus

Author: Moritz Philipp Haaf | Submission: 30 June 2026

---

## Overview

Reproducible pipeline that derives Pitch Control from broadcast video without proprietary tracking hardware or ground-truth pitch annotations. Targets set-piece situations (corners, direct free kicks) where broadcast cameras are near-static and all relevant players are in frame.

**Pipeline stages:**
1. Player detection — Soccana (YOLOv11n, football-finetuned)
2. Multi-object tracking — ByteTrack for persistent player IDs across frames
3. Team assignment — KMeans on per-track mean HSV jersey colour
4. Camera calibration — TVCalib (Theiner & Ewerth, WACV 2023) autonomous homography
5. Pitch Control — Laurie Shaw time-to-intercept model on metric pitch (105 m × 68 m)

**Validation:** Distributional comparison (KS test, histogram overlap) against SoccerNet GSR ground-truth annotations on 33 set-piece clips.

**Key result:** 33/33 clips processed end-to-end. Bias near zero on `pc_at_ball` (Δ=+0.001) and `pc_in_box` (Δ=+0.013); histogram overlap ≥ 0.81 on 4/5 metrics.

---

## Repository Structure

```
soccernet-setpiece-vision/
├── notebooks/                    # Five CRISP-DM phases
│   ├── 01_business_and_data_understanding.ipynb
│   ├── 02_data_preparation_and_pipeline.ipynb
│   ├── 03_pitch_control.ipynb
│   ├── 04_evaluation_and_validation.ipynb
│   └── 05_visualizations.ipynb
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
│   └── render_annotated_clips.py # Team-coloured bbox overlays to MP4
├── outputs/                      # Parquet outputs + figures/
├── docs/                         # Project proposal documents
├── report.md                     # Thesis source (pandoc → PDF)
├── requirements.txt              # pip freeze (lock snapshot)
├── CITATION.cff
└── LICENSE
```

---

## Prerequisites

- **Python 3.11** — install via [Miniconda](https://docs.anaconda.com/miniconda/) or any Python version manager
- **SoccerNet GSR data** on external SSD (~35 GB) — only needed for full reproduction
- **Internet** — first run downloads Soccana weights from HuggingFace (~5 MB, cached)

**Note on homographies:** `outputs/homographies_tvcalib.parquet` contains pre-computed camera calibration matrices for all 33 clips, produced by [TVCalib](https://github.com/MM4SPA/tvcalib) (Theiner & Ewerth, WACV 2023). This file is included in the project folder and used directly by the pipeline. The script `scripts/run_tvcalib_batch.py` documents exactly how it was produced — regenerating it requires the TVCalib tool set up in a sibling directory (see the script's docstring for full setup instructions).

---

## Quick Start (from committed outputs, no SSD needed)

Notebooks 03–05 read from committed Parquet files and require no video data.

```bash
# 1. Install dependencies (Python 3.11 required)
pip install -r requirements.txt

# 2. Run analysis notebooks (order matters)
jupyter nbconvert --to notebook --execute notebooks/03_pitch_control.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/04_evaluation_and_validation.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/05_visualizations.ipynb --inplace

# 3. Run validation table (from committed pipeline outputs)
python scripts/ks_table_tvcalib.py
```

Or open each notebook in JupyterLab / VS Code and run interactively.

---

## Full Reproduction (from raw video data)

Reproduces all results end-to-end from the SoccerNet GSR video clips.

### 1. Environment setup

```bash
# Create a fresh conda environment (one-time)
conda create -n py311-dev python=3.11 -y
conda activate py311-dev

# Install all dependencies
pip install -r requirements.txt
```

If you don't use conda, any Python 3.11 environment works — just run `pip install -r requirements.txt`.

### 2. Configure data paths

Create `.env` in the project root:

```env
SOCCERNET_PASSWORD=your_password_here
SOCCERNET_LOCAL_DIR=/Volumes/MPH-ExternalStorage/soccernet-gsr
```

The `SOCCERNET_LOCAL_DIR` defaults to the Mac SSD path if not set. On Windows/Linux, set it to your local mount point.

### 3. Download SoccerNet GSR data

```bash
python scripts/download_soccernet.py
```

Idempotent — skips splits already on disk.

### 4. Run notebook 01 — Business & Data Understanding

```bash
jupyter nbconvert --to notebook --execute notebooks/01_business_and_data_understanding.ipynb --inplace
```

Produces: `outputs/setpieces.parquet`, `outputs/gt_spatial_benchmarks.parquet`. Requires internet (StatsBomb API, cached after first run).

### 5. Run notebook 02 — Data Preparation & Pipeline (legacy baseline)

```bash
jupyter nbconvert --to notebook --execute notebooks/02_data_preparation_and_pipeline.ipynb --inplace
```

Requires SSD. Produces: `outputs/detections_pipeline.parquet`, `outputs/detections_gt.parquet`, `outputs/pipeline_diagnostics.parquet`.

### 6. Run the primary pipeline (Soccana + TVCalib)

```bash
# Run Soccana detector + ByteTrack + team assignment (~30 min, SSD required)
python scripts/run_soccana_tvcalib.py

# Cache ball positions and GT detections from SSD
python scripts/dump_ball_positions.py
python scripts/dump_gt_setpieces.py

# Compute pitch control surfaces (SSD-free from here)
python scripts/run_pc_soccana_tvcalib.py
python scripts/run_pc_gt_full.py

# Generate validation table
python scripts/ks_table_tvcalib.py
```

Produces:
- `outputs/detections_soccana_tvcalib.parquet`
- `outputs/ball_positions.parquet`
- `outputs/detections_gt_full.parquet`
- `outputs/pitch_control_soccana_tvcalib.parquet`
- `outputs/pitch_control_gt_full.parquet`
- `outputs/validation_summary_tvcalib.parquet`
- `outputs/figures/14_ks_table_tvcalib.png`

### 7. Run notebook 03 — Pitch Control

```bash
jupyter nbconvert --to notebook --execute notebooks/03_pitch_control.ipynb --inplace
```

Requires SSD (reads ball positions from `Labels-GameState.json`). Reads `detections_pipeline.parquet` and `detections_gt.parquet` from step 5. Produces: `outputs/pitch_control.parquet`.

### 8. Run notebook 04 — Evaluation & Validation

```bash
jupyter nbconvert --to notebook --execute notebooks/04_evaluation_and_validation.ipynb --inplace
```

SSD-free. Reads `pitch_control.parquet` from step 7. Produces: `outputs/validation_summary.parquet`, `outputs/validation_paired.parquet`.

### 9. Run notebook 05 — Visualizations

```bash
jupyter nbconvert --to notebook --execute notebooks/05_visualizations.ipynb --inplace
```

Requires SSD (reads broadcast frames for overlays). Produces: animated GIFs and static figures in `outputs/figures/`.

### 10. (Optional) Render annotated broadcast clips

```bash
python scripts/render_annotated_clips.py                  # all clips
python scripts/render_annotated_clips.py --clip SNGS-066  # single clip
```

Produces: MP4 files in `outputs/figures/annotated/`.

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

### SoccerNet Game State Reconstruction

```bibtex
@inproceedings{Somers2024SoccerNetGameState,
  title   = {{SoccerNet} Game State Reconstruction: End-to-End Athlete Tracking and Identification on a Minimap},
  author  = {Somers, Vladimir and Joos, Victor and Giancola, Silvio and Cioppa, Anthony
             and Ghasemzadeh, Seyed Abolfazl and Magera, Floriane and Standaert, Baptiste
             and Mansourian, Amir Mohammad and Zhou, Xin and Kasaei, Shohreh
             and Ghanem, Bernard and Alahi, Alexandre
             and Van Droogenbroeck, Marc and De Vleeschouwer, Christophe},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  month   = {Jun},
  year    = {2024},
  address = {Seattle, WA, USA},
}
```

### TVCalib

```bibtex
@inproceedings{Theiner2023TVCalib,
  title     = {{TVCalib}: Camera Calibration for Sports Field Registration in Soccer},
  author    = {Theiner, Jonas and Ewerth, Ralph},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2023},
}
```

### Pitch Control model

Shaw, L. (2020). *Friends of Tracking: Pitch Control implementation*. GitHub. Reference commit: `21f4c2d`. https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking

### ByteTrack

```bibtex
@inproceedings{zhang2022bytetrack,
  title   = {{ByteTrack}: Multi-Object Tracking by Associating Every Detection Box},
  author  = {Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng
             and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year    = {2022},
}
```

### Other dependencies

- Jocher, G. et al. (2023). *Ultralytics YOLO*. https://github.com/ultralytics/ultralytics
- Adit Jain. *Soccana: YOLOv11n football detector*. HuggingFace: `Adit-jain/soccana`.
- StatsBomb (2024). *StatsBomb Open Data*. https://github.com/statsbomb/open-data
- Spearman, W. (2018). Beyond Expected Goals. MIT Sloan Sports Analytics Conference.

---

## License

[MIT License](LICENSE).
