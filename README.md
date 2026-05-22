# soccernet-setpiece-vision

**Pitch Control from Broadcast Video: A Computer Vision Pipeline for Set-Piece Analysis**

Master's Final Project, MSc AI Applied to Sports, Sports Data Campus

Author: Moritz Philipp Haaf | Submission: 30 June 2026

---

## Overview

Reproducible, open-source pipeline that derives Pitch Control from broadcast video without proprietary tracking hardware or ground-truth pitch annotations. Targets set-piece situations (corners, direct free kicks) where broadcast cameras are near-static and all relevant players are in frame.

**Pipeline stages:**
1. Player detection: YOLOv8x (off-the-shelf baseline) or Soccana YOLOv11n (football-finetuned, primary)
2. Multi-object tracking: ByteTrack for persistent player IDs across frames
3. Team assignment: KMeans on per-track mean HSV jersey colour, two-pass, once per clip
4. Camera calibration: TVCalib (Theiner & Ewerth, WACV 2023) autonomous homography from segmentation
5. Pitch Control: Laurie Shaw time-to-intercept model on metric pitch (105 m × 68 m)

**Validation:** Distributional comparison against SoccerNet GSR ground-truth annotations on 33 set-piece clips from the 2024 dataset. Distributional only — no frame-level matching, since SoccerNet GSR clips do not overlap with StatsBomb Euro 2024 matches.

**Key result (TVCalib + Soccana, primary configuration):** 33/33 clips processed end-to-end (zero homography failures vs 13/33 dropped in GT-line baseline). Bias near zero on `pc_at_ball` (Δ=+0.001) and `pc_in_box` (Δ=+0.013); histogram overlap ≥ 0.81 on 4/5 metrics. Strict KS regresses (1/5 → 0/5) due to inflated power on n=457 vs n=286 frames; bias and overlap improve everywhere. Full ablation table in `report.md`.

---

## Repository Structure

```
soccernet-setpiece-vision/
├── notebooks/                # Five CRISP-DM phases, run in order 01 → 05
│   ├── 01_business_and_data_understanding.ipynb
│   ├── 02_data_preparation_and_pipeline.ipynb
│   ├── 03_pitch_control.ipynb
│   ├── 04_evaluation_and_validation.ipynb
│   └── 05_visualizations.ipynb
├── scripts/                  # Reproducibility scripts (run after notebooks for ablations)
│   ├── download_soccernet.py
│   ├── dump_ball_positions.py
│   ├── dump_gt_setpieces.py
│   ├── repair_setpieces_freeze_frames.py
│   ├── render_annotated_clips.py    # Render team-coloured bbox overlays to MP4
│   ├── _pipeline_core.py            # Shared helpers (TVCalib scripts only, not for notebooks)
│   │
│   ├── # Detector ablation (YOLOv8x vs Soccana under GT-line H)
│   ├── run_soccana_ablation.py
│   ├── run_pc_soccana.py
│   ├── compare_detectors.py
│   ├── ablation_ks_table.py
│   │
│   ├── # TVCalib autonomous H pipeline (replaces GT-line homography leak)
│   ├── tvcalib_rmse_check.py        # Phase 1: 5-frame sanity vs GT-line H
│   ├── run_tvcalib_batch.py         # Phase 2: batch H over 33 clips × 16 frames
│   ├── run_pipeline_tvcalib.py      # Phase 3: YOLOv8x detections under TVCalib H
│   ├── run_pc_tvcalib.py            # Phase 4: PC for YOLOv8x + TVCalib
│   ├── run_pc_gt_full.py            # Phase 4: GT PC over 33-clip cohort
│   ├── ks_table_tvcalib.py          # Phase 4: 3-way KS table
│   ├── run_soccana_tvcalib.py       # Phase 5: Soccana detections under TVCalib H
│   └── run_pc_soccana_tvcalib.py    # Phase 5: PC for Soccana + TVCalib (primary config)
├── outputs/                  # All Parquet outputs + figures/
├── docs/                     # Project proposal and reference documents
├── CITATION.cff
├── CLAUDE.md
├── LICENSE
├── README.md
├── report.md                 # Thesis source (pandoc → PDF)
└── requirements.txt          # pip freeze of py311-dev (lock snapshot, not curated)
```

Video data (~35 GB) lives on an external SSD (`SOCCERNET_LOCAL_DIR`), never in the repo. TVCalib lives in a sibling directory `../tvcalib/` with its own venv — see [TVCalib Setup](#tvcalib-setup-phases-13-only) below.

---

## Setup

Requires **Python 3.11**. Install dependencies:

```bash
pip install -r requirements.txt
```

Open notebooks in any Jupyter-compatible environment (JupyterLab, VS Code, PyCharm) and select a Python 3.11 kernel, or run non-interactively:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_business_and_data_understanding.ipynb --inplace
```

```bash
# Download SoccerNet GSR (requires SOCCERNET_PASSWORD in .env)
python scripts/download_soccernet.py
```

Notebooks run in order (01 → 05). Outputs cached as Parquet so nb03–nb05 re-execute offline. Soccana weights auto-fetched from HuggingFace (`Adit-jain/soccana`) on first ablation run; YOLOv8x weights auto-downloaded by ultralytics.

---

## Reproducibility Without the Raw Data

All intermediate outputs are committed to the repo. Anyone without access to the SoccerNet GSR video clips can reproduce every result downstream of detection by checking out this repo and running:

```bash
conda activate py311-dev
jupyter nbconvert --to notebook --execute notebooks/03_pitch_control.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/04_evaluation_and_validation.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/05_visualizations.ipynb --inplace
```

These notebooks read from committed Parquet files in `outputs/` and write all figures to `outputs/figures/`. No SSD or internet access required beyond the initial `conda activate`.

Notebook 01 fetches StatsBomb Open Data via `statsbombpy` (auto-cached to `~/.cache/statsbombpy/`) and produces `outputs/setpieces.parquet` and `outputs/gt_spatial_benchmarks.parquet` (GT player spatial statistics used as validation benchmarks in nb04). Notebook 02 requires the SSD and produces the detection Parquets; its committed outputs are already in `outputs/` so nb02 re-execution is optional for result verification.

**Scripts that require the SSD** (phases 1–3 of both ablations): `download_soccernet.py`, `run_soccana_ablation.py`, `run_pipeline_tvcalib.py`, `run_soccana_tvcalib.py`, `tvcalib_rmse_check.py`, `run_tvcalib_batch.py`, `render_annotated_clips.py`.

**Scripts that are SSD-free** (phases 4–5): `run_pc_*.py`, `run_pc_gt_full.py`, `ablation_ks_table.py`, `ks_table_tvcalib.py`, `dump_ball_positions.py` (reads committed Parquet).

---

## TVCalib Setup (Phases 1–3 Only)

TVCalib is only needed to re-run camera calibration from raw frames. If you are reproducing results from the committed Parquet outputs, skip this section.

TVCalib lives in a **sibling directory** outside this repo with its own Python venv. It is not a submodule of this project.

```bash
# 1. Clone TVCalib alongside this repo
cd ..
git clone https://github.com/MM4SPA/tvcalib.git
cd tvcalib
git submodule update --init --recursive
# submodule: sn_segmentation at commit ffdb308

# 2. Create the TVCalib venv (Python 3.11, separate from py311-dev)
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.1.* torchvision kornia==0.8.2 pytorch-lightning==2.6.1
pip install SoccerNet==0.1.62 opencv-python numpy

# 3. Download the segmentation checkpoint (~466 MB)
mkdir -p data/segment_localization
# checkpoint: tvcalib/data/segment_localization/train_59.pt
# Download from the MM4SPA/tvcalib release or HuggingFace model card
# (see upstream README for the current download link)

# 4. Apply two patches for PyTorch 2.x compatibility
```

**Patch 1** — `tvcalib/sncalib_dataset.py`, line 13: replace the broken import:
```python
# before
from torch._six import string_classes
# after
string_classes = (str, bytes)
```

**Patch 2** — `tvcalib/inference.py`, line 89: add `weights_only=False`:
```python
# before
checkpoint = torch.load(checkpoint_path, map_location=device)
# after
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
```

After setup, the scripts in this repo invoke TVCalib by calling `tvcalib/run_inference.py` via subprocess. No import of TVCalib into `py311-dev` is needed.

---

## Reproducing the Ablations

After running the five notebooks, two ablations extend the analysis:

**Detector ablation (YOLOv8x vs Soccana, fixed H):**
```bash
python scripts/run_soccana_ablation.py
python scripts/run_pc_soccana.py
python scripts/ablation_ks_table.py
```

**H-source ablation (GT-line baseline vs TVCalib autonomous):**
```bash
python scripts/tvcalib_rmse_check.py            # Phase 1 sanity
python scripts/run_tvcalib_batch.py             # Phase 2 batch H
python scripts/run_pipeline_tvcalib.py          # Phase 3 detections
python scripts/dump_ball_positions.py
python scripts/dump_gt_setpieces.py
python scripts/run_pc_tvcalib.py
python scripts/run_pc_gt_full.py
python scripts/ks_table_tvcalib.py              # Phase 4 KS
python scripts/run_soccana_tvcalib.py           # Phase 5 best combo
python scripts/run_pc_soccana_tvcalib.py
python scripts/ks_table_tvcalib.py              # auto-includes Soccana row
```

---

## Coordinate Systems

| System | Convention |
|---|---|
| StatsBomb | 120 yards × 80 yards, origin top-left |
| Pipeline / mplsoccer | 105 m × 68 m, origin top-left |
| SoccerNet GSR `bbox_pitch` | centred origin (±52.5 m, ±34 m) |

`x_m = x_sb × (105/120)`, `y_m = y_sb × (68/80)`
GSR centred → Pipeline: `x_m = x_gsr + 52.5`, `y_m = y_gsr + 34`

---

## Citations

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

### SoccerNet Game State Reconstruction (dataset and annotations)

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

### TVCalib (autonomous camera calibration)

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

### ByteTrack (multi-object tracking)

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

- Jocher, G. et al. (2023). *Ultralytics YOLOv8 / YOLOv11*. https://github.com/ultralytics/ultralytics
- Adit Jain. *Soccana: YOLOv11n football detector*. HuggingFace: `Adit-jain/soccana`.
- StatsBomb (2024). *StatsBomb Open Data*. https://github.com/statsbomb/open-data
- Spearman, W. (2018). Beyond Expected Goals. MIT Sloan Sports Analytics Conference.

---

## License

[MIT License](LICENSE).
