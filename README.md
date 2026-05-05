# soccernet-setpiece-vision

**Pitch Control from Broadcast Video: A Computer Vision Pipeline for Set-Piece Analysis**

Master's Final Project — MSc AI Applied to Sports, Sports Data Campus
Author: Moritz Philipp Haaf | Submission: 30 June 2026

---

## Overview

A reproducible, open-source pipeline that derives Pitch Control from broadcast video without proprietary tracking hardware. The pipeline targets set-piece situations (corners and direct free kicks), where broadcast cameras are near-static and all relevant players are in frame.

**Pipeline stages:**
1. Player detection via YOLOv8x
2. Team assignment via KMeans on HSV jersey colours
3. Pixel-to-pitch coordinate transformation via homography (OpenCV RANSAC)
4. Pitch Control computation using Laurie Shaw's time-to-intercept model

**Validation:** Distributional comparison against SoccerNet GSR ground-truth annotations on 20 processable clips from the 2024 dataset (33 identified, 13 excluded due to homography failure).

**Key result:** `pc_at_ball` (control at the ball location) passes distributional validation at the pooled level (KS p=0.061, histogram overlap 0.857). Global surface metrics are systematically underestimated due to YOLOv8 under-detection of defenders in crowded penalty-area crops.

---

## Repository Structure

```
soccernet-setpiece-vision/
├── notebooks/
│   ├── 01_business_and_data_understanding.ipynb   # StatsBomb EDA, set-piece benchmarks
│   ├── 02_data_preparation_and_pipeline.ipynb     # YOLO, KMeans, homography
│   ├── 03_pitch_control.ipynb                     # Laurie Shaw TTI model, both tracks
│   ├── 04_evaluation_and_validation.ipynb         # KS tests, bias diagnosis
│   └── 05_visualizations.ipynb                    # Animated GIFs, broadcast stills
├── outputs/
│   ├── *.parquet                                  # All intermediate outputs
│   └── figures/                                   # PNGs and animated GIFs
├── scripts/
│   ├── download_soccernet.py                      # Idempotent SoccerNet GSR download
│   └── dump_ball_positions.py                     # Export ball positions for offline runs
├── CITATION.cff
├── CLAUDE.md
├── report.md
└── requirements.txt
```

---

## Setup

```bash
conda activate py311-dev
jupyter lab
```

SoccerNet GSR data is stored on an external SSD and not committed to the repository. Set `SOCCERNET_LOCAL_DIR` in `.env` before downloading.

```bash
# Download SoccerNet GSR (requires SOCCERNET_PASSWORD in .env)
python scripts/download_soccernet.py
```

Run notebooks in order (01 → 05). Outputs are cached as Parquet so nb03–nb05 can be re-executed offline.

---

## Coordinate Systems

| System | Convention |
|---|---|
| StatsBomb | 120 yards × 80 yards, origin top-left |
| Pipeline | 105 m × 68 m, origin top-left |
| SoccerNet GSR `bbox_pitch` | centred origin (±52.5 m, ±34 m) |

Conversion: `x_m = x_sb × (105/120)`, `y_m = y_sb × (68/80)`

---

## Citations

This project builds on the following works. See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

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

### TrackLab

```bibtex
@misc{Joos2024Tracklab,
  title        = {{TrackLab}},
  author       = {Joos, Victor and Somers, Vladimir and Standaert, Baptiste},
  journal      = {GitHub repository},
  year         = {2024},
  howpublished = {\url{https://github.com/TrackingLaboratory/tracklab}},
}
```

### PRTReid — Multi-task Re-identification and Team Affiliation

```bibtex
@inproceedings{Mansourian2023Multitask,
  title     = {Multi-task Learning for Joint Re-identification, Team Affiliation, and Role Classification for Sports Visual Tracking},
  author    = {Mansourian, Amir M. and Somers, Vladimir and De Vleeschouwer, Christophe and Kasaei, Shohreh},
  booktitle = {Proceedings of the 6th International Workshop on Multimedia Content Analysis in Sports (MMSports)},
  pages     = {103--112},
  month     = {Oct},
  year      = {2023},
  publisher = {ACM},
  address   = {Ottawa, Canada},
  doi       = {10.1145/3606038.3616172},
}
```

### Pitch Control model

Shaw, L. (2020). *Friends of Tracking: Pitch Control implementation*. GitHub. Reference commit: `21f4c2d`. https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking

### Other dependencies

- Jocher, G. et al. (2023). *Ultralytics YOLOv8*. https://github.com/ultralytics/ultralytics
- StatsBomb (2024). *StatsBomb Open Data*. https://github.com/statsbomb/open-data
- Spearman, W. (2018). Beyond Expected Goals. MIT Sloan Sports Analytics Conference.
- Nie, X. et al. (2021). A robust and efficient framework for sports-field registration. WACV 2021.

---

## License

MIT
