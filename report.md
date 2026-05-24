---
title: "Pitch Control from Broadcast Video: A Computer Vision Pipeline for Set-Piece Analysis"
author: "Moritz Philipp Haaf"
date: "30 June 2026"
subject: "Master in Artificial Intelligence Applied to Sports - Master's Final Project"
---

# Pitch Control from Broadcast Video: A Computer Vision Pipeline for Set-Piece Analysis

**Master in Artificial Intelligence Applied to Sports**
**Master's Final Project**

Author: Moritz Philipp Haaf
MSc AI Applied to Sports · Sports Data Campus
Submission Deadline: 30 June 2026

---

## 1. Executive Summary

Optical player tracking powers modern tactical analysis but remains commercially restricted to elite clubs and leagues. This project closes a portion of that gap with an open-source computer vision pipeline that derives Pitch Control from broadcast video, requiring no proprietary tracking hardware.

**Problem.** Resource-constrained clubs (second divisions, women's football, academies) lack access to positional tracking data. Set-piece analysis — where spatial dominance determines danger — is out of reach without it.

**Solution.** A fully autonomous pipeline: Soccana (YOLOv11n, football-finetuned) detects and tracks players via ByteTrack, assigns stable team labels via per-track KMeans on HSV jersey colour, projects pixel coordinates to metric pitch via TVCalib autonomous camera calibration, and computes attacking Pitch Control using Laurie Shaw's time-to-intercept model.

**Validation.** Distributional comparison (KS test, histogram overlap) plus per-frame paired statistics against SoccerNet GSR ground-truth annotations on 33 set-piece clips (17 corners, 16 direct free kicks).

**Key results.**

- Pipeline processes all 33/33 clips end-to-end with zero homography failures.
- `pc_at_ball` (control at ball location): bias = -0.003, histogram overlap = 0.863, per-frame Pearson r = 0.682.
- Histogram overlap exceeds 0.80 on all five metrics.
- Global metrics (`pc_mean`, `pc_area_gt_0p5`) are systematically underestimated by ~0.06 due to defender under-detection in crowded penalty-area crops — a structural, explainable bias.
- Full pipeline runs on a MacBook Air M3 in ~30 minutes, no cloud dependency.

**Impact.** A broadcast-only Pitch Control pipeline that produces distributionally honest estimates of the most operationally meaningful set-piece signal, deployable by any club with broadcast video access and a laptop.

---

## 2. Introduction

### 2.1 Problem Statement

Pitch Control — the probability that a team could reach any point on the pitch first — is a standard tool for evaluating spatial dominance in elite football. Commercial systems (StatsBomb 360, SkillCorner, Tracab) deliver this data but their cost restricts access to top-tier competitions.

For the majority of professional clubs, women's leagues, academies, and scouting departments, data-driven set-piece analysis remains out of reach not because of analytical sophistication but because of data access.

### 2.2 Why Set Pieces

Set pieces (corners and direct free kicks) are tactically high-leverage and analytically tractable. Across 706 set pieces in UEFA Euro 2024, 32.4% produced a shot within 10 seconds and 1.8% produced a goal. Set pieces are also optimal for a CV pipeline: the broadcast camera is near-static, relevant players are in frame, and ball position is known from the event feed.

### 2.3 Research Gap

Prior literature covers player detection, tracking, calibration, and Pitch Control modelling individually. The end-to-end chain — broadcast pixels through to a distributionally validated tactical metric, without proprietary tracking and without ground-truth annotations leaking into the calibration step — remains underdeveloped.

### 2.4 Contribution

A reproducible, validated pipeline from broadcast video to Pitch Control, with:
- Fully autonomous operation (no GT annotations consumed at inference time)
- Distributional validation against open SoccerNet GSR annotations
- Bias diagnosis attributing residual error to detector recall in crowded scenes
- Consumer-hardware execution (MacBook Air M3)

---

## 3. Objectives

### 3.1 Primary Objective

Develop a reproducible computer vision pipeline that extracts Pitch Control from broadcast set-piece frames and produces distributions comparable to ground-truth annotation-derived distributions, using only open-source tools and consumer hardware.

### 3.2 Research Questions

- **RQ1.** Can a broadcast-video-only pipeline produce Pitch Control distributions comparable to GT-annotation distributions for set-piece frames?
- **RQ2.** What is the dominant source of systematic bias in pipeline-derived Pitch Control?
- **RQ3.** Which Pitch Control summary metrics are most robust to pipeline noise?

### 3.3 Success Criteria

| Criterion | Target |
|---|---|
| End-to-end execution | All 33 clips, MacBook Air M3, <45 min |
| Bias | Near-zero on at least one metric |
| Histogram overlap | >0.80 on majority of metrics |
| Bias explanation | Mechanistic, attributable to specific component |

---

## 4. Conceptual and Technological Architecture

### 4.1 Pipeline Overview

```
SoccerNet GSR clips (external SSD)
         |
         v
+----------------------------------+
|  Soccana + ByteTrack             |
|  - Player detection (YOLOv11n)   |
|  - Persistent track IDs          |
|  - KMeans HSV team assignment    |
+----------------------------------+
         |
         v
+----------------------------------+
|  TVCalib Homography              |
|  - Autonomous camera calibration |
|  - Pixel to metric pitch coords  |
+----------------------------------+
         |
         v
+----------------------------------+
|  Laurie Shaw Pitch Control       |
|  - Time-to-intercept model       |
|  - Zero-velocity (static frame)  |
|  - 60x40 grid on 105x68 m pitch |
+----------------------------------+
         |
         v
+----------------------------------+
|  Validation vs GT                |
|  - KS test, histogram overlap    |
|  - Per-frame paired statistics   |
+----------------------------------+
```

### 4.2 Technologies

| Component | Technology |
|---|---|
| Detection | Soccana (YOLOv11n, football-finetuned, 2.6M params) |
| Tracking | ByteTrack (via ultralytics) |
| Team assignment | KMeans on per-track mean HSV |
| Calibration | TVCalib (Theiner & Ewerth, WACV 2023) |
| Pitch Control | Laurie Shaw TTI model (Friends of Tracking) |
| Data | SoccerNet GSR 2024, StatsBomb Euro 2024 |
| Language | Python 3.11 |
| Hardware | MacBook Air M3, Apple Silicon MPS |

### 4.3 Coordinate Systems

| System | Convention |
|---|---|
| StatsBomb | 120 yd x 80 yd, origin top-left |
| Pipeline / mplsoccer | 105 m x 68 m, origin top-left |
| SoccerNet GSR bbox_pitch | centred origin (+-52.5 m, +-34 m) |

---

## 5. Methodology: CRISP-DM

The project follows CRISP-DM (Cross-Industry Standard Process for Data Mining), chosen because its iterative feedback loop mirrors the actual development trajectory.

| Phase | Notebook / Script | Output |
|---|---|---|
| Business Understanding | nb01 | Problem framing, stakeholder identification |
| Data Understanding | nb01 | StatsBomb EDA, SoccerNet GSR scan |
| Data Preparation | scripts (run_soccana_tvcalib, dump_*) | Detection parquets |
| Modeling | nb02, scripts (run_pc_*) | Pitch control surfaces |
| Evaluation | nb03, ks_table_tvcalib.py | Validation tables, figures |
| Deployment | nb04, render scripts | Visualizations, assessment |

---

## 6. Work Development

### 6.1 Phase 1: Business Understanding

**Core question:** Can a broadcast-only pipeline produce Pitch Control distributions comparable to GT for set-piece frames?

**Stakeholders:** Clubs without tracking providers (second-tier professional, women's football, academies, scouting).

**Why set pieces:** Near-static camera, all players in frame, ball position known. Of 706 Euro 2024 set pieces, 65.7% produced no shot within 10 s, 32.4% produced a shot, 1.8% produced a goal.

### 6.2 Phase 2: Data Understanding

**StatsBomb Euro 2024 (nb01).** 51 matches, 706 set-piece events (508 corners, 198 direct free kicks). 100% freeze-frame coverage. Provides distributional context for player counts and set-piece outcomes.

**SoccerNet GSR.** 33 clips identified with action_class in {Corner, Direct free-kick}: 17 corners, 16 direct free kicks. Per-frame player annotations (bbox_pitch) and pitch-line annotations. TVCalib homographies pre-computed for all 528 frames (33 clips x 16 frames).

### 6.3 Phase 3: Data Preparation

**Pipeline track (run_soccana_tvcalib.py):**
1. Soccana detection at confidence 0.40, class 0 (Player).
2. ByteTrack persistent ID assignment across frames.
3. Per-track mean HSV with KMeans (k=3, smallest-cluster drop if <15% or referee-like centroid, re-fit k=2).
4. TVCalib homography: pixel foot-point to metric pitch coordinates.
5. Output: 6,387 detection rows across 33 clips.

**GT track (dump_gt_setpieces.py):**
- SoccerNet GSR bbox_pitch annotations parsed directly.
- Centred coordinates re-centred to (0-105, 0-68).
- Output: 7,186 rows across 33 clips.

**Detection shortfall:** Pipeline produces 799 fewer rows than GT (11.1% shortfall). This is the primary source of downstream bias.

**Ball positions (dump_ball_positions.py):** 528 frame positions cached, 458 with valid ball coordinates.

### 6.4 Phase 4: Modeling

**Model:** Laurie Shaw TTI Pitch Control (zero-velocity, static-frame adaptation).

**Parameters (locked):**
- MAX_SPEED: 5.0 m/s
- REACTION_TIME: 0.7 s
- SIGMA: 0.45 s
- Grid: 60 x 40 cells on 105 x 68 m pitch

**Attacking team:** Per frame, whichever team has the player closest to the ball.

**Output:** 458 pipeline frames (30 clips), 442 GT frames (29 clips).

**Summary metrics per frame:**
- `pc_mean`: Mean attacking PC across all 2,400 grid cells
- `pc_at_ball`: PC at grid cell nearest ball position
- `pc_in_box`: Mean attacking PC within relevant penalty box
- `pc_in_third`: Mean attacking PC within relevant attacking third
- `pc_area_gt_0p5`: Fraction of cells where attacking PC > 0.5

### 6.5 Phase 5: Evaluation

**Distributional comparison (pipeline n=458, GT n=442):**

| Metric | Delta (bias) | KS stat | KS p-value | Hist. overlap | Passes KS? |
|---|---|---|---|---|---|
| pc_mean | -0.055 | 0.198 | <0.001 | 0.802 | No |
| pc_at_ball | **-0.003** | 0.100 | 0.020 | **0.863** | No |
| pc_in_box | +0.009 | 0.124 | 0.002 | 0.806 | No |
| pc_in_third | -0.042 | 0.117 | 0.004 | 0.811 | No |
| pc_area_gt_0p5 | -0.062 | 0.174 | <0.001 | 0.809 | No |

**Per-frame paired comparison (n=442 paired frames):**

| Metric | Pearson r | Spearman r | MAE | Bias |
|---|---|---|---|---|
| pc_mean | 0.246 | 0.250 | 0.135 | -0.057 |
| pc_at_ball | **0.682** | **0.603** | **0.071** | **-0.006** |
| pc_in_box | 0.382 | 0.434 | 0.226 | +0.017 |
| pc_in_third | 0.455 | 0.409 | 0.173 | -0.037 |
| pc_area_gt_0p5 | 0.219 | 0.211 | 0.152 | -0.063 |

**Bias diagnosis:** GT annotations record more player positions per frame than Soccana recovers (7,186 vs 6,387 rows, 11.1% shortfall). In the Shaw model, additional defenders compress attacking PC uniformly across the surface. `pc_at_ball` is structurally insensitive because the ball-proximate cell is dominated by the nearest attacker regardless of total defender count.

### 6.6 Phase 6: Deployment

- Pipeline runs end-to-end on MacBook Air M3 (Apple Silicon MPS), no cloud dependency.
- Runtime: ~30 minutes for 33 clips.
- Parquet outputs compatible with DuckDB, pandas, polars.
- TVCalib removes any dependency on GT pitch-line annotations.

---

## 7. Discussion of Results

### 7.1 What the pipeline gets right

`pc_at_ball` — control probability at the ball location — is preserved with near-zero bias (-0.003), high histogram overlap (0.863), and strong per-frame correlation (r=0.682). This captures whether the executing team has spatial dominance at the point of delivery, the primary determinant of set-piece danger.

`pc_in_box` and `pc_in_third` show moderate paired correlation (r=0.38-0.46) and small bias, indicating the pipeline captures penalty-area and attacking-third dominance reasonably well.

### 7.2 What the pipeline underestimates

Global metrics (`pc_mean`, `pc_area_gt_0p5`) are systematically underestimated by ~0.06. The bias is structural: Soccana detects fewer defenders per frame than GT annotations in crowded penalty-area crops. The Shaw model is sensitive to total defender count — more defenders compress attacking control across the whole surface.

This is a detection completeness issue, not a modelling error. The bias direction is predictable from the model's mathematics.

### 7.3 Why no metrics pass strict KS

No metrics pass KS at alpha=0.05. With 442 paired frames, KS has high statistical power and detects small distributional differences. The bias-and-overlap evidence shows distributions are close (overlap >0.80 on all five metrics); the strict KS rejection reflects sample size, not poor fit.

### 7.4 Practical implications

1. Use `pc_at_ball` as the primary tactical signal — it is calibrated and reliable.
2. Treat `pc_mean` and `pc_area_gt_0p5` as relative indicators for comparing set pieces within the same pipeline run, not as absolute values.
3. Monitor `n_defenders` in the output; frames with unusually low counts are lower-confidence.

### 7.5 Limitations

- **Cohort size:** 33 clips; conclusions should not be generalised beyond this cohort.
- **Static-frame assumption:** Zero-velocity Shaw TTI is appropriate for set pieces but does not extend to open play.
- **Per-frame identity ambiguity:** Pipeline track IDs are not matched to GT player IDs.
- **Structural occlusion:** Defenders in tight clusters partially hidden from broadcast angles cannot be recovered by any detector.

---

## 8. Conclusions and Future Work

### 8.1 Conclusions

1. **Broadcast-only Pitch Control is viable** for the most decision-relevant set-piece signal. Bias = -0.003 on `pc_at_ball`, histogram overlap = 0.863, per-frame r = 0.682.
2. **TVCalib autonomous calibration** enables fully autonomous operation: 33/33 clips processed, zero homography failures, no GT annotations consumed.
3. **Residual bias is structural**, attributable to detector recall in crowded penalty-area crops. The Shaw model's mathematics explain the direction of bias.
4. **ByteTrack persistent IDs** enable single-pass team assignment per clip, eliminating per-frame label instability.
5. **The pipeline is fully reproducible** on consumer hardware (MacBook Air M3, ~30 min, no cloud).

### 8.2 Future Work

- **Velocity estimation:** Optical flow between consecutive frames (exploiting ByteTrack persistent IDs) to enable the full Shaw model.
- **Cohort expansion:** Additional SoccerNet GSR clips or club-provided broadcast sets.
- **Open-play extension:** Throw-ins, goal kicks, dynamic possession sequences.
- **Detector recall tuning:** Confidence/NMS tuning to reduce defender under-detection.
- **Operational packaging:** CLI + Docker for adoption by clubs without notebook expertise.

---

## 9. Bibliography

Deliege, A., Cioppa, A., Giancola, S., et al. (2021). SoccerNet-v2: A dataset and benchmarks for holistic understanding of broadcast soccer videos. CVPRW 2021.

Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO (Version 8.0). https://github.com/ultralytics/ultralytics

Mansourian, A. M., Somers, V., De Vleeschouwer, C., & Kasaei, S. (2023). Multi-task learning for joint re-identification, team affiliation, and role classification for sports visual tracking. MMSports '23.

Shaw, L. (2020). Pitch control model (Commit 21f4c2d). Friends of Tracking Data. https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking

Somers, V., Joos, V., Giancola, S., et al. (2024). SoccerNet game state reconstruction: End-to-end athlete tracking and identification on a minimap. CVPRW 2024.

Spearman, W. (2018). Beyond expected goals. MIT Sloan Sports Analytics Conference.

StatsBomb. (2024). StatsBomb open data. https://github.com/statsbomb/open-data

Theiner, J., & Ewerth, R. (2023). TVCalib: Camera calibration for sports field registration in soccer. WACV 2023.

Zhang, Y., Sun, P., Jiang, Y., et al. (2022). ByteTrack: Multi-object tracking by associating every detection box. ECCV 2022.

---

## 10. Annexes

### Annex A: Repository Structure

```
soccernet-setpiece-vision/
    notebooks/
        01_business_and_data_understanding.ipynb
        02_pitch_control.ipynb
        03_evaluation_and_validation.ipynb
        04_visualizations.ipynb
    scripts/
        _pipeline_core.py
        download_soccernet.py
        run_tvcalib_batch.py
        run_soccana_tvcalib.py
        dump_ball_positions.py
        dump_gt_setpieces.py
        run_pc_soccana_tvcalib.py
        run_pc_gt_full.py
        ks_table_tvcalib.py
        render_annotated_clips.py
        render_pc_overlay.py
    outputs/
        homographies_tvcalib.parquet
        detections_soccana_tvcalib.parquet
        detections_gt_full.parquet
        ball_positions.parquet
        pitch_control_soccana_tvcalib.parquet
        pitch_control_gt_full.parquet
        pitch_control.parquet
        validation_summary_tvcalib.parquet
        validation_paired.parquet
        setpieces.parquet
        gt_spatial_benchmarks.parquet
        figures/
    requirements.txt
    report.md
```

### Annex B: Key Model Parameters

| Parameter | Value | Location |
|---|---|---|
| Detector | Soccana (YOLOv11n, HuggingFace) | run_soccana_tvcalib.py |
| Confidence threshold | 0.40 | _pipeline_core.py |
| Player class | 0 | run_soccana_tvcalib.py |
| Tracker | ByteTrack | _pipeline_core.py |
| KMeans k | 3 (drop smallest if <15%) | _pipeline_core.py |
| Calibration | TVCalib | homographies_tvcalib.parquet |
| PC grid | 60 x 40 | _pipeline_core.py |
| MAX_SPEED | 5.0 m/s | _pipeline_core.py |
| REACTION_TIME | 0.7 s | _pipeline_core.py |
| SIGMA | 0.45 s | _pipeline_core.py |
| Frame window | +-15 frames | _pipeline_core.py |
| KS alpha | 0.05 | nb03 |
| Histogram bins | 12 | nb03 |

### Annex C: Data Sources

| Dataset | Access |
|---|---|
| SoccerNet GSR 2024 | Credentialed download (scripts/download_soccernet.py) |
| StatsBomb Euro 2024 | statsbombpy (open, no auth) |
| Soccana weights | HuggingFace (Adit-jain/soccana) |
| TVCalib | Pre-computed, committed as homographies_tvcalib.parquet |

### Annex D: Reproducibility

**Environment:** Python 3.11, conda env `py311-dev`. Key packages: ultralytics 8.3.107, torch >=2.1.0, scipy, scikit-learn, mplsoccer, statsbombpy.

**Hardware:** MacBook Air M3, 16 GB unified memory, MPS backend.

**Runtime:** ~30 minutes for full 33-clip pipeline.

**Run order:**
```bash
python scripts/run_soccana_tvcalib.py
python scripts/dump_gt_setpieces.py
python scripts/dump_ball_positions.py
python scripts/run_pc_soccana_tvcalib.py
python scripts/run_pc_gt_full.py
python scripts/ks_table_tvcalib.py
jupyter nbconvert --execute notebooks/02_pitch_control.ipynb --inplace
jupyter nbconvert --execute notebooks/03_evaluation_and_validation.ipynb --inplace
jupyter nbconvert --execute notebooks/04_visualizations.ipynb --inplace
```
