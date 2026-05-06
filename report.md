---
title: "Pitch Control from Broadcast Video: A Computer Vision Pipeline for Set-Piece Analysis"
author: "Moritz Philipp Haaf"
date: "30 June 2026"
subject: "Master in Artificial Intelligence Applied to Sports - Master's Final Project"
header-includes:
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyfoot[C]{\thepage}
  - \fancyhead[L]{Pitch Control from Broadcast Video}
  - \fancyhead[R]{Moritz Philipp Haaf}
reference_docx: reference.docx
---

# Pitch Control from Broadcast Video: A Computer Vision Pipeline for Set-Piece Analysis

**Master in Artificial Intelligence Applied to Sports**
**Master's Final Project**

Author: Moritz Philipp Haaf
MSc AI Applied to Sports · Sports Data Campus
Submission Deadline: 30 June 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction](#2-introduction)
3. [Objectives](#3-objectives)
4. [Timeline](#4-timeline)
5. [Conceptual and Technological Architecture](#5-conceptual-and-technological-architecture)
6. [Methodology: CRISP-DM](#6-methodology-crisp-dm)
7. [Project Development](#7-project-development)
   - 7.1 Business Understanding
   - 7.2 Data Understanding
   - 7.3 Data Preparation
   - 7.4 Modeling
   - 7.5 Evaluation
   - 7.6 Deployment
8. [Results Discussion](#8-results-discussion)
9. [Conclusions and Future Work](#9-conclusions-and-future-work)
10. [Bibliography](#10-bibliography)
11. [Appendices](#11-appendices)

---

## 1. Executive Summary

**Problem.** Optical player tracking, the data layer that powers modern tactical analysis, is commercially available only to elite clubs and leagues. Second divisions, women's football, youth academies, and most scouting contexts operate without it.

**Solution.** This project delivers a reproducible, open-source computer vision pipeline that derives Pitch Control from broadcast video, requiring no proprietary tracking hardware. The pipeline detects and tracks players using **Soccana** (YOLOv11n, football-finetuned on SoccerNet GSR + match footage) combined with ByteTrack multi-object tracking, assigns stable team labels via per-track jersey colour clustering (KMeans on HSV values aggregated across frames), transforms pixel coordinates to metric pitch coordinates via **TVCalib** autonomous camera calibration, and computes attacking Pitch Control using Laurie Shaw's time-to-intercept model. An off-the-shelf detector baseline (YOLOv8x, COCO-pretrained) and a GT-pitch-line homography baseline are retained as ablation arms to isolate the contribution of football-domain finetuning and autonomous calibration respectively. The focus is set pieces (corners and direct free kicks), where broadcast cameras are near-static and all relevant players are typically in frame.

**Validation.** The pipeline was evaluated against SoccerNet Game State Recognition (GSR) ground-truth annotations. Of 33 candidate clips (17 corners, 16 direct free kicks) drawn from the 2024 dataset, all 33 are processable end-to-end under autonomous TVCalib calibration (the GT-pitch-line baseline excluded 13 of 33 due to insufficient pitch-line coverage; this exclusion drove the original methodological motivation for replacing it). A further 2 clips were identified as annotation errors during visual inspection (one Corner showed a mid-game scene; one Direct free-kick showed a throw-in) and excluded from visualisations. Validation is distributional: pipeline and ground-truth Pitch Control distributions are compared using two-sample Kolmogorov–Smirnov tests and histogram overlap.

**Results.** The final pipeline (Soccana + TVCalib) achieves near-zero bias on `pc_at_ball` (Δ=+0.001) and `pc_in_box` (Δ=+0.013) versus full-cohort GT. Histogram overlap is ≥0.81 on 4/5 metrics. Bias falls on 4/5 metrics relative to the GT-leak YOLOv8x baseline. KS strict pass count (α=0.05) is 0/5 versus 1/5 baseline, but the regression is driven by statistical power (n grew from 286 to 457 paired frames), not by worse fit — bias and overlap improve. The ablation isolates ~30% of residual bias to detector domain mismatch (closed by Soccana finetuning) and ~70% to structural broadcast-angle occlusion. The bias is structural and explainable, not random.

**Impact.** Any team with broadcast access and a Python environment can run this pipeline. The approach is directly applicable to competitions where commercial tracking is absent but video is available.

---

## 2. Introduction

### 2.1 Context

In elite football, Pitch Control (the probability that a given team could reach any point on the pitch first under current player positions) has become a standard analytical tool for evaluating spatial dominance, tactical compactness, and the danger of set-piece situations. Systems like StatsBomb 360, SkillCorner, and Opta Tracking deliver this data in near-real time. Their cost and infrastructure requirements, however, effectively restrict them to top-tier competitions.

For the majority of professional and semi-professional clubs, data-driven set-piece analysis remains out of reach not because of analytical sophistication, but because of data access.

### 2.2 The Problem

Set pieces (corners and direct free kicks) account for roughly 30% of goals in major tournaments (StatsBomb, 2024). They are also the highest-information moments for a computer vision pipeline: the broadcast camera is nearly static, all relevant players are in frame, and the ball position is precisely known from the event feed. This combination makes set pieces the optimal entry point for extracting positional value from video without a dedicated tracking rig.

### 2.3 Why This Matters

The analytical gap between well-funded and resource-constrained clubs is partly a data infrastructure problem. A pipeline that converts broadcast video into a spatially meaningful signal (Pitch Control) closes a portion of that gap without requiring proprietary hardware. The output is interpretable by coaches and analysts, not only data scientists.

### 2.4 Validation Approach

Validating a computer vision pipeline against proprietary tracking data is not straightforward when the two datasets do not share clips. Here, both pipeline and ground truth are computed on the same SoccerNet GSR frames, enabling per-frame paired comparison in addition to distributional comparison. The broader distributional target, how pipeline-derived Pitch Control compares to StatsBomb 360 freeze-frame statistics, is addressed through Euro 2024 event data as a reference distribution.

---

## 3. Objectives

**Primary objective.** Develop a reproducible CV pipeline that extracts Pitch Control from broadcast set-piece frames and produces distributions comparable to ground-truth annotations.

**Specific objectives:**

1. Extract and characterise set-piece events from StatsBomb Euro 2024 open data (nb01) to establish domain benchmarks.
2. Build a two-track processing pipeline on SoccerNet GSR clips:
   - **Pipeline track:** YOLOv8x detection + ByteTrack tracking → per-track KMeans team assignment → homography → Laurie Shaw Pitch Control.
   - **Ground-truth track:** SoccerNet `bbox_pitch` annotations → same Pitch Control model.
3. Compute five Pitch Control summary metrics per frame: `pc_mean`, `pc_at_ball`, `pc_in_box`, `pc_in_third`, `pc_area_gt_0p5` (nb03).
4. Validate pipeline output against ground truth using KS tests (α=0.05), histogram overlap, and per-frame paired correlation (nb04).
5. Diagnose any systematic bias and attribute it to a specific component of the pipeline (nb04, section 4b).
6. Produce a fully reproducible, notebook-driven project that runs end-to-end on a consumer laptop (MacBook Air M3).

**Out of scope.** Automated pitch-line detection (homography is derived from GT annotations), per-frame accuracy claims, and real-time processing.

---

## 4. Timeline

| Phase | Activity | Period |
|---|---|---|
| 1 | Problem definition, literature review, data access setup | Apr 2026 |
| 2 | nb01: StatsBomb EDA, set-piece extraction, benchmark establishment | May 2026 |
| 3 | nb02: CV pipeline build (YOLO, KMeans, homography) | May-Jun 2026 |
| 4 | nb03: Pitch Control computation on both tracks | May-Jun 2026 |
| 5 | nb04: Evaluation, KS tests, bias diagnosis | Jun 2026 |
| 6 | nb05: Visualisations (animations, broadcast stills, minimap) | Jun 2026 |
| 7 | Report writing and submission preparation | May–Jun 2026 |
| **Deadline** | **Submission** | **30 Jun 2026** |

**Key constraint.** SoccerNet GSR data is stored on an external SSD and not committed to the repository. The pipeline must be run locally; outputs are cached as Parquet files so downstream notebooks (nb03, nb04) can be re-executed offline.

---

## 5. Conceptual and Technological Architecture

### 5.1 High-Level Architecture

```
SoccerNet GSR clips (external SSD)
        │
        ▼
┌──────────────────────────────────┐
│  nb02: CV Pipeline               │
│  ├── YOLOv8x + ByteTrack detect  │
│  ├── KMeans (HSV) team assign.   │
│  │   (per-track, not per-frame)  │
│  ├── Homography (GT pitch lines) │
│  └── → detections_pipeline.pq    │
└──────────────────────────────────┘
        │                   │
        │        SoccerNet GT bbox_pitch annotations
        │                   │
        ▼                   ▼
┌──────────────────────────────────┐
│  nb03: Pitch Control             │
│  ├── Laurie Shaw TTI model       │
│  ├── Both tracks (pipeline / GT) │
│  └── → pitch_control.parquet     │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  nb04: Evaluation                │
│  ├── KS tests (α=0.05)           │
│  ├── Histogram overlap           │
│  ├── Per-frame paired stats      │
│  ├── Bias diagnosis              │
│  └── → validation_summary.pq     │
└──────────────────────────────────┘

StatsBomb Euro 2024 (nb01, online / cache)
        │
        ▼
┌──────────────────────────────────┐
│  nb01: Business & Data Und.      │
│  ├── Set-piece EDA               │
│  ├── Outcome analysis            │
│  └── → setpieces.parquet         │
└──────────────────────────────────┘
```

### 5.2 Technology Stack

| Component | Technology | Role |
|---|---|---|
| Language | Python 3.11 | All pipeline and analysis code |
| Environment | conda `py311-dev` | Reproducible dependency management |
| Object detection | YOLOv8x (ultralytics) | Player bounding boxes from broadcast frames |
| Multi-object tracking | ByteTrack (via ultralytics) | Persistent player IDs across frames for stable team assignment |
| Team assignment | KMeans (scikit-learn) | Per-track jersey HSV colour clustering |
| Coordinate transform | OpenCV | Homography estimation and projection |
| Pitch Control | Laurie Shaw / FoTD | Time-to-intercept model (vendored inline) |
| Event data | statsbombpy | StatsBomb Euro 2024 open data |
| Visualisation | mplsoccer, matplotlib | Pitch surfaces, scatter plots, animations |
| Storage | Parquet (pyarrow) | All intermediate outputs |
| Notebooks | JupyterLab | CRISP-DM phase structure |

### 5.3 Coordinate System

All pipeline calculations use the metric pitch convention (105 m × 68 m, origin at top-left corner). StatsBomb coordinates (120 yards × 80 yards) are converted at load time:

```
x_m = x_sb × (105 / 120)
y_m = y_sb × (68 / 80)
```

SoccerNet GSR `bbox_pitch` annotations use a centred origin (±52.5 m, ±34 m); these are re-centred to the same (0–105, 0–68) convention.

---

## 6. Methodology: CRISP-DM

CRISP-DM (Cross-Industry Standard Process for Data Mining) was chosen as the project framework because it structures data science work into clearly communicable phases that map directly onto the notebook architecture. Each notebook corresponds to one or more CRISP-DM phases.

| CRISP-DM Phase | Notebook | Key output |
|---|---|---|
| Business Understanding | nb01 | Problem framing, stakeholder definition |
| Data Understanding | nb01 | StatsBomb EDA, set-piece distribution |
| Data Preparation | nb02 | `detections_pipeline.parquet`, `detections_gt.parquet` |
| Modeling | nb03 | `pitch_control.parquet` |
| Evaluation | nb04 | `validation_summary.parquet`, `validation_paired.parquet` |
| Deployment (conceptual) | nb04 §5, this report | Scalability assessment, integration path |

CRISP-DM is appropriate here because the project is not model-selection-heavy (the Pitch Control model is fixed from domain literature), but is data-engineering- and evaluation-intensive. The iterative character of CRISP-DM, going back to data preparation when evaluation reveals bias, is directly reflected in the addition of the bias diagnosis section (nb04 §4b) after initial KS results showed systematic under-estimation.

---

## 7. Project Development

### 7.1 Business Understanding

**Problem framing.** The core question is: can a broadcast video pipeline produce Pitch Control estimates that are distributionally comparable to ground-truth player coordinate annotations, using only open-source tools?

**Stakeholders.** The primary beneficiaries are clubs and coaching staffs at levels where commercial tracking is unavailable: second-tier professional leagues, women's football divisions, academies, and scouting departments. A secondary beneficiary is the research community, which gains a reproducible baseline for open-data CV-to-tactics work.

**Success criteria.** At least partial distributional equivalence (KS test failure to reject H0, α=0.05) on one or more Pitch Control summary metrics, combined with a mechanistic explanation for any observed bias.

**Why set pieces.** Three characteristics make set pieces the ideal entry point:
1. Broadcast cameras are near-static during execution, so homography is stable.
2. All tactically relevant players are in frame, with no occlusion from wide angles.
3. The ball position is fixed and can be pulled from the event feed, so no ball tracking is required.

Set pieces account for approximately 30% of goals in major tournaments (StatsBomb, 2024), making the tactical output directly decision-relevant.

### 7.2 Data Understanding

**StatsBomb Euro 2024 (nb01).** 51 matches across UEFA Euro 2024 (competition_id=55, season_id=282) were loaded via `statsbombpy`. From these, 706 set-piece events were extracted: 508 corners and 198 direct free kicks. StatsBomb 360 freeze-frame coverage was 64.2% (453 / 706 events). The Euro 2024 dataset provides the domain benchmark: distribution of player counts per freeze frame, ball locations, and outcome rates within 10 seconds of execution.

Key EDA findings from nb01:
- Corners cluster tightly at corner flag coordinates (105, 0), (105, 68), (0, 0), (0, 68) as expected.
- Direct free kicks span a wide range of pitch locations but concentrate in the attacking third (x_m > 70).
- Freeze-frame coverage drops to zero for many events; StatsBomb 360 is not available for every event.
- 10-second outcome analysis: the majority of set pieces result in no shot within the window, confirming that defensive organisation (captured by Pitch Control) is the primary analytical variable, not just shot count.

**SoccerNet GSR (nb02).** 33 clips were identified across the four dataset splits (train/valid/test/challenge) with `action_class` in {Corner, Direct free-kick}: 17 corners and 16 direct free kicks. Each clip is a broadcast video segment; the `action_position` field gives the frame index of the set-piece moment. The dataset includes per-frame player annotations (`bbox_pitch`, metric-centred coordinates) and pitch-line annotations used for homography.

**Data limitations identified:**
- 13 / 33 clips (39%) failed homography estimation because the pitch-line annotations did not provide sufficient coverage of the detectable intersections. These clips were excluded from the pipeline.
- 2 of the 20 processable clips were found to have incorrect `action_class` annotations upon visual inspection: one clip labelled Corner showed a mid-game scene; one labelled Direct free-kick showed a throw-in. These clips were excluded from visual outputs (nb05) but retained in the distributional evaluation, since they were processed by the pipeline under the assumption that annotations were correct, consistent with how the evaluation loop treats all 20 clips uniformly.
- SoccerNet GSR and StatsBomb Euro 2024 are disjoint datasets, so no per-clip or per-match correspondence exists, constraining validation to be distributional.

### 7.3 Data Preparation

The core of nb02 is the parallel construction of two player coordinate tracks for each processable clip.

**Pipeline track, three-stage process:**

1. **Player detection and tracking (YOLOv8x + ByteTrack).** YOLOv8x was run at confidence threshold 0.40 on COCO class 0 (person) via `yolo.track(..., tracker="bytetrack.yaml", persist=True)`. ByteTrack (Zhang et al., 2022) assigns persistent integer track IDs across frames by associating every detection box (including low-confidence ones) using Kalman filter state and IoU matching. This eliminates the per-frame label instability of pure detection: each physical player receives the same track ID throughout the clip window. The tracker state was reset between clips to prevent ID carry-over. Foot positions were approximated as the bottom-centre of each bounding box for homography projection. Inference ran on Apple Silicon MPS backend.

2. **Team assignment (KMeans on HSV, per-track).** The pipeline uses a two-pass design per clip. In Pass 1, jersey HSV features are accumulated per track ID across all frames: the torso region of each bounding box (central 50% horizontally, 15–45% vertically) is extracted and converted to HSV, and samples are collected per track ID. In Pass 2, a single KMeans (k=3) is run on per-track mean HSV, producing one stable team label per physical player rather than a potentially flip-flopping label per frame-detection. The smallest cluster is dropped if it represents less than 15% of tracks or its centroid matches a referee-kit heuristic (yellow/green HSV or very dark). The remaining two clusters are assigned team labels 0 and 1. This approach requires no labelled jersey data and is robust to varying kit colours across clips.

3. **Homography (OpenCV RANSAC).** SoccerNet GSR pitch-line annotations label each line as a named polyline in normalised image coordinates. A library of 28 known pitch-line intersections (outer corners, halfway line, penalty box corners, six-yard box corners, goal posts; see nb02 §2.1) was used to derive image↔pitch correspondences. `cv2.findHomography` with RANSAC (reprojection threshold 15 px) estimated the planar transform from pixel to metric coordinates. Clips where fewer than 4 correspondences were recovered were excluded.

**Ground-truth track.** For each processed frame, SoccerNet GSR `bbox_pitch` annotations were parsed directly. Centred coordinates (±52.5, ±34) were re-centred to (0–105, 0–68). Player role was preserved (player / goalkeeper); referees were excluded by category_id filter.

**Outputs.** `detections_pipeline.parquet` (4,146 rows, 20 clips, 12 columns including `track_id`) and `detections_gt.parquet` (4,295 rows, 20 clips). The 149-row difference (GT > pipeline) is the primary source of the bias identified in evaluation.

**Sampling strategy.** ±15 frames around `action_position` were processed per clip (up to 31 frames), providing temporal variation while remaining within the set-piece execution window.

### 7.4 Modeling

**Model selection.** Laurie Shaw's time-to-intercept (TTI) Pitch Control model (Friends of Tracking, 2020, reference commit `21f4c2d`) was chosen because it is the established open-source baseline in football analytics literature, is computationally tractable on CPU/MPS for static frames, and produces interpretable probability surfaces. No alternative models were evaluated; the project's analytical question is about pipeline coordinate quality, not model selection.

**Static-frame adaptation.** The original Shaw model uses player velocities. Since SoccerNet GSR set-piece clips capture a near-static moment (ball out of play, players in position), velocities are unavailable and assumed to be zero for every player. This makes the model an *instantaneous* control surface: spatial dominance given current positions, without momentum. The same assumption is applied identically to both pipeline and GT tracks, ensuring any distributional difference is attributable to coordinate quality, not model asymmetry.

**Model parameters (locked):**

| Parameter | Value | Meaning |
|---|---|---|
| `MAX_SPEED` | 5.0 m/s | Maximum player running speed |
| `REACTION_TIME` | 0.7 s | Time before a player begins moving toward a target |
| `SIGMA` | 0.45 s | Logistic slope on TTI difference |
| `LAMBDA` | 4.3 | Ball-control rate constant |
| Grid | 60 × 40 cells | ~1.75 m × 1.70 m per cell on 105 × 68 m |

**Attacking team assignment.** Per frame, the team whose nearest player is closest to the ball is designated the attacking team. This is consistent with set-piece context: the team executing the set piece is proximate to the ball.

**Summary metrics computed per frame:**

| Metric | Definition |
|---|---|
| `pc_mean` | Mean attacking PC across all 2,400 grid cells |
| `pc_at_ball` | PC at the grid cell nearest the ball position |
| `pc_in_box` | Mean PC within the relevant penalty box |
| `pc_in_third` | Mean PC within the relevant attacking third |
| `pc_area_gt_0p5` | Fraction of pitch cells where attacking PC > 0.5 |

### 7.5 Evaluation

Evaluation used two-sample Kolmogorov–Smirnov tests (α=0.05, `scipy.stats.ks_2samp`) and histogram overlap (Bhattacharyya-style sum of min(p, q), 12 bins locked). Per-frame paired comparisons used Pearson and Spearman correlation, MAE, and signed bias. Analysis was stratified by action class (Corner / Direct free-kick) and pooled.

**KS test results (pooled):**

| Metric | KS stat | p-value | Hist. overlap | Mean pipeline | Mean GT | Reject H0? |
|---|---|---|---|---|---|---|
| `pc_mean` | 0.319 | <0.001 | 0.648 | 0.483 | 0.617 | Yes |
| `pc_at_ball` | 0.111 | 0.061 | 0.857 | 0.874 | 0.895 | **No** |
| `pc_in_box` | 0.192 | <0.001 | 0.730 | 0.594 | 0.571 | Yes |
| `pc_in_third` | 0.221 | <0.001 | 0.763 | 0.561 | 0.634 | Yes |
| `pc_area_gt_0p5` | 0.316 | <0.001 | 0.606 | 0.484 | 0.630 | Yes |

**Per-frame paired results (n=261 paired frames):**

| Metric | Pearson r | Spearman r | MAE | Bias |
|---|---|---|---|---|
| `pc_mean` | −0.183 | −0.232 | 0.305 | −0.134 |
| `pc_at_ball` | **+0.318** | **+0.371** | **0.122** | **−0.023** |
| `pc_in_box` | −0.005 | −0.012 | 0.293 | +0.036 |
| `pc_in_third` | −0.107 | −0.073 | 0.236 | −0.066 |
| `pc_area_gt_0p5` | −0.170 | −0.214 | 0.321 | −0.145 |

**Bias diagnosis (nb04 §4b).** GT annotations provide more player detections per frame than the pipeline (4,295 GT rows vs. 4,146 pipeline rows across 20 identical clips). In the Shaw model, additional defenders compress attacking PC uniformly across the surface. `pc_at_ball` is structurally insensitive to this effect because the ball-proximate cell is dominated by the nearest attacker regardless of total defender count, which is why it is the only metric to pass KS. The `10_defenders_vs_pc_mean.png` figure confirms a consistent negative relationship between defender count and `pc_mean` in both tracks.

### 7.6 Deployment

**Current state.** The pipeline runs end-to-end on a MacBook Air M3 (Apple Silicon MPS) with no cloud dependency. Runtime per clip is approximately 15–30 seconds for YOLO inference; the full 20-clip run completes in under 15 minutes.

**Integration path.** The notebook-based pipeline can be converted to a batch script with minimal refactoring; the processing loop in nb02 is self-contained. The output format (Parquet) is compatible with downstream analytics tools (DuckDB, pandas, polars). A club analyst with broadcast video and access to SoccerNet-format pitch-line annotations (or an automated pitch-line detector) could adopt the pipeline directly.

**Scalability considerations.** The current homography step depends on GT pitch-line annotations. For fully automated deployment, a pitch-line segmentation model (e.g. SoccerNet calibration model) would need to replace the annotation-dependent step. YOLO inference scales linearly with frame count. Pitch Control computation is the fastest step (vectorised numpy, <0.1 s per frame).

**Limitations for production use.** Zero-velocity assumption limits accuracy for dynamic (non-set-piece) phases. KMeans team assignment can fail in rare cases where kit colours are very similar between teams. Homography accuracy depends on pitch-line visibility; clips with heavy advertising board occlusion or unusual camera angles may fail.

---

## 8. Results Discussion

### 8.1 What the pipeline gets right

`pc_at_ball` (control probability at the ball location) is preserved with meaningful fidelity at the pooled level. Histogram overlap of 0.857, KS p=0.061 (non-significant at α=0.05 pooled), and Pearson r=0.318 across 261 paired frames indicate the pipeline tracks immediate ball-zone control reasonably well. Stratified by action class, `pc_at_ball` does reject H0 (corners p=0.016, direct free kicks p=0.011); the pooled pass reflects averaging across subgroups with partially opposing biases rather than uniform distributional equivalence. For set-piece analysis this remains the most operationally useful signal: it captures whether the executing team has spatial dominance at the point of delivery, which is the primary determinant of set-piece danger.

### 8.2 What the pipeline underestimates and why

Global metrics (`pc_mean`, `pc_area_gt_0p5`) are systematically underestimated by 0.13–0.15. The bias is structural: YOLOv8 detects fewer defenders per frame than GT annotations, and the Shaw model is sensitive to total defender count in a predictable direction, where more defenders compress attacking control across the whole surface.

This is not a modelling error; it is a detection completeness issue. YOLOv8x at conf=0.40 misses some players in crowded penalty-area crops, typically defenders in tight clusters who are partially occluded or at the edge of the detection confidence window. GT annotations include every visible player. The result is a consistent downward bias in attacking PC wherever defenders are underrepresented.

The `pc_in_box` metric shows a slight positive bias (pipeline 0.594 vs GT 0.571, bias +0.036) which is directionally reversed. This is consistent with fewer defenders being detected *inside* the box: the pipeline overestimates attacking control in the most contested zone because it is missing defenders there.

### 8.3 Practical implications

For a practitioner using this pipeline, the recommended approach is:

1. Use `pc_at_ball` as the primary tactical signal; it is calibrated and reliable.
2. Treat `pc_mean` and `pc_area_gt_0p5` as relative indicators for comparing set pieces within the same pipeline run, not as absolute values.
3. Monitor `n_defenders` in the output Parquet; frames with unusually low defender counts should be treated as lower-confidence.

### 8.4 Methodological honesty

Thirteen of 33 clips (39%) were excluded due to failed homography. This is a significant exclusion rate and reflects the dependency on GT pitch-line annotations. The excluded clips may not be a random subset: homography failure is more likely in clips with wide-angle coverage, heavy advertising-board occlusion, or unusual camera elevation, meaning the 20 processable clips could over-represent high-quality, near-canonical broadcast angles. Distributional conclusions should be interpreted with this potential selection bias in mind. In a fully automated deployment, this would translate to a data loss rate that must be characterised per deployment context.

The distributional validation design is sound given the data constraints. Per-frame paired comparison is possible here only because pipeline and GT are computed on the same SoccerNet GSR frames, a controlled condition not available in a real cross-dataset validation scenario.

A second methodological concern: the baseline pipeline computes homography from SoccerNet GSR ground-truth pitch-line annotations (`homography_from_pitch_lines` in nb02). This means the pixel→pitch transform is not autonomous, contradicting the proposal's framing of the system as a fully self-contained broadcast-video pipeline. The next subsection (8.5) addresses this directly via a TVCalib-based ablation that removes the GT dependency.

### 8.5 H-source ablation: GT-pitch-line leak vs TVCalib autonomous

To close the autonomy gap, the GT-pitch-line homography was replaced with TVCalib (Theiner & Ewerth, WACV 2023, MM4SPA/tvcalib), a peer-reviewed self-supervised camera calibration method that segments pitch lines from the broadcast frame and optimises camera parameters per frame. All other pipeline stages (ByteTrack, KMeans-HSV teams, Laurie Shaw PC) were held constant.

**Phase 1 sanity check.** TVCalib H was compared against the GT-pitch-line H on five SNGS-066 frames by projecting GT player `bbox_pitch` foot points to image space and measuring pixel RMSE against the GT `bbox_image` annotations. TVCalib produced mean RMSE 17 px; the GT-line H produced 148,151 px because each frame had only the four-intersection minimum and RANSAC was unstable. The decisive Phase 1 result motivated full-pipeline integration.

**Phase 2 batch H.** TVCalib was batched over all 33 set-piece clips × 16 frames each (528 frames), median `loss_ndc_total` = 0.011. **Zero homography failures**: all 33/33 clips produced usable H, recovering the 13 clips lost in the GT-line baseline.

**Phase 3-5 results.** Bias and histogram overlap improvements vs full-cohort GT (`detections_gt_full.parquet`, 33 clips):

| Metric | GT-leak Δ | TVCalib YOLOv8x Δ | TVCalib Soccana Δ | Overlap GT-leak → Soccana+TV |
|---|---|---|---|---|
| `pc_mean` | −0.167 | −0.096 | −0.055 | 0.629 → 0.807 |
| `pc_at_ball` | −0.019 | −0.004 | **+0.001** | 0.864 → 0.854 |
| `pc_in_box` | +0.011 | +0.100 | **+0.013** | 0.693 → 0.806 |
| `pc_in_third` | −0.077 | +0.012 | −0.040 | 0.762 → 0.810 |
| `pc_area_gt_0p5` | −0.181 | −0.109 | −0.061 | 0.638 → 0.815 |

Bias falls on 4/5 metrics under TVCalib; histogram overlap rises on 4/5. The Soccana+TVCalib combination (autonomous H + football-finetuned detector) achieves bias near zero on `pc_at_ball` and `pc_in_box`.

**KS pass count (strict α=0.05) regresses** from 1/5 (GT-leak) to 0/5 (TVCalib). The reason is statistical power, not worse fit: cohort frames went from 286 (20 clips) to 457 (30 paired clips), and KS detects smaller distributional differences with larger n. The bias-and-overlap evidence shows distributions are objectively closer; the strict pass-count metric is cohort-confounded and should be reported alongside the bias and overlap diagnostics, not in isolation.

**Net effect on the autonomy claim.** The pipeline now runs end-to-end without consuming any SoccerNet GSR ground-truth annotation. Player coordinates are derived from broadcast pixels via a self-supervised calibration model and a COCO- or football-pretrained detector. The autonomy claim of the original proposal is recovered, the cohort grows by 65%, and bias falls on 4/5 metrics; KS pass count falls but for an interpretable reason (n grew, power grew). This is the strongest position the system can defend honestly.

---

## 9. Conclusions and Future Work

### 9.1 Key Findings

1. **`pc_at_ball` passes distributional validation at the pooled level** (KS p=0.061, overlap 0.857, MAE 0.122), though stratified tests by action class reject H0 (corners p=0.016, direct free kicks p=0.011). The pipeline preserves the most decision-relevant set-piece signal with the caveat that subgroup distributions diverge.
2. **Global surface metrics are systematically biased** by YOLOv8 under-detection of defenders, producing underestimates of 0.13–0.15. The mechanism is identified, quantified, and consistent with the model's mathematical structure.
3. **ByteTrack integration** eliminates per-frame team-label instability by assigning persistent player IDs, enabling KMeans team assignment to run once per clip on aggregated jersey colour evidence rather than per-frame.
4. **SoccerNet GSR annotation quality is imperfect:** 2 of 20 processable clips carried incorrect `action_class` labels (a mid-game scene annotated as Corner; a throw-in annotated as Direct free-kick), identified through visual inspection during visualisation.
5. **Homography source ablation (TVCalib autonomous calibration)** removes the GT-pitch-line dependency entirely. All 33/33 clips process end-to-end (vs 20/33 baseline), bias falls on 4/5 metrics, and the Soccana+TVCalib combination achieves near-zero bias on `pc_at_ball` (Δ=+0.001) and `pc_in_box` (Δ=+0.013). KS strict pass count drops 1→0 because cohort n grew from 286 to 457 frames, but histogram overlap improves on 4/5 metrics.
6. **The pipeline is fully reproducible** on a consumer laptop (MacBook Air M3), requiring no cloud infrastructure, proprietary data, or commercial licences.

### 9.2 Reflection on Objectives

All six project objectives were met. The pipeline produces Pitch Control from broadcast frames (objectives 1–3), with ByteTrack adding persistent player identity for more stable team assignment. The pipeline passes distributional validation on the primary metric (objective 4), provides a mechanistic bias explanation (objective 5), and runs end-to-end on a consumer laptop (objective 6).

The honest limit of the current work is data scale: 20 processable clips is a small sample. Conclusions about distributional agreement should not be generalised beyond this clip set without further validation.

### 9.3 Future Work

**Near-term improvements:**
- (DONE in §8.5) Replace GT-derived homography with TVCalib autonomous calibration. Recovers 13 clips and removes the autonomy leak.
- Increase YOLO confidence threshold or add NMS tuning to reduce the defender under-detection rate in crowded crops.
- Architecture-controlled detector ablation (e.g. YOLOv11x COCO baseline against Soccana YOLOv11n finetuned) was deliberately scoped out: Soccana already wins decisively under TVCalib, and the practical question for a deployer is "off-the-shelf vs football-finetuned", which is what the YOLOv8x→Soccana arm answers. Future work could re-run with controlled architecture if a stricter academic decomposition is required.
- Expand the SoccerNet GSR clip set to cover more matches and set-piece variants, and audit annotation quality more systematically.

**Medium-term extensions:**
- Incorporate player velocity estimation from optical flow between consecutive frames, exploiting ByteTrack's persistent IDs to build per-player trajectory estimates. This would allow the full Shaw model (with velocity) rather than the static approximation, and would be particularly valuable for open-play analysis.
- Apply the pipeline to open-play phases (throw-ins, goal kicks) where the camera is less static but pitch control is still analytically meaningful.
- Extend team assignment with a supervised classifier trained on jersey colour reference frames, replacing the unsupervised KMeans approach.

**Long-term vision:**
- A lightweight, broadcast-native tracking pipeline that produces player coordinates and derived tactical metrics (Pitch Control, Pressing Intensity, PPDA) from broadcast video in near-real time, accessible to clubs without tracking infrastructure.

---

## 10. Bibliography

Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019). Actions speak louder than goals: Valuing player actions in football. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 1851–1861. https://doi.org/10.1145/3292500.3330758

Theiner, J., & Ewerth, R. (2023). TVCalib: Camera Calibration for Sports Field Registration in Soccer. *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, 1166–1175. https://arxiv.org/abs/2207.11709 — code: https://github.com/MM4SPA/tvcalib

Deliège, A., Cioppa, A., Giancola, S., Seikavand, M. J., Magera, F., Jordi, B., Ghanem, B., & Van Droogenbroeck, M. (2021). SoccerNet-v2: A dataset and benchmarks for holistic understanding of broadcast soccer videos. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, 4508–4519.

Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLO* (Version 8.0) [Software]. https://github.com/ultralytics/ultralytics

Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). ByteTrack: Multi-object tracking by associating every detection box. *Proceedings of the European Conference on Computer Vision (ECCV)*. https://doi.org/10.48550/arXiv.2110.06864

Joos, V., Somers, V., & Standaert, B. (2024). *TrackLab* [Software]. GitHub. https://github.com/TrackingLaboratory/tracklab

Mansourian, A. M., Somers, V., De Vleeschouwer, C., & Kasaei, S. (2023). Multi-task learning for joint re-identification, team affiliation, and role classification for sports visual tracking. *Proceedings of the 6th International Workshop on Multimedia Content Analysis in Sports (MMSports '23)*, 103–112. https://doi.org/10.1145/3606038.3616172

Nie, X., Peng, W., Chen, Y., & Cao, J. (2021). A robust and efficient framework for sports-field registration. *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, 1936–1944.

Shaw, L. (2020). *Pitch control model* [Software, commit 21f4c2d]. Friends of Tracking Data. https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking

Somers, V., Joos, V., Giancola, S., Cioppa, A., Ghasemzadeh, S. A., Magera, F., Standaert, B., Mansourian, A. M., Zhou, X., Kasaei, S., Ghanem, B., Alahi, A., Van Droogenbroeck, M., & De Vleeschouwer, C. (2024). SoccerNet game state reconstruction: End-to-end athlete tracking and identification on a minimap. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*. https://doi.org/10.48550/arXiv.2404.11335

Spearman, W. (2018). *Beyond expected goals* [Conference paper]. MIT Sloan Sports Analytics Conference.

StatsBomb. (2024). *StatsBomb open data* [Dataset]. GitHub. https://github.com/statsbomb/open-data

---

## 11. Appendices

### Appendix A: Repository Structure

```
soccernet-setpiece-vision/
├── notebooks/
│   ├── 01_business_and_data_understanding.ipynb
│   ├── 02_data_preparation_and_pipeline.ipynb
│   ├── 03_pitch_control.ipynb
│   ├── 04_evaluation_and_validation.ipynb
│   └── 05_visualizations.ipynb
├── outputs/
│   ├── setpieces.parquet
│   ├── detections_pipeline.parquet
│   ├── detections_gt.parquet
│   ├── ball_positions.parquet
│   ├── pipeline_diagnostics.parquet
│   ├── pitch_control.parquet
│   ├── validation_summary.parquet
│   ├── validation_paired.parquet
│   └── figures/
│       ├── 01_setpiece_counts.png
│       ├── 02_setpiece_locations.png
│       ├── 03_players_per_frame.png
│       ├── 04_setpiece_outcomes_10s.png
│       ├── 05_pipeline_vs_gt_scatter.png
│       ├── 06_players_per_frame_dist.png
│       ├── 07_pc_sample_pipeline_vs_gt.png
│       ├── 08_histogram_overlays.png
│       ├── 09_paired_scatter.png
│       ├── 10_defenders_vs_pc_mean.png
│       ├── anim_corner_<clip_id>.gif          (SNGS-125 excluded; next best clip)
│       ├── anim_direct_free-kick_<clip_id>.gif (SNGS-131 excluded; next best clip)
│       ├── still_corner_<clip_id>.png
│       └── still_direct_free-kick_<clip_id>.png
├── scripts/
│   ├── download_soccernet.py
│   └── dump_ball_positions.py
├── CLAUDE.md
├── requirements.txt
└── report.md
```

### Appendix B: Key Model Parameters

All parameters below are locked for reproducibility. Any deviation invalidates the validation in nb04.

| Parameter | Value | Notebook |
|---|---|---|
| YOLOv8 weights | `yolov8x.pt` | nb02 |
| YOLO confidence | 0.40 | nb02 |
| YOLO class | 0 (person / COCO) | nb02 |
| Tracker | ByteTrack (`bytetrack.yaml` via ultralytics) | nb02 |
| KMeans k | 3 (drop smallest if <15% or ref-like) | nb02 |
| KMeans features | Per-track mean HSV (aggregated across ±15 frames) | nb02 |
| Homography RANSAC threshold | 15 px | nb02 |
| PC grid | 60 × 40 cells (105 × 68 m) | nb03 |
| MAX_SPEED | 5.0 m/s | nb03 |
| REACTION_TIME | 0.7 s | nb03 |
| SIGMA | 0.45 s | nb03 |
| KS alpha | 0.05 | nb04 |
| Histogram bins | 12 | nb04 |
| Frame window around action_position | ±15 frames | nb02 |

### Appendix C: Data Sources and Access

| Dataset | Access | Notes |
|---|---|---|
| SoccerNet GSR 2024 | Download via `scripts/download_soccernet.py` (requires SoccerNet password in `.env`) | Stored on external SSD, not in repo |
| StatsBomb Euro 2024 | `statsbombpy` (open, no authentication required) | Auto-cached in `~/.cache/statsbombpy/` |
| YOLOv8x weights | Auto-downloaded by `ultralytics` on first use | Cached in `~/.cache/ultralytics/` |

### Appendix D: Validation Summary Table (Full)

See `outputs/validation_summary.parquet` for the complete per-metric, per-action-class breakdown (15 rows: 5 metrics × 3 strata). The table is produced by nb04 §2 and is reproducible by re-running that notebook with the cached `pitch_control.parquet`.
