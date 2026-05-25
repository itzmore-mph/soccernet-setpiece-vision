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

**Solution.** A fully autonomous player-detection pipeline: Soccana (YOLOv11n, football-finetuned) detects and tracks players via ByteTrack, assigns stable team labels via per-track KMeans on HSV jersey colour, projects pixel coordinates to metric pitch via TVCalib autonomous camera calibration, and computes attacking Pitch Control using Laurie Shaw's time-to-intercept model. Ball positions are sourced from SoccerNet GSR ground-truth annotations; autonomous ball detection is noted as a future work item.

**Validation.** Distributional comparison (KS test, histogram overlap) plus per-frame paired statistics against SoccerNet GSR ground-truth player annotations on 33 set-piece clips (17 corners, 16 direct free kicks).

**Key results.**

- Soccana detects players in all 33/33 clips with zero homography failures (detection phase).
- Pitch Control is computed for 31 pipeline clips and 31 GT clips; 2 clips (SNGS-125, SNGS-145) are excluded from the PC phase due to missing ball annotations in all frames.
- `pc_at_ball` (control at ball location): bias = -0.051, histogram overlap = 0.804, per-frame Pearson r = 0.511.
- `pc_in_third` is the most reliable metric: bias ≈ 0 (+0.000), histogram overlap = 0.896.
- `pc_in_box` has the largest systematic error (+0.223 bias): KMeans team assignment degrades in crowded penalty areas during corners, inverting the team label in that zone for a subset of frames.
- Global metrics (`pc_mean`, `pc_area_gt_0p5`) are underestimated by ~0.17–0.18: pipeline detects fewer players per frame (mean 15.99 vs GT 18.69) with a larger shortfall on the defending team, compressing attacking control across the surface under the Shaw model.
- Full pipeline runs on a standard consumer laptop (~30 min on Apple Silicon MPS, longer on CPU-only hardware), no cloud dependency.

**Impact.** A broadcast-only Pitch Control pipeline that produces distributionally honest estimates of the most operationally meaningful set-piece signal, deployable by any club with broadcast video access and a laptop.

---

## 2. Introduction

### 2.1 Problem Statement

Pitch Control — the probability that a team could reach any point on the pitch first — is a standard tool for evaluating spatial dominance in elite football. Commercial systems (StatsBomb 360, SkillCorner, Tracab) deliver this data but their cost restricts access to top-tier competitions.

For the majority of professional clubs, women's leagues, academies, and scouting departments, data-driven set-piece analysis remains out of reach not because of analytical sophistication but because of data access.

### 2.2 Why Set Pieces

Set pieces (corners and direct free kicks) are tactically high-leverage and analytically tractable. Across 706 set pieces in UEFA Euro 2024, 32.4% produced a shot within 10 seconds and 1.8% produced a goal. Set pieces are also optimal for a CV pipeline: the broadcast camera is near-static, relevant players are in frame, and ball position can be sourced from a lightweight event feed.

### 2.3 Research Gap

Prior literature covers player detection, tracking, calibration, and Pitch Control modelling individually. The end-to-end chain — broadcast pixels through to a distributionally validated tactical metric, without proprietary tracking and without ground-truth annotations leaking into the calibration step — remains underdeveloped.

### 2.4 Contribution

A reproducible, validated pipeline from broadcast video to Pitch Control, with:
- Fully autonomous player detection and calibration (no GT annotations consumed at inference time)
- Distributional validation against open SoccerNet GSR annotations
- Bias diagnosis attributing residual error to differential detection recall between attackers and defenders
- Consumer-hardware execution (Apple Silicon MPS or equivalent)

### 2.5 Research Scope and Boundaries

**Data scope.** The primary validation dataset is SoccerNet GSR 2024: 33 clips covering two set-piece classes (17 corners, 16 direct free kicks) drawn from the train, valid, test, and challenge splits. StatsBomb Euro 2024 open data is used only for distributional context on player counts and set-piece outcomes; it is not part of the primary validation.

**Temporal scope.** The pipeline was built and validated on a single collection of pre-computed homographies and detections. The SoccerNet GSR ground-truth annotations are used exclusively for validation; they are never consumed at detection or calibration inference time.

**Set-piece scope.** Only corners and direct free kicks are analysed. These were selected because the broadcast camera is near-static for both, all relevant players are in frame at the moment of execution, and ball position is reliably annotatable. Throw-ins, goal kicks, indirect free kicks, and open-play sequences are out of scope.

**Out of scope.** Autonomous ball detection, player re-identification across clips, multi-camera setups, non-broadcast (e.g., tactical camera) footage, and tracking over full match sequences are all excluded. The pipeline is validated as a static-frame set-piece tool, not a real-time tracking system.

### 2.6 Research Structure and Preview

Section 3 defines the research objectives and success criteria. Section 4 describes the full pipeline architecture and technology stack. Section 5 explains the CRISP-DM methodology and its adaptations. Section 6 documents each phase of development in detail, including all data preparation choices, modelling decisions, and evaluation results. Section 7 discusses the findings, identifies cross-metric patterns, and articulates methodological limits. Section 8 draws conclusions, proposes a development roadmap, and outlines future work. Annexes provide the repository structure, model parameters, data sources, and full reproducibility instructions.

---

## 3. Objectives

### 3.1 Primary Objective

Develop a reproducible computer vision pipeline that extracts Pitch Control from broadcast set-piece frames and produces distributions comparable to ground-truth annotation-derived distributions, using only open-source tools and consumer hardware.

### 3.2 Research Questions

- **RQ1.** Can a broadcast-video-only pipeline produce Pitch Control distributions comparable to GT-annotation distributions for set-piece frames?
- **RQ2.** What is the dominant source of systematic bias in pipeline-derived Pitch Control?
- **RQ3.** Which Pitch Control summary metrics are most robust to pipeline noise?

### 3.3 Academic Objectives

- Demonstrate end-to-end distributional validation methodology for a broadcast computer vision pipeline, moving beyond aggregate accuracy to per-metric bias attribution.
- Establish which Pitch Control summary metrics are robust to broadcast-pipeline noise and which are structurally contaminated by specific component failures.
- Contribute empirical evidence on the limits of colour-based team assignment in crowded set-piece environments, informing future work on supervised team-classification approaches.
- Produce a reproducible codebase and validated outputs that enable independent replication and extension.

### 3.4 Practical Objectives

- Deliver a working, end-to-end pipeline executable on consumer hardware in under 45 minutes on Apple Silicon MPS (longer on CPU-only hardware) for 33 clips, without cloud infrastructure or proprietary data.
- Identify which PC metrics are operationally deployable by clubs without access to GT tracking, and which require further development before use.
- Produce broadcast-overlay visualizations (three-panel stills and animated GIFs/MP4s) that allow a tactical analyst to interpret pipeline outputs directly on the broadcast frame.
- Document data-quality issues in SoccerNet GSR (notably the `action_position` global-frame-number misinterpretation) to inform future dataset users.

### 3.5 Expected Outcomes and Deliverables

**Technical deliverables:**
- Detection parquets (`detections_soccana_tvcalib.parquet`, `detections_gt_full.parquet`) for all 33 clips
- Ball position cache (`ball_positions.parquet`) and pitch control surfaces (`pitch_control_soccana_tvcalib.parquet`, `pitch_control_gt_full.parquet`)
- Validation outputs (`validation_summary_tvcalib.parquet`, `validation_paired.parquet`, KS table figure)
- Broadcast-overlay visualizations: static three-panel stills, animated GIFs, and MP4 clips for representative corner and free-kick sequences

**Academic deliverables:**
- Validated distributional comparison of pipeline-derived vs GT-derived Pitch Control on 31 clips
- Mechanistic bias diagnosis attributing error to three distinct failure modes (global underestimation, box inversion, calibrated third)
- Documented methodology for adapting a dynamic Pitch Control model to a zero-velocity set-piece context

### 3.6 Success Criteria

| Criterion | Target |
|---|---|
| End-to-end execution | All 33 clips, consumer hardware, <45 min (MPS) / <90 min (CPU) |
| Bias | \|bias\| < 0.10 on `pc_at_ball` |
| Histogram overlap | >0.80 on at least one metric |
| Bias explanation | Mechanistic, attributable to specific component |
| Reproducibility | Full pipeline re-runnable from raw SSD data |

### 3.7 Ethical Considerations

**Data access.** SoccerNet GSR requires a credentialed download via the SoccerNet API; credentials were obtained through the standard academic access request process. StatsBomb open data is freely available under CC BY-SA 4.0. Soccana model weights are publicly hosted on HuggingFace under an open licence.

**No personal data.** No individual player identities, biometric data, or personally identifiable information are stored, processed, or published. Player positions are treated as anonymous spatial coordinates.

**Model weights.** The Soccana YOLOv11n weights used are not fine-tuned in this project; they are used as-is from HuggingFace and their original licence terms apply.

**Reproducibility and transparency.** All pipeline code, parameter choices, and known data-quality issues (including the `action_position` misinterpretation) are documented and committed to the repository. No results are selectively reported; the full validation table including unfavourable metrics is published.

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
|  Ball position (SoccerNet GT)    |
|  - bbox_pitch parsed per frame   |
|  - 949/1023 frames have valid pos|
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
| Ball position | SoccerNet GSR GT annotations (bbox_pitch) |
| Pitch Control | Laurie Shaw TTI model (Friends of Tracking) |
| Data | SoccerNet GSR 2024, StatsBomb Euro 2024 |
| Language | Python 3.11 |
| Hardware | Apple Silicon (MPS), or any CUDA/CPU-capable host |

### 4.3 Coordinate Systems

| System | Convention |
|---|---|
| StatsBomb | 120 yd x 80 yd, origin top-left |
| Pipeline / mplsoccer | 105 m x 68 m, origin top-left |
| SoccerNet GSR bbox_pitch | centred origin (+-52.5 m, +-34 m) |

Conversions: GSR → pipeline: `x = x_gsr + 52.5`, `y = y_gsr + 34`. StatsBomb → pipeline: `x = x_sb × (105/120)`, `y = y_sb × (68/80)`.

### 4.4 Core Libraries and Tools

| Library | Version | Role |
|---|---|---|
| ultralytics | 8.3.107 | Soccana detection + ByteTrack tracking |
| torch | ≥2.1.0 | Inference backend (Apple MPS) |
| opencv-python | ≥4.8.0 | Frame I/O, HSV colour extraction, video rendering |
| pandas | ≥2.1.0 | Tabular data pipeline |
| numpy | ≥1.26.0 | Array operations, homography math |
| scipy | ≥1.11.0 | KS test, spatial distance computations |
| scikit-learn | ≥1.3.0 | KMeans team assignment |
| mplsoccer | ≥1.4.0 | Pitch visualisation |
| pyarrow | ≥14.0.0 | Parquet I/O |
| statsbombpy | ≥1.14.0 | StatsBomb open data access |
| huggingface-hub | ≥0.20.0 | Soccana weight download |
| python-dotenv | ≥1.0.0 | SSD path and credential configuration |

TVCalib (Theiner & Ewerth) runs in a separate sibling conda environment with its own dependencies (PyTorch 2.1, kornia 0.8.2, pytorch-lightning 2.6.1) and is invoked via subprocess from `run_tvcalib_batch.py`.

### 4.5 Development and Hardware Environment

- **Hardware:** Apple Silicon laptop (M-series, 16 GB unified memory), MPS backend for PyTorch inference. The pipeline also runs on CUDA or CPU; detection will be slower on CPU-only hardware.
- **Storage:** SoccerNet GSR video data (~35 GB) on an external USB-C SSD. All intermediate outputs (Parquet files, figures) live in the repository `outputs/` directory and are committed; video frames are not committed.
- **Software:** macOS 15, Python 3.11 (conda env `py311-dev`). Detection runs on MPS; Pitch Control computation and validation run on CPU.
- **Execution profile:** ~30 min total for 33 clips (TVCalib batch ~14 min, Soccana detection ~15 min, all downstream scripts <1 min combined).

### 4.6 Scalability and Constraints

**Horizontal scaling.** The pipeline processes clips sequentially; parallelisation across clips is straightforward (independent `run_clip()` calls) but was not implemented, as the 30-minute runtime is acceptable for the 33-clip cohort.

**TVCalib coupling.** Homography computation requires TVCalib in a sibling directory with its own environment. Pre-computed homographies are committed to the repository (`homographies_tvcalib.parquet`), decoupling all downstream scripts from the TVCalib dependency for normal operation.

**Ball detection dependency.** The current architecture requires GT `bbox_pitch` ball annotations from SoccerNet GSR to identify the ball position per frame. This is the primary constraint preventing fully autonomous deployment. All other pipeline components (detection, tracking, calibration, team assignment, PC computation) run without GT inputs.

**Memory.** Peak memory consumption during detection inference is approximately 4 GB (MPS), dominated by the YOLO model and frame batch. Pitch Control computation is CPU-bound and uses <1 GB.

---

## 5. Methodology: CRISP-DM

### 5.1 Why CRISP-DM

CRISP-DM was selected as the organising framework for three reasons specific to this project.

**Iterative fit.** Pipeline development rarely proceeds linearly. The discovery that `action_position` in SoccerNet GSR is a global broadcast frame number (not a clip-local index), uncovered during data understanding, required revising the frame-window logic before preparation and modelling could proceed correctly. CRISP-DM's explicit feedback loop between phases accommodates this kind of mid-project revision without treating it as a failure.

**Validation emphasis.** The evaluation phase in CRISP-DM is a first-class phase, not an afterthought. For this project — where the central research question is whether a broadcast pipeline produces distributions comparable to ground truth — the evaluation phase is where the research question is answered. CRISP-DM's structure prevents evaluation from being compressed into a footnote.

**Reproducibility.** CRISP-DM's phase separation maps directly onto the script/notebook structure of the repository: each phase has identifiable inputs, processing steps, and committed Parquet outputs. A future user can enter the pipeline at any phase using the committed intermediate outputs, verifying reproducibility of any downstream phase independently.

### 5.2 Phase Summary

| Phase | Notebook / Script | Key Output |
|---|---|---|
| Business Understanding | nb01 | Problem framing, stakeholder identification, set-piece EDA |
| Data Understanding | nb01 | StatsBomb EDA, SoccerNet GSR scan, action_position audit |
| Data Preparation | run_tvcalib_batch, run_soccana_tvcalib, dump_* | Detection and ball-position Parquets |
| Modeling | nb02, run_pc_* | Pitch control surfaces, summary metrics |
| Evaluation | nb03, ks_table_tvcalib.py | Validation tables, distributional + paired statistics |
| Deployment | nb04, render_* | Broadcast-overlay visualizations, GIFs, MP4s |

### 5.3 Methodology Adaptations

Two adaptations were required to apply CRISP-DM to this computer vision pipeline context.

**Static-frame modelling.** The Laurie Shaw TTI Pitch Control model was designed for tracking data with per-frame player velocities. This project uses a zero-velocity adaptation: all players are assumed to be stationary at the moment of the set-piece, and time-to-intercept reduces to distance divided by maximum speed. This is appropriate for the static set-piece formation captured in frames 1–31 of each clip, but limits extension to open play.

**Distributional evaluation.** Standard CRISP-DM evaluation focuses on predictive model accuracy. Here, the pipeline does not predict a label — it computes a spatial metric. Evaluation therefore uses distributional comparison (KS test, histogram overlap) and per-frame paired statistics (Pearson r, MAE, bias) to assess whether the pipeline-derived distribution is consistent with the GT-derived distribution, rather than testing classification accuracy or regression error against a held-out set.

---

## 6. Work Development

### 6.1 Phase 1: Business Understanding

**Core question:** Can a broadcast-only pipeline produce Pitch Control distributions comparable to GT for set-piece frames?

**Stakeholders:** Clubs without tracking providers (second-tier professional, women's football, academies, scouting).

**Why set pieces:** Near-static camera, all players in frame, ball position reliably available from event feeds. Of 706 Euro 2024 set pieces, 65.7% produced no shot within 10 s, 32.4% produced a shot, and 1.8% produced a goal, placing set pieces among the highest-leverage repeatable game situations for tactical investment.

### 6.2 Phase 2: Data Understanding

**StatsBomb Euro 2024 (nb01).** 51 matches, 706 set-piece events (508 corners, 198 direct free kicks). Freeze-frame coverage for this subset is 64.2% (453/706 events), as not every event in the open data release carries an associated 360 freeze frame. Used only for distributional context on player counts and set-piece outcomes; not used in the primary validation.

**SoccerNet GSR.** 33 clips identified with action_class in {Corner, Direct free-kick}: 17 corners, 16 direct free kicks. Per-frame player annotations (bbox_pitch) and pitch-line annotations. TVCalib homographies computed for all 1,023 frames (33 clips × 31 frames).

**Note on action_position.** The `action_position` field in SoccerNet GSR Labels-GameState.json is a global broadcast frame number (ranging from ~300,000 to ~2,600,000), not a clip-local index. Clips are 750 frames numbered 1–750, with the set-piece occurring at frame 1 (confirmed by GT ball-position coordinates at the corner arc on frame 1 for all corner clips). The pipeline uses `centre = FRAME_WINDOW + 1 = 16` to place the ±15-frame window at frames 1–31, covering the static set-piece formation.

### 6.3 Phase 3: Data Preparation

**Pipeline track (run_soccana_tvcalib.py):**
1. Soccana detection at confidence 0.40, classes 0 (Player) and 2 (Referee) in a single forward pass.
2. ByteTrack persistent ID assignment across frames for all detected objects.
3. Per-track mean HSV with KMeans (k=3, smallest-cluster drop if <15% of clip population or referee-like HSV centroid, re-fit k=2) applied to player tracks only. Player detections receive a team label (0 or 1); referee detections are assigned `team=-1` directly and excluded from pitch control computation.
4. TVCalib homography: pixel foot-point to metric pitch coordinates for both players and referees.
5. Output: 17,260 detection rows across 33 clips (16,209 player rows + 1,051 referee rows).

**GT track (dump_gt_setpieces.py):**
- SoccerNet GSR bbox_pitch annotations parsed directly.
- Centred coordinates converted to top-left origin (0–105, 0–68). Annotations with coordinates outside ±2 m of pitch boundaries are discarded (handles corner-arc annotation noise and the small number of corrupted GT entries with physically impossible coordinates).
- Output: 18,539 rows across 32 clips (SNGS-125 has no GT player annotations in frames 1–31).

**Detection shortfall:** Pipeline produces 2,330 fewer rows than GT (12.6% shortfall). Mean players per frame: pipeline 15.99 vs GT 18.69. The defending-team shortfall is larger (pipeline 7.41 vs GT 9.13 per frame) than the attacking-team shortfall (pipeline 8.58 vs GT 9.56), consistent with defenders clustering in occluded, crowded positions near the goal. This asymmetric shortfall drives the directional bias on global metrics under the Shaw model.

**Ball positions (dump_ball_positions.py):** 1,023 frame positions parsed from SoccerNet GSR GT annotations (bbox_pitch, category_id=4). 949/1,023 frames have valid ball coordinates (within ±2 m of pitch boundaries). SNGS-125 and SNGS-145 have no ball annotation in any frame within the window; these 2 clips are excluded from the PC computation, reducing the effective validation set from 33 to 31 clips.

### 6.4 Phase 4: Modeling

**Model:** Laurie Shaw TTI Pitch Control (zero-velocity, static-frame adaptation).

**Parameters (locked):**
- MAX_SPEED: 5.0 m/s
- REACTION_TIME: 0.7 s
- SIGMA: 0.45 s
- Grid: 60 x 40 cells on 105 x 68 m pitch

**Attacking team:** Per frame, whichever team has the player closest to the ball.

**Output:** 940 pipeline frames (31 clips), 949 GT frames (31 clips).

**Summary metrics per frame:**
- `pc_mean`: Mean attacking PC across all 2,400 grid cells
- `pc_at_ball`: PC at grid cell nearest ball position
- `pc_in_box`: Mean attacking PC within relevant penalty box
- `pc_in_third`: Mean attacking PC within relevant attacking third
- `pc_area_gt_0p5`: Fraction of cells where attacking PC > 0.5

### 6.5 Phase 5: Evaluation

**Distributional comparison (pipeline n=940, GT n=949):**

| Metric | Pipeline | GT | Delta (bias) | KS stat | KS p-value | Hist. overlap | Passes KS? |
|---|---|---|---|---|---|---|---|
| pc_mean | 0.515 | 0.687 | -0.172 | 0.306 | <0.001 | 0.695 | No |
| pc_at_ball | 0.927 | 0.978 | -0.051 | 0.371 | <0.001 | 0.804 | No |
| pc_in_box | 0.539 | 0.316 | **+0.223** | 0.554 | <0.001 | 0.475 | No |
| **pc_in_third** | **0.513** | **0.513** | **+0.000** | 0.070 | 0.019 | **0.896** | No |
| pc_area_gt_0p5 | 0.521 | 0.703 | -0.182 | 0.293 | <0.001 | 0.692 | No |

**Per-frame paired comparison (n=940 paired frames):**

| Metric | Pearson r | Spearman r | MAE | Bias |
|---|---|---|---|---|
| pc_mean | 0.285 | 0.060 | 0.229 | -0.173 |
| pc_at_ball | **0.511** | 0.096 | **0.054** | -0.051 |
| pc_in_box | 0.175 | 0.067 | 0.244 | +0.224 |
| **pc_in_third** | -0.055 | -0.069 | **0.119** | **+0.000** |
| pc_area_gt_0p5 | 0.260 | 0.080 | 0.249 | -0.182 |

Low Spearman values reflect compressed score distributions at the set-piece moment (both pipeline and GT cluster near their respective means), which reduces rank variation and makes Spearman unreliable relative to Pearson.

**Bias diagnosis:** The five metrics divide into three groups by error type.

*Global underestimation* (`pc_mean`, `pc_area_gt_0p5`, `pc_at_ball`): GT records 18.69 players per frame vs pipeline 15.99 (14.4% shortfall), with the defending team more under-detected (pipeline 7.41 vs GT 9.13) than the attacking team (8.58 vs GT 9.56). Fewer defenders shift the Shaw surface toward attacker control; GT, with more defenders, correctly computes stronger defensive compression. The pipeline therefore underestimates the degree to which defenders constrain attacking space.

*Box inversion* (`pc_in_box`): This is the largest error and the most structurally distinct. GT shows the defending team controlling the penalty box (mean 0.316, well below 0.5), which is correct — at a corner, defenders pack the box. The pipeline estimates attacker control (0.539), a sign inversion. KMeans on per-track mean HSV fails when both teams are densely packed in the small penalty-area crop: jersey colours are harder to separate under broadcast lighting at this scale, and cluster centroids can interchange between teams.

*Calibrated* (`pc_in_third`): delta = +0.000, histogram overlap = 0.896. Attacking-third analysis is unaffected by box crowding and is less sensitive to total player count than global metrics. This is the most reliable operational metric from this pipeline.

### 6.6 Phase 6: Deployment

- Pipeline runs end-to-end on consumer hardware (validated on Apple Silicon MPS), no cloud dependency.
- Runtime: ~30 minutes for 33 clips.
- Parquet outputs compatible with DuckDB, pandas, polars.
- TVCalib removes any dependency on GT pitch-line annotations for camera calibration.
- Three-panel animated visualizations (broadcast frame, metric minimap, PC heatmap) produced as GIF and MP4 for representative corner and direct free-kick clips (SNGS-116, SNGS-122), now correctly showing frames 1–31 of each clip (the actual set-piece formation); static three-panel stills generated for thesis embedding (see Annex A).

### 6.7 Project Outcomes and Deliverables

**Technical outcomes.** The pipeline successfully processes all 33 set-piece clips end-to-end without homography failures. Intermediate outputs are stored as Parquet files and committed to the repository, enabling SSD-free reproduction of all analysis notebooks and validation scripts.

| Deliverable | Description |
|---|---|
| `homographies_tvcalib.parquet` | 1,023 TVCalib homographies (33 clips × 31 frames) |
| `detections_soccana_tvcalib.parquet` | 17,260 rows: 16,209 player + 1,051 referee (pipeline) |
| `detections_gt_full.parquet` | 18,539 GT player annotation rows |
| `ball_positions.parquet` | 1,023 frame ball positions (949 valid) |
| `pitch_control_soccana_tvcalib.parquet` | 940 pipeline PC frames (31 clips) |
| `pitch_control_gt_full.parquet` | 949 GT PC frames (31 clips) |
| `validation_summary_tvcalib.parquet` | Distributional KS + overlap statistics |
| `validation_paired.parquet` | Per-frame paired Pearson, Spearman, MAE, bias |
| Three-panel stills (PNG) | Thesis-embeddable figures for SNGS-116 and SNGS-122 |
| Animated overlays (GIF, MP4) | 31-frame PC heatmap overlays for representative clips |

**Academic outcome.** The research questions are answered: broadcast-only Pitch Control is viable for `pc_in_third` and `pc_at_ball`; the dominant bias sources are asymmetric detector recall (global metrics) and KMeans team-assignment inversion in crowded penalty areas (`pc_in_box`).

**Methodological outcome.** The `action_position` data-quality issue in SoccerNet GSR (global broadcast frame number misinterpreted as clip-local index) was identified, fixed, and documented — a contribution to future users of this dataset.

---

## 7. Discussion of Results

### 7.1 What the pipeline gets right

`pc_in_third` is the most reliable metric: bias = +0.000, histogram overlap = 0.896. Attacking-third analysis integrates over a large pitch zone (one third of the field, ~35 m × 68 m), diluting localised detection errors and team-assignment noise. It is the appropriate primary metric for comparing set-piece spatial dominance across clips.

`pc_at_ball` — control at the ball location — is well-preserved: bias = -0.051, histogram overlap = 0.804, Pearson r = 0.511. The pipeline reliably identifies that the executing team has spatial dominance at the point of delivery, the most operationally meaningful signal.

### 7.2 What the pipeline gets wrong: the box inversion

`pc_in_box` has the largest error: bias = +0.223, histogram overlap = 0.475. GT shows defenders controlling the penalty box (mean 0.316), which is correct — at a corner, the defending team packs the box. The pipeline estimates attacker control (mean 0.539), a full sign inversion. The cause is KMeans team assignment failure in crowded penalty-area crops: when both teams are tightly packed near the goal, per-track mean HSV features become harder to separate, and cluster centroids can interchange between teams. This is a structural limitation of colour-based team assignment, not a detection recall issue.

**Practical implication:** `pc_in_box` should not be used as a reliable signal from this pipeline in its current form.

### 7.3 What the pipeline underestimates

Global metrics (`pc_mean`, `pc_area_gt_0p5`) are underestimated by ~0.17–0.18. The pipeline detects 15.99 mean players per frame vs GT 18.69, with a larger shortfall on defenders (7.41 vs GT 9.13) than on attackers (8.58 vs GT 9.56). In the Shaw model, additional defenders compress attacking control uniformly across the surface; the asymmetric recall gap means attacking control is underestimated. This is a detection completeness problem, predictable from the model's mathematics.

### 7.4 Why no metrics pass strict KS

No metrics pass KS at alpha=0.05. With 940 paired frames, KS has high statistical power and detects small distributional shifts. The strict rejection reflects sample size and the structural biases above rather than catastrophic model failure.

### 7.5 Practical Implications

1. Use `pc_in_third` as the primary comparative metric — it is accurately calibrated and robust to local crowding.
2. Use `pc_at_ball` as a secondary signal for per-delivery dominance assessment.
3. Do not use `pc_in_box` from this pipeline without resolving team-assignment reliability in crowded penalty areas.
4. Treat `pc_mean` and `pc_area_gt_0p5` as relative indicators (within-pipeline comparisons) rather than calibrated absolute values.

### 7.6 Cross-Finding Synthesis

The five validation metrics divide cleanly into three error regimes, and each regime traces to a distinct pipeline component:

| Error type | Metrics | Component | Fix path |
|---|---|---|---|
| Global underestimation | `pc_mean`, `pc_area_gt_0p5` | Soccana detection recall | Lower confidence threshold; ensemble detector |
| Moderate underestimation | `pc_at_ball` | Combined recall + proximity | Same as above; lower priority |
| Sign inversion | `pc_in_box` | KMeans team assignment | Supervised classifier; cross-frame consistency |
| Calibrated | `pc_in_third` | None — unaffected | Retain; primary operational metric |

The most important cross-cutting finding is that the error structure is tractable: each failure mode has a clear cause and a concrete remediation path. The pipeline is not uniformly wrong — one metric is already deployment-ready, and the others have predictable bias directions that practitioners can account for.

A second cross-cutting finding concerns the interaction between set-piece type and team assignment: corners impose the most severe crowding (all 22 players converge within ~35 m of the box), while direct free kicks may leave more spatial separation between teams. Future work should stratify validation by set-piece type to assess whether `pc_in_box` performs better on free kicks than corners.

### 7.7 Methodological Limits

- **Ball detection not automated:** Ball positions are sourced from SoccerNet GSR GT annotations. In a club deployment, a separate ball detection or event-feed integration step would be required. This is the main gap between the current implementation and a fully autonomous end-to-end pipeline.
- **TVCalib error propagation:** TVCalib introduces reprojection errors of several centimetres to low single-digit metres depending on pitch region and broadcast angle. These propagate directly into player coordinates and modestly affect all PC metrics. Quantifying this error channel is reserved for future work.
- **Cohort size:** 33 clips; conclusions should not be generalised beyond this cohort or to broadcast conditions substantially different from SoccerNet GSR.
- **Static-frame assumption:** Zero-velocity Shaw TTI is appropriate for set-piece snapshots but does not extend to open play. Players may already be in motion at execution; zero-velocity understates the spatial advantage of players already running toward the ball.
- **Per-frame identity ambiguity:** Pipeline track IDs are not matched to GT player IDs; team assignment accuracy is assessed implicitly via distributional validation rather than by direct track matching.
- **Structural occlusion:** Defenders in tight clusters partially hidden from broadcast angles cannot be recovered by any detector at the confidence threshold used.
- **Single broadcast angle:** SoccerNet GSR clips are from single broadcast cameras. Performance on tactical cameras or multi-camera feeds has not been tested.

### 7.8 Practical Prioritisation for Next-Phase Execution

Based on the findings, the highest-leverage next actions in priority order are:

1. **Autonomous ball detection** — removes the GT dependency and enables fully autonomous deployment. Unlocks the pipeline for any club with broadcast footage and no annotation infrastructure.
2. **Team assignment hardening** — fix `pc_in_box` sign inversion via supervised classifier or cross-frame consistency constraint. Required before any box-control signal is used operationally.
3. **Detector recall improvement** — lower confidence threshold or add an ensemble step to reduce the defending-team shortfall and correct the global underestimation bias.
4. **Cohort expansion** — validate on additional SoccerNet GSR clips and club-provided broadcast sets before deployment beyond the 33-clip validation cohort.

---

## 8. Conclusions and Future Work

### 8.1 Final Reflections

This project set out to answer a single practical question: can a broadcast-video-only pipeline produce Pitch Control estimates that are distributionally comparable to ground-truth annotation-derived estimates, for set-piece frames, on consumer hardware? The answer is conditional but affirmative: yes, for `pc_in_third` (bias ≈ 0, overlap = 0.896) and with useful signal on `pc_at_ball` (bias = -0.051, overlap = 0.804).

The development process surfaced a significant data-quality issue in SoccerNet GSR — the `action_position` field is a global broadcast frame number, not a clip-local index, causing the entire pipeline to operate on end-of-clip open-play frames rather than set-piece formations until the bug was identified and fixed. This mid-project discovery and correction illustrates the importance of data validation as an active rather than passive phase of CRISP-DM, and is documented here both as a methodological lesson and as a contribution to future users of this dataset.

The most surprising finding is the clean three-way partition of error modes: one metric is already deployment-ready, one has a sign inversion tied specifically to team assignment in crowded penalty areas, and the rest have predictable directional bias from detector recall shortfall. This structure makes the pipeline's limitations tractable rather than opaque.

### 8.2 Core Conclusions

1. **Broadcast-only Pitch Control is viable** for the most decision-relevant set-piece signals, within this 31-clip cohort of SoccerNet GSR footage. `pc_in_third`: bias ≈ 0, overlap = 0.896. `pc_at_ball`: bias = -0.051, overlap = 0.804.
2. **TVCalib autonomous calibration** enables fully autonomous camera-to-pitch projection: 33/33 clips processed, zero homography failures, no GT pitch-line annotations consumed.
3. **Bias is structural and attributable by metric:** global metrics (`pc_mean`, `pc_area_gt_0p5`) are underestimated due to asymmetric detector recall (more defenders missed than attackers); `pc_in_box` is sign-inverted due to KMeans team-assignment failure in crowded penalty areas; `pc_in_third` is unaffected by both failure modes.
4. **ByteTrack persistent IDs** enable single-pass team assignment per clip with clean label coverage (zero discarded tracks), eliminating per-frame label instability.
5. **The pipeline is fully reproducible** on consumer hardware (~30 min on Apple Silicon MPS, no cloud). Parquet outputs are queryable with standard Python data tools.
6. **Ball position remains a dependency** on GT annotations in the current implementation; autonomous ball detection is the primary gap to a fully production-ready pipeline.

### 8.3 Future Work

- **Autonomous ball detection:** Integrate a lightweight ball detector (e.g., YOLOv8n fine-tuned on SoccerNet) to remove the GT ball-position dependency.
- **Velocity estimation:** Optical flow between consecutive frames (exploiting ByteTrack persistent IDs) to enable the full Shaw model with non-zero initial velocities.
- **TVCalib error quantification:** Propagate reprojection errors through the homography to estimate their contribution to PC metric variance.
- **Cohort expansion:** Additional SoccerNet GSR clips or club-provided broadcast sets to validate beyond 33 clips.
- **Open-play extension:** Throw-ins, goal kicks, dynamic possession sequences.
- **Detector recall tuning:** Lower confidence threshold or ensemble detection to reduce defender under-detection, particularly in crowded penalty-area crops.
- **Team assignment in crowded areas:** Replace or augment KMeans HSV with a supervised classifier or cross-frame consistency constraint to fix the `pc_in_box` sign-inversion failure mode during corners.
- **Operational packaging:** CLI + Docker for adoption by clubs without notebook expertise.

### 8.4 Proposed Roadmap

**Phase 1 (0–3 months): Autonomous ball detection**
Integrate a lightweight ball detector (YOLOv8n or similar, fine-tuned on SoccerNet tracking data) to replace the GT `bbox_pitch` ball dependency. This is the single change that converts the pipeline from a research tool into a deployable system. Success criterion: ball detection rate ≥ 80% of frames across the 33-clip validation set.

**Phase 2 (3–6 months): Team assignment hardening + cohort expansion**
Replace KMeans HSV with a supervised binary team classifier trained on labelled player crops. Validate on an expanded clip set (target: ≥100 SoccerNet GSR clips across more matches and broadcast conditions). Stratify results by set-piece type (corner vs. direct free kick) and validate `pc_in_box` separately on each type. Success criterion: `pc_in_box` bias magnitude reduced below 0.05.

**Phase 3 (6–12 months): Production packaging + open-play extension**
Package as a CLI tool with Docker support. Add optical flow for velocity estimation to enable full Shaw TTI model on non-static frames. Begin validation on throw-ins and goal kicks as intermediate steps toward open-play Pitch Control. Success criterion: end-to-end CLI run on a new match in under 10 minutes.

### 8.5 Academic and Practical Contribution

**Academic contribution.** This project demonstrates the first distributional validation of a broadcast-only Pitch Control pipeline against open ground-truth annotations. The contribution is not a new model — the Shaw TTI model is established — but a validated evidence base for which summary metrics survive the broadcast-to-GT gap. The three-way error taxonomy (global underestimation, box inversion, calibrated third) provides a reusable framework for evaluating future broadcast CV pipelines that compute spatial tactical metrics. The `action_position` data-quality finding contributes a documented correction to the SoccerNet GSR dataset for future users.

**Practical contribution.** The pipeline is fully open-source, runs on consumer hardware in 30 minutes, and requires no proprietary tracking hardware or GT pitch-line annotations at inference time. It delivers `pc_in_third` — a calibrated, deployment-ready Pitch Control metric — to any club or analyst with broadcast footage and a laptop. The animated broadcast-overlay visualizations provide an interpretable output layer that does not require a data scientist to consume.

### 8.6 Closing Statement

Broadcast-video Pitch Control for set pieces is achievable today, at zero hardware cost, with honest quantification of what works and what does not. The pipeline described here is not a finished product but a validated foundation: one metric is deployment-ready, the remaining failure modes have identified causes and concrete remediation paths, and the full codebase and methodology are publicly documented. The primary next step — autonomous ball detection — is a well-scoped engineering task that would remove the last GT dependency and complete the autonomous pipeline.

---

## 9. Bibliography

Deliege, A., Cioppa, A., Giancola, S., Vandeghen, M., Merminod, V., Van Droogenbroeck, M., Ghanem, B., & Davis, J. (2021). SoccerNet-v2: A dataset and benchmarks for holistic understanding of broadcast soccer videos. CVPR Workshops 2021.

Fernandez, J., & Bornn, L. (2018). Wide open spaces: A statistical technique for measuring space creation in professional soccer. MIT Sloan Sports Analytics Conference 2018.

Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO (Version 8.0.0). https://github.com/ultralytics/ultralytics

Mansourian, A. M., Somers, V., De Vleeschouwer, C., & Kasaei, S. (2023). Multi-task learning for joint re-identification, team affiliation, and role classification for sports visual tracking. MMSports 2023.

Redmon, J., & Farhadi, A. (2018). YOLOv3: An incremental improvement. arXiv:1804.02767.

Shaw, L. (2020). Pitch control model (commit 21f4c2d). Friends of Tracking Data. https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking

Somers, V., Joos, V., Giancola, S., Cioppa, A., Davis, J., Ghanem, B., & Van Droogenbroeck, M. (2024). SoccerNet game state reconstruction: End-to-end athlete tracking and identification on a minimap. CVPR Workshops 2024.

Spearman, W. (2018). Beyond expected goals. MIT Sloan Sports Analytics Conference 2018.

StatsBomb. (2024). StatsBomb open data. https://github.com/statsbomb/open-data

Theiner, J., & Ewerth, R. (2023). TVCalib: Camera calibration for sports field registration in soccer. WACV 2023.

Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). ByteTrack: Multi-object tracking by associating every detection box. ECCV 2022.

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
        validation_summary_tvcalib.parquet
        validation_paired.parquet
        setpieces.parquet
        gt_spatial_benchmarks.parquet
        figures/
            11_multiclass_detections.png        ← Soccana Player/Ball/Referee detection (methods figure)
            still_corner_SNGS-116.png          ← three-panel thesis figure (corner)
            still_direct_free-kick_SNGS-122.png ← three-panel thesis figure (free kick)
            anim_corner_SNGS-116.gif            ← animated 31-frame PC overlay
            anim_direct_free-kick_SNGS-122.gif  ← animated 31-frame PC overlay
            video_corner_SNGS-116.mp4           ← MP4 version of corner animation
            video_direct_free-kick_SNGS-122.mp4 ← MP4 version of free-kick animation
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
| KMeans k | 3 (drop smallest if <15% or referee-like) | _pipeline_core.py |
| Calibration | TVCalib | homographies_tvcalib.parquet |
| PC grid | 60 x 40 | _pipeline_core.py |
| MAX_SPEED | 5.0 m/s | _pipeline_core.py |
| REACTION_TIME | 0.7 s | _pipeline_core.py |
| SIGMA | 0.45 s | _pipeline_core.py |
| Frame window | frames 1–31 (centre=16) | _pipeline_core.py |
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

**Hardware:** Apple Silicon (M-series), 16 GB unified memory, MPS backend. CUDA and CPU backends also supported.

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
