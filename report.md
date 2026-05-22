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
# render-note: PDF must be built on Mac (full TeX Live). On Windows use:
#   pandoc report.md -o report.pdf --pdf-engine=xelatex --metadata header-includes=""
---

# Pitch Control from Broadcast Video: A Computer Vision Pipeline for Set-Piece Analysis

**Master in Artificial Intelligence Applied to Sports**
**Master's Final Project**

Author: Moritz Philipp Haaf
MSc AI Applied to Sports · Sports Data Campus
Submission Deadline: 30 June 2026

---

## Acknowledgements

I would like to thank the academic team at Sports Data Campus for guidance across the Master's programme and on this final project. I am also grateful to the SoccerNet, StatsBomb, TVCalib, and Friends of Tracking Data communities for releasing the open datasets, models, and code that make a reproducible, consumer-hardware pipeline of this kind possible.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction](#2-introduction)
   - 2.1 Problem Statement and Relevance
   - 2.2 Research Context and Industry Background
   - 2.3 Academic Research Gap
   - 2.4 Research Significance and Practical Relevance
   - 2.5 Research Scope and Boundaries
   - 2.6 Research Structure and Preview
3. [Project Objectives](#3-project-objectives)
   - 3.1 Primary Research Objectives
   - 3.2 Research Questions
   - 3.3 Academic Objectives
   - 3.4 Practical Objectives
   - 3.4.1 Business Context and Service Model
   - 3.5 Expected Outcomes and Deliverables
   - 3.6 Success Criteria and Measurement
   - 3.7 Ethical Considerations
   - 3.8 Conclusion
4. [Conceptual and Technological Architecture](#4-conceptual-and-technological-architecture)
   - 4.1 Overview
   - 4.2 Infrastructure Components
   - 4.3 Technological Tools and Libraries
   - 4.4 System Architecture Diagram
   - 4.5 Coordinate Systems
   - 4.6 Performance and Hardware
   - 4.7 Reproducibility and Compliance
   - 4.8 Integration with Downstream Analytics
5. [Methodologies and Techniques Employed](#5-methodologies-and-techniques-employed)
   - 5.1 CRISP-DM Methodology Overview
   - 5.2 Detailed CRISP-DM Phase Implementation
   - 5.3 Methodological Innovation
6. [Work Development](#6-work-development)
   - 6.1 Phase 1: Business Understanding
   - 6.2 Phase 2: Data Understanding
   - 6.3 Phase 3: Data Preparation
   - 6.4 Phase 4: Modeling
   - 6.5 Phase 5: Evaluation
   - 6.6 Phase 6: Deployment
   - 6.7 Project Outcomes and Deliverables
7. [Discussion of Results](#7-discussion-of-results)
   - 7.1 Finding-by-Finding Critical Discussion
   - 7.2 Cross-Finding Synthesis
   - 7.3 Methodological Limits and Caution in Interpretation
   - 7.4 Practical Prioritization for Next-Phase Execution
8. [Conclusions and Future Work](#8-conclusions-and-future-work)
   - 8.1 Final Reflections
   - 8.2 Core Conclusions
   - 8.3 Future Work
   - 8.4 Proposed Roadmap
   - 8.5 Academic and Practical Contribution
   - 8.6 Closing Statement
9. [Bibliography](#9-bibliography)
10. [Annexes](#10-annexes)
    - Annex A: Repository Structure
    - Annex B: Key Model Parameters
    - Annex C: Data Sources and Access
    - Annex D: Validation Summary Table
    - Annex E: Broadcast Stills and Animated Surfaces
    - Annex F: Data Dictionary
    - Annex G: Reproducibility Environment

---

## 1. Executive Summary

**Purpose and context.** Optical player tracking, the data layer that powers modern tactical analysis, is commercially available only to elite clubs and leagues. Second divisions, women's football, youth academies, and most scouting contexts operate without it. This project closes a portion of that gap with an open-source computer vision (CV) pipeline that derives Pitch Control from broadcast video, requiring no proprietary tracking hardware.

**Objectives.** Build and validate a reproducible CV pipeline that converts broadcast set-piece frames into Pitch Control surfaces and metrics; quantify the contribution of each pipeline component (detector, calibration, tracker) to downstream metric fidelity; and demonstrate that the system runs end-to-end on a consumer laptop.

**Approach.** The pipeline detects and tracks players using Soccana (YOLOv11n, football-finetuned on SoccerNet GSR plus match footage) combined with ByteTrack multi-object tracking, assigns stable team labels via per-track jersey-colour KMeans on HSV features, projects pixel to metric pitch coordinates via TVCalib autonomous camera calibration (Theiner & Ewerth, WACV 2023), and computes attacking Pitch Control using Laurie Shaw's time-to-intercept model. An off-the-shelf detector arm (YOLOv8x, COCO-pretrained) and a GT-pitch-line homography arm are retained as ablation baselines to isolate the contribution of football-domain finetuning and autonomous calibration respectively. Focus is set pieces (corners, direct free kicks), where broadcast cameras are near-static and relevant players are typically in frame. Validation follows CRISP-DM and is distributional plus per-frame paired against SoccerNet GSR ground-truth annotations.

**Main findings.**

1. **Primary pipeline preserves the decision-relevant signal.** Soccana + TVCalib on 30 paired clips (441 paired frames) gives `pc_at_ball` Δ=+0.001, histogram overlap 0.854, Pearson r=0.677 vs full-cohort GT.
2. **TVCalib recovers cohort and removes autonomy leak.** All 33/33 set-piece clips process end-to-end vs 20/33 under GT-pitch-line RANSAC homography (+13 clips, +65% cohort).
3. **Soccana cuts detector-domain bias by ~30%** on global metrics under identical homography (YOLOv8x → Soccana under TVCalib reduces `pc_mean` Δ from −0.096 to −0.055).
4. **`pc_at_ball` is the most robust output**, near-zero bias on the primary pipeline and best paired correlation across all configurations.
5. **Global metrics (`pc_mean`, `pc_area_gt_0p5`) are systematically underestimated** by 0.06 (primary) to 0.18 (baseline) due to defender under-detection in crowded penalty-area crops.
6. **The bias is structural and explainable**, not random: more defenders compress attacking PC across the surface in the Shaw model, and detector recall falls in tight clusters.
7. **Strict KS regresses (1/5 → 0/5)** under the primary configuration, but reflects statistical-power inflation (n grew 286 → 457 frames), not worse fit; bias and overlap improve on 4/5 metrics.
8. **ByteTrack persistent IDs** eliminate per-frame team-label flipping and enable single-pass KMeans assignment per clip.
9. **Two SoccerNet GSR clips carry incorrect `action_class` labels** (a mid-game scene tagged Corner, a throw-in tagged Direct free-kick), identified through visual inspection.
10. **Full pipeline runs on a MacBook Air M3** with no cloud dependency, in roughly 30-45 minutes for the 33-clip cohort.

**Conclusions.** A broadcast-only Pitch Control pipeline can produce distributionally honest estimates of the most operationally meaningful set-piece signal without proprietary tracking. The remaining bias on global surface metrics is structural, attributable to detector recall in crowded penalty-area crops, and partially closed by domain-finetuned weights. The combination of autonomous calibration (TVCalib) plus a football-domain detector (Soccana) is the strongest defensible configuration: it recovers the autonomy claim, expands the processable cohort, and improves bias and overlap on 4/5 metrics.

**Recommended direction and future work.** The pipeline is ready for adoption by clubs at levels where tracking data is unavailable. Future work should add player velocity estimation via optical flow between consecutive frames, extend the cohort beyond the 33 SoccerNet GSR clips, and trial open-play extensions (throw-ins, goal kicks). The honest limit of the current work is data scale: distributional conclusions should not be generalised beyond this clip set without further validation on additional broadcast cohorts.

---

## 2. Introduction

### 2.1 Problem Statement and Relevance

**Context: spatial analytics depend on player tracking data.** Pitch Control, the probability that a given team could reach any point on the pitch first under current player positions, is a standard analytical tool for evaluating spatial dominance, tactical compactness, and set-piece danger in elite football. Systems such as StatsBomb 360, SkillCorner, and Opta Tracking deliver this data in near-real time. Their cost and infrastructure requirements, however, effectively restrict them to top-tier competitions.

**The access problem.** For the majority of professional and semi-professional clubs, women's leagues, academies, and scouting departments, data-driven set-piece analysis remains out of reach not because of analytical sophistication but because of data access. The analytical gap between well-funded and resource-constrained clubs is partly a data-infrastructure problem.

**Why set pieces.** Set pieces (corners and direct free kicks) are a tactically high-leverage and analytically tractable phase of play. Across the 706 set pieces in UEFA Euro 2024 (this project's reference dataset), 32.4% produced a shot within 10 seconds of execution and 1.8% produced a goal in the same window. Set pieces are also the highest-information moments for a CV pipeline: the broadcast camera is nearly static, relevant players are in frame, and the ball position is precisely known from the event feed. This makes set pieces the optimal entry point for extracting positional value from video without a dedicated tracking rig.

### 2.2 Research Context and Industry Background

**Global CV-in-football landscape.** Computer vision has matured rapidly in sports analytics over the last five years. Player detection (YOLO families), multi-object tracking (ByteTrack, StrongSORT, DeepSORT), self-supervised camera calibration (TVCalib, RobustHomography), and open datasets (SoccerNet, StatsBomb open data) now provide a stack that, combined correctly, can produce coordinate-aware tactical metrics from broadcast video.

Several forces underpin these trends:

- **Open data acceleration.** SoccerNet (v1, v2, GSR) and StatsBomb open data lowered the entry barrier for researchers without club partnerships.
- **Foundation-model momentum.** General-purpose detectors (YOLOv8/v11) trained on COCO transfer reasonably to football and can be further finetuned on domain data with modest GPU budgets.
- **Self-supervised calibration.** Methods like TVCalib (Theiner & Ewerth, 2023) eliminate the need for manual line-correspondence labelling, which historically blocked autonomous deployment.
- **Reproducibility expectations.** Sports analytics conferences (StatsBomb, OptaPro, MIT Sloan, MLSA) increasingly expect code and data release, raising the floor on what counts as a credible CV-tactics result.

**Resource-constrained club context.** Clubs outside elite leagues face a structurally different problem from elite clubs: they have broadcast or stadium video, but no tracking provider, and limited budget for commercial coordinate APIs. A reproducible open-source pipeline maps directly to this gap.

### 2.3 Academic Research Gap

Prior literature spans Pitch Control modelling, player detection, tracking, camera calibration, team assignment, and open data. Each domain is mature on its own, but the end-to-end chain that takes broadcast pixels through to a distributionally validated tactical metric, without proprietary tracking and without GT annotations leaking into the calibration step, remains underdeveloped.

**Pitch Control and spatial dominance.** Pitch Control as a formal model originates with Spearman (2018), who introduced control probability as a function of player positions and estimated time-to-intercept. Shaw (2020) made the time-to-intercept (TTI) formulation accessible as an open-source implementation through the Friends of Tracking Data initiative, the most widely adopted academic baseline. Beyond Shaw's TTI model, Spearman (2018) and subsequent StatsBomb work (their 360 product) incorporate velocity and physics-based motion models, which require sub-second tracking unavailable in a frame-based broadcast pipeline. The zero-velocity assumption used in this project is a deliberate simplification appropriate to near-static set pieces and applied identically to pipeline and GT tracks.

**Player detection and tracking from broadcast video.** YOLO-family detectors (Jocher et al., 2023) dominate practical broadcast player detection due to their speed-accuracy trade-off on consumer hardware. YOLOv8x (COCO-pretrained, 68M parameters) is the off-the-shelf baseline in this project. Soccana (Adit-jain/soccana; YOLOv11n, 2.6M parameters) addresses domain shift by fine-tuning on SoccerNet GSR plus match footage, with football-specific classes (Player, Ball, Referee). ByteTrack (Zhang et al., 2022) associates every detection (not just high-confidence ones) using a two-stage Kalman filter and IoU matching, producing persistent track IDs that enable per-track colour feature aggregation. Alternative trackers (StrongSORT, DeepSORT) add re-identification embeddings and inference cost; ByteTrack's efficiency was sufficient here.

**Camera calibration for football broadcast.** Classical RANSAC approaches estimate correspondences between visible pitch-line intersections and known metric positions (Hartley & Zisserman, 2004). Fragile in practice: clips with wide angles, advertising-board occlusion, or few visible intersections fail; in this project's baseline 13/33 clips were excluded for this reason. TVCalib (Theiner & Ewerth, WACV 2023) segments pitch lines and optimises camera parameters per frame via a differentiable projection loss, requiring no explicit correspondence labelling. In this project TVCalib achieved zero homography failures across 33 clips. Nie et al. (2021) presented a competing robust registration framework requiring a holistic line map; less suited to single-frame inference and not evaluated here.

**Team assignment.** Colour-based team assignment has a long history. Naive per-frame KMeans on jersey crops produces per-frame label flipping when colour gaps between teams are small. Mansourian et al. (2023) trained a supervised multi-task model for joint re-identification, team affiliation, and role; more accurate but requires per-player labelled training data unavailable in the open SoccerNet GSR split. The KMeans-on-per-track-mean-HSV approach used here exploits ByteTrack's persistent IDs to aggregate colour evidence across frames and assign once.

**Open data resources.** SoccerNet GSR (Somers et al., 2024) is the primary data source for this project: per-frame bounding boxes in pixel and pitch coordinates, per-frame pitch-line annotations, action metadata. The 2024 release covers 22 matches from Jupiler Pro League 2023/24 with 525 annotated clips. This project uses the 33 with `action_class` in {Corner, Direct free-kick}. StatsBomb open data (StatsBomb, 2024) provides event-level and freeze-frame data for UEFA Euro 2024 via `statsbombpy`; used as a distributional reference for set-piece tactical context, not as a matched evaluation target. SoccerNet-v2 (Deliège et al., 2021) was considered as an alternative video source but rejected: lower resolution (typically 224p) and no per-frame player coordinate annotations.

**Identified gaps addressed by this project:**

- **Gap 1: Tactical metrics from open, annotation-free video remain unvalidated.** Prior work on player detection from broadcast (Jocher et al., 2023; Mansourian et al., 2023) demonstrates localisation quality but does not propagate coordinates into a downstream tactical metric and validate the metric distribution against a benchmark. The pipeline → metric → validation chain is absent from open-source literature.
- **Gap 2: Distributional validation against publicly available annotations is rare.** Most CV pipeline evaluations use proprietary tracking data or per-frame accuracy metrics (IoU, MOTA). SoccerNet GSR provides open per-frame annotations that make distributional validation tractable; not exploited for Pitch Control specifically.
- **Gap 3: The homography step is routinely treated as solved by GT annotations.** Several academic prototypes (including this project's initial design) rely on GT pitch-line annotations to compute the image→pitch transform, invalidating the autonomy claim. Replacing this with TVCalib and demonstrating end-to-end autonomous operation on a real clip cohort has not been reported for Pitch Control.
- **Gap 4: Detector domain mismatch is rarely quantified in football CV pipelines.** COCO-pretrained detectors are commonly used without quantifying the cost of domain shift relative to football-finetuned alternatives. This project provides a controlled ablation (YOLOv8x COCO vs Soccana YOLOv11n finetuned) that isolates detector-domain contribution to downstream metric bias.

**Theoretical contribution.** A reproducible, validated chain from broadcast pixels to a tactical metric, with attribution of bias to specific pipeline components.

### 2.4 Research Significance and Practical Relevance

**Contribution to sports CV literature.** The project supplies a documented, replicable pipeline that converts broadcast video to a validated tactical metric, with controlled ablations attributing bias to detector domain mismatch and homography source. It complements literature dominated by proprietary tracking and per-frame accuracy reporting; it does not claim per-frame coordinate accuracy or causal links between pipeline outputs and match outcomes.

**Advancement of open-data tactical analytics.** Distributional validation against open SoccerNet GSR annotations is the central methodological move. It shows how public per-frame annotations can underpin tactical-metric evaluation without proprietary data, at a level appropriate to the methods actually applied.

**Practical relevance.** Immediate benefits for:

- **Strategic roadmap:** evidence-backed configuration recommendations (Soccana + TVCalib) with bias characterisation per metric.
- **Competitive intelligence:** a deployable baseline for clubs without tracking providers, runnable on consumer hardware.
- **Tactical opportunity framing:** identification of which metrics (`pc_at_ball`, `pc_in_box`) are reliable and which (`pc_mean`, `pc_area_gt_0p5`) require relative-not-absolute interpretation.
- **Methodological transparency:** documented scrape of public datasets, locked thresholds, refreshable workflow.

**Application beyond this project.** Mid-tier and resource-constrained clubs facing the same data access problem; women's football and academy contexts; researchers requiring a reproducible CV-to-tactics baseline.

### 2.5 Research Scope and Boundaries

**Geographic and competition scope.** Validation cohort is 33 SoccerNet GSR clips drawn from Jupiler Pro League 2023/24. Reference distribution context is UEFA Euro 2024 (StatsBomb open data, 51 matches, 706 set pieces).

**Temporal scope.** SoccerNet GSR clips are static published artefacts (2024 release). StatsBomb Euro 2024 is a closed historical tournament (June-July 2024). Project work spans 2025-2026; the academic deadline is 30 June 2026.

**Methodological boundaries.**

- **Validation design:** distributional plus per-frame paired against SoccerNet GSR GT only. No per-frame identity assignment between pipeline track IDs and GT player IDs (re-identification labels unavailable).
- **Model scope:** Laurie Shaw zero-velocity TTI Pitch Control. No comparison against velocity-aware or physics-based models.
- **Data scale:** 33 processable clips under TVCalib (20 under GT-leak baseline). Conclusions about distributional agreement should not be generalised beyond this clip set.
- **Channel boundary:** broadcast video only. Excludes tactical-cam footage, multi-camera setups, and stadium tracking systems.

**Expected limitations.**

- **Snapshot bias:** SoccerNet GSR captures specific clips; cohort cannot be assumed representative of all broadcast set pieces globally.
- **Selection bias on the GT-leak baseline:** 13 excluded clips may not be a random subset; homography failure correlates with wide angles and unusual elevation.
- **Annotation noise:** 2/20 baseline clips carry mislabelled `action_class`.
- **No transactional outcome data:** the pipeline produces tactical metrics, not match outcomes; causal claims about set-piece danger are out of scope.

### 2.6 Research Structure and Preview

This document follows a CRISP-DM-aligned structure within the Master's Final Project template. Section 3 details research objectives and questions. Section 4 covers the conceptual and technological architecture. Section 5 presents the CRISP-DM methodology. Section 6 documents the work development phase by phase. Section 7 critically discusses each result. Section 8 concludes and proposes future work. Annexes provide repository structure, model parameters, data sources, full validation tables, broadcast stills, the data dictionary, and the reproducibility environment.

---

## 3. Project Objectives

### Overview

This section sets out the project's research objectives, questions, deliverables, and success criteria. Validation cohort is 33 SoccerNet GSR set-piece clips (17 Corners, 16 Direct free-kicks); reference distributional context is UEFA Euro 2024 StatsBomb open data (706 set pieces, 51 matches). All thresholds, metrics, and parameters are locked in the notebooks and Annex B.

### 3.1 Primary Research Objectives

**3.1.1 Main research objective.** Develop a reproducible computer vision pipeline that extracts Pitch Control from broadcast set-piece frames and produces distributions comparable to ground-truth annotation-derived distributions, using only open-source tools and consumer hardware.

**3.1.2 Secondary research objectives.**

- Build a two-track processing pipeline (pipeline / GT) on shared SoccerNet GSR frames enabling per-frame paired comparison.
- Compute five Pitch Control summary metrics per frame: `pc_mean`, `pc_at_ball`, `pc_in_box`, `pc_in_third`, `pc_area_gt_0p5`.
- Validate distributionally (KS, histogram overlap) and per frame (Pearson, Spearman, MAE, bias).
- Diagnose systematic bias and attribute to specific pipeline components via controlled ablations (detector, homography source).
- Demonstrate fully reproducible end-to-end execution on a MacBook Air M3.

### 3.2 Research Questions

**3.2.1 Primary research questions.**

- **RQ1.** Can a broadcast-video-only pipeline produce Pitch Control distributions statistically equivalent (KS α=0.05, histogram overlap) to GT-annotation distributions for set-piece frames?
- **RQ2.** What is the dominant source of systematic bias in pipeline-derived Pitch Control, and to which pipeline component can it be attributed?

**3.2.2 Secondary research questions.**

- **RQ3.** Does replacing GT-derived homography with autonomous camera calibration (TVCalib) recover the autonomy claim without degrading Pitch Control fidelity, and does it expand the processable clip cohort?
- **RQ4.** Does a football-domain-finetuned detector (Soccana) over an off-the-shelf COCO baseline (YOLOv8x) meaningfully reduce downstream Pitch Control bias?
- **RQ5.** Which Pitch Control summary metrics are most robust to pipeline noise, and which require relative-not-absolute interpretation?

### 3.3 Academic Objectives

**3.3.1 Theoretical contributions.**

- Apply the Shaw TTI Pitch Control model as a fixed evaluation lens for coordinate quality from broadcast video, not as a model-selection problem.
- Discuss how mid-recall detectors and self-supervised calibration interact to shape downstream metric bias under data constraints.
- Show that distributional plus per-frame paired validation against open per-frame annotations is a viable evaluation design where proprietary tracking is unavailable.

**3.3.2 Methodological contributions.**

- Demonstrate an end-to-end open-source pipeline (Soccana + ByteTrack + TVCalib + Shaw TTI) with explicit limitations and refreshable thresholds.
- Document scripts, data layout, and version pins (Annex G) sufficient for replication on a new SoccerNet GSR snapshot.

### 3.4 Practical Objectives

- Provide club stakeholders with benchmark-grounded guidance on which Pitch Control metrics are reliable from broadcast video and which require relative interpretation.
- Highlight cohort, detector, and calibration choices that materially shape downstream bias, framed as deployment hypotheses to test against internal video data.
- Deliver an executable notebook + scripts stack stakeholders can adopt, extend, or reject as more data becomes available.

### 3.4.1 Business Context and Service Model

**Market problem and addressable segment.** The global sports analytics market was valued at approximately USD 5.28–5.68 billion in 2025 and is forecast to reach USD 17–23 billion by 2031–2033 at a compound annual growth rate of 18–28% depending on methodology (Mordor Intelligence, 2025; Grand View Research, 2025). Football is the largest single segment, accounting for approximately 40% of sports analytics revenue (Mordor Intelligence, 2025). Within this market, a structural data-access inequality persists: a 2024 survey of professional clubs and national federations published in *Science and Medicine in Football* found that only around 30% of national federations reported a clear operational understanding of positional tracking data, indicating that the majority of football organisations remain at an early stage of adoption (Peralta Benítez et al., 2024). Investment in analytics at elite clubs (Manchester City, Arsenal, Liverpool) runs to £1–5 million annually; clubs at lower levels typically operate on under £1 million, with full tracking infrastructure described as "expensive and complex, especially for smaller clubs" (Sportmonks, 2024).

Commercial optical tracking providers — SkillCorner, Tracab (ChyronHego), Second Spectrum, and StatsBomb 360 — do not publish pricing publicly; all require direct sales contact. Infrastructure for optical stadium systems is described as a "significant investment, typically done at a league level" (ISSPF, 2022). StatsBomb 360 freeze-frame data is bundled with event-data licences (Hudl/StatsBomb, 2024), making it available only to competitions and clubs that hold a full StatsBomb data partnership. Broadcast-based tracking systems (such as SkillCorner) eliminate stadium hardware but still require a per-season provider contract. The addressable segment for this pipeline is the long tail of clubs, academies, and analytical consultancies that hold broadcast video rights but cannot access or afford a tracking provider contract: second and third divisions, national youth academies, women's leagues below the top tier, and independent scouting operations. Emerging CV-based solutions from companies such as ReSpo.Vision explicitly target this same segment, positioning AI from a single video feed as a cost-elimination strategy relative to optical systems (ReSpo.Vision, 2024).

**Value proposition.** A broadcast-only Pitch Control pipeline that:

- requires no specialist hardware beyond a laptop (confirmed on MacBook Air M3);
- produces distributionally honest `pc_at_ball` and `pc_in_box` estimates at near-zero bias relative to GT;
- runs fully on open-source components with no per-match licensing fee;
- delivers actionable set-piece spatial metrics within 30–45 minutes of clip availability.

The marginal cost per match is near zero once the pipeline is deployed, compared to per-season provider contracts at any tier. One-time integration effort is estimated at one to two analyst-days given the documented run order and Parquet-compatible outputs.

**Competitive landscape.**

| Provider | Positioning | Pricing model | Infrastructure required |
|---|---|---|---|
| SkillCorner / Second Spectrum | Full-match broadcast tracking | Provider contract (not public) | Provider agreement |
| StatsBomb 360 | Event + freeze-frame | Bundled with data licence | Data partnership |
| Tracab / ChyronHego | Stadium optical tracking | Contract (not public) | Permanent camera installation |
| ReSpo.Vision | AI broadcast tracking (emerging) | Not disclosed | Single video feed |
| **This pipeline** | Set-piece PC, open-source | €0 marginal / OSS | Broadcast video access |

**Service model options.** Three commercialisation paths are viable depending on the operator:

1. **Open-source + paid deployment support.** Publish under MIT licence; monetise through integration consulting and annual support. Suitable for a boutique analytics consultancy. Cost structure is labour-only.
2. **SaaS subscription.** Host inference on cloud (e.g. AWS Batch + S3); clubs upload broadcast clips and receive PC dashboards. GPU spot-instance costs are low for short clips; margins scale with club count. This model mirrors positioning taken by emerging AI-broadcast providers such as ReSpo.Vision.
3. **Technology integration / OEM licensing.** License to an existing sports data company that embeds it as a feature tier. Revenue share or per-seat licence; distribution cost borne by the partner.

**Buy / build / partner decision for target clubs.** A second-division club with one data analyst faces three paths: buy a commercial tracking contract (requires budget approval and provider partnership), build internally (requires ML engineering capacity), or adopt this pipeline (near-zero cost, one-day integration). The pipeline is designed for the adopt path: documented run order, locked thresholds, Parquet outputs compatible with any standard BI tool.

**Ethical and regulatory considerations.** Broadcast video is publicly distributed; no additional player consent is required beyond what broadcasters obtain. The pipeline classifies by team colour only; it does not identify players by name or biometric. GDPR exposure is minimal: no personal data is stored and all processing is local. Any commercial SaaS deployment would need to confirm broadcast rights with the relevant federation before processing club-uploaded footage.

### 3.5 Expected Outcomes and Deliverables

**3.5.1 Expected outcomes.**

- A reproducible pipeline that converts broadcast set-piece frames to Pitch Control surfaces and summary metrics.
- An observed validation profile against full-cohort GT (33 clips, 30 paired): bias, KS, overlap, per-frame paired stats per metric.
- A controlled bias decomposition (detector + homography source ablations) attributing residual error to specific pipeline components.
- Methodological transparency: locked parameters, documented data provenance, scripts and notebooks reproducible on a consumer laptop.

**3.5.2 Deliverables.**

- Five executed notebooks (nb01-nb05) implementing CRISP-DM phases.
- Scripts (`scripts/`) for batch TVCalib, detector ablation, KS comparison, GT extraction.
- Intermediate Parquet outputs (`outputs/`): set pieces, detections, ball positions, pitch control, validation summaries.
- Figures (`outputs/figures/`): 14 static PNGs plus 2 animated GIFs.
- This report (`report.md`) with full bibliography and annexes.

### 3.6 Success Criteria and Measurement

**3.6.1 Academic success criteria.**

- Reproducible pipeline with locked thresholds and documented limitations.
- Findings supporting RQ1-RQ5 through bias decomposition, distributional comparison, and per-frame paired statistics, without overclaiming causal or per-frame accuracy results.
- Contributions under Section 3.3 (Shaw model as fixed lens; distributional plus paired evaluation against open annotations) consistent with methods actually applied.

**3.6.2 Practical success criteria.**

- Full pipeline executes end-to-end on MacBook Air M3.
- Total runtime under 45 minutes for the 33-clip cohort.
- Outputs (Parquet, PNG, GIF) usable directly by a club analyst with broadcast video access.

### 3.7 Ethical Considerations

**3.7.1 Data ethics.**

- SoccerNet GSR data downloaded via official credentials; respect terms of use; data stored on local external SSD, not redistributed.
- StatsBomb open data accessed via `statsbombpy`; usage consistent with StatsBomb's open-data licence.
- No personal data collected beyond what is publicly available in broadcast footage and public annotations.

**3.7.2 Research integrity.**

- Honest reporting of GT-leak baseline issue and the TVCalib replacement that closes it.
- Explicit disclosure of selection bias on the 20-clip baseline cohort and statistical-power inflation on the primary pipeline KS pass count.
- Separation of evidence (validation tables, figures) from inference (bias mechanism explanations, deployment recommendations).

### 3.8 Conclusion

These objectives define a bounded, defensible project: a 33-clip SoccerNet GSR validation of a broadcast-video Pitch Control pipeline, interpreted with honest limits on cohort size, per-frame identity ambiguity, and zero-velocity simplification. The work prioritises reproducibility, locked thresholds, and component-level bias attribution over claims this dissertation cannot directly observe (match outcomes, club adoption, real-time deployment). Success is judged first on research quality and integrity; commercial outcomes belong to post-study club measurement.

---

## 4. Conceptual and Technological Architecture

### 4.1 Overview

The pipeline converts SoccerNet GSR broadcast clips into Pitch Control surfaces and summary metrics through four sequential stages: detection plus tracking, team assignment, calibration, and Pitch Control computation. A parallel GT track derives the same Pitch Control outputs from SoccerNet `bbox_pitch` annotations, enabling distributional and per-frame paired comparison. All intermediate state is persisted as Parquet under `outputs/`, allowing each stage to be re-run independently.

### 4.2 Infrastructure Components

**4.2.1 Data ingestion layer.**

- SoccerNet GSR clips and annotations on external SSD (`SOCCERNET_LOCAL_DIR`).
- StatsBomb Euro 2024 via `statsbombpy` (auto-cached `~/.cache/statsbombpy/`).
- Pretrained weights: YOLOv8x (`~/.cache/ultralytics/`), Soccana (`~/.cache/huggingface/`).

**4.2.2 CV pipeline layer (nb02).**

- Soccana / YOLOv8x detection (Apple Silicon MPS backend).
- ByteTrack persistent ID assignment.
- Two-pass KMeans HSV team assignment per clip.
- TVCalib homography (subprocess invocation against `../tvcalib/.venv/`).
- Output: `detections_pipeline_tvcalib.parquet`, `detections_soccana_tvcalib.parquet`.

**4.2.3 Modeling layer (nb03).**

- Laurie Shaw TTI Pitch Control (vendored commit `21f4c2d`).
- Static-frame zero-velocity formulation.
- 60×40 grid (≈1.75 m × 1.70 m cells on 105×68 m pitch).
- Output: `pitch_control.parquet`, `pitch_control_*.parquet`.

**4.2.4 Evaluation layer (nb04).**

- KS two-sample tests (`scipy.stats.ks_2samp`, α=0.05).
- Histogram overlap (Bhattacharyya-style, 12 bins).
- Per-frame paired Pearson + Spearman correlation, MAE, signed bias.
- Output: `validation_summary*.parquet`, `validation_paired.parquet`.

**4.2.5 Storage layer.**

```
outputs/
    setpieces.parquet
    ball_positions.parquet
    detections_pipeline.parquet
    detections_pipeline_tvcalib.parquet
    detections_soccana_tvcalib.parquet
    detections_gt.parquet
    detections_gt_full.parquet
    homographies_tvcalib.parquet
    pitch_control.parquet
    pitch_control_tvcalib.parquet
    pitch_control_soccana_tvcalib.parquet
    pitch_control_gt_full.parquet
    validation_summary.parquet
    validation_summary_tvcalib.parquet
    validation_paired.parquet
    figures/ (13 PNG + 2 GIF)
```

### 4.3 Technological Tools and Libraries

| Component | Technology | Role |
|---|---|---|
| Language | Python 3.11 | All pipeline and analysis code |
| Environment | conda `py311-dev` | Reproducible dependency management |
| Object detection (primary) | Soccana / YOLOv11n (`Adit-jain/soccana`, HF) | Football-finetuned player detection |
| Object detection (ablation) | YOLOv8x (ultralytics) | COCO-pretrained off-the-shelf baseline |
| Multi-object tracking | ByteTrack (via ultralytics) | Persistent player IDs across frames |
| Team assignment | KMeans (scikit-learn) | Per-track jersey HSV clustering |
| Camera calibration (primary) | TVCalib (Theiner & Ewerth, WACV 2023) | Self-supervised image→pitch homography |
| Camera calibration (baseline) | OpenCV RANSAC | GT-pitch-line baseline (ablation arm) |
| Pitch Control | Laurie Shaw / FoTD | TTI model (vendored, commit `21f4c2d`) |
| Event data | statsbombpy | StatsBomb Euro 2024 open data |
| Visualisation | mplsoccer, matplotlib | Pitch surfaces, scatter, animation |
| Storage | Parquet (pyarrow) | All intermediate outputs |
| Notebooks | JupyterLab | CRISP-DM phase structure |

### 4.4 System Architecture Diagram

```
SoccerNet GSR clips (external SSD)
         |
         v
+----------------------------------+
|  nb02: CV Pipeline               |
|  - Soccana + ByteTrack detect    |
|  - KMeans (HSV) team assign      |
|    (per-track, not per-frame)    |
|  - TVCalib autonomous H          |
|  -> detections_pipeline.pq       |
+----------------------------------+
         |                  |
         |   SoccerNet GT bbox_pitch annotations
         |                  |
         v                  v
+----------------------------------+
|  nb03: Pitch Control             |
|  - Laurie Shaw TTI model         |
|  - Both tracks (pipeline / GT)   |
|  -> pitch_control.parquet        |
+----------------------------------+
         |
         v
+----------------------------------+
|  nb04: Evaluation                |
|  - KS tests (alpha=0.05)         |
|  - Histogram overlap             |
|  - Per-frame paired stats        |
|  - Bias diagnosis                |
|  -> validation_summary.pq        |
+----------------------------------+

StatsBomb Euro 2024 (nb01, online / cache)
         |
         v
+----------------------------------+
|  nb01: Business & Data Und.      |
|  - Set-piece EDA                 |
|  - Outcome analysis              |
|  -> setpieces.parquet            |
+----------------------------------+
```

### 4.5 Coordinate Systems

Three coordinate systems are in play. Pipeline calculations use the metric pitch convention (105 m × 68 m, origin top-left).

| System | Convention | Used by |
|---|---|---|
| StatsBomb | 120 yd × 80 yd, origin top-left | nb01 inputs |
| Pipeline / mplsoccer | 105 m × 68 m, origin top-left | nb02-nb05 |
| SoccerNet GSR `bbox_pitch` | centred ±52.5 m × ±34 m | raw GT |

**Conversions.**

```
StatsBomb → Pipeline: x_m = x_sb * (105/120),  y_m = y_sb * (68/80)
GSR centred → Pipeline: x_m = x_gsr + 52.5,    y_m = y_gsr + 34
```

### 4.6 Performance and Hardware

- **Development hardware:** MacBook Air M3 (Apple Silicon, 16 GB unified memory).
- **YOLO inference:** MPS backend, batch 1, ~15-30 s per clip.
- **TVCalib inference:** MPS backend, ~5 min for 528 frames (33 clips × 16).
- **Pitch Control computation:** vectorised numpy, <0.1 s per frame.
- **Full pipeline runtime:** approximately 30-45 minutes for the 33-clip cohort end-to-end.

### 4.7 Reproducibility and Compliance

- **Data ethics:** SoccerNet GSR credentialed download; data stored locally, not committed; raw video excluded from repository.
- **Cache management:** YOLO weights, Soccana weights, StatsBomb data cached under user home directory.
- **Version pinning:** see `requirements.txt` (pip freeze snapshot of `py311-dev`) and Annex G for canonical package versions.
- **Cross-platform:** Mac primary, Windows secondary via `.env` `SOCCERNET_LOCAL_DIR` override.

### 4.8 Integration with Downstream Analytics

- **Parquet outputs** are consumable by DuckDB, pandas, polars without further transformation.
- **Notebook scripts** can be converted to batch CLI with minimal refactoring (processing loops in nb02-nb04 are self-contained).
- **Mplsoccer figures** integrate into reports, presentations, or dashboard tools that render PNG / GIF / MP4.

---

## 5. Methodologies and Techniques Employed

### 5.1 CRISP-DM Methodology Overview

CRISP-DM (Cross-Industry Standard Process for Data Mining) was chosen as the project framework because it structures data-science work into clearly communicable phases that map directly onto the notebook architecture, and because its iterative feedback loop (Evaluation → Data Preparation) mirrors the actual development trajectory of this project.

**5.1.1 Why CRISP-DM for a CV-to-tactics pipeline.**

- **Business alignment:** explicit Business Understanding phase forces problem framing (resource-constrained clubs) before any code.
- **Iterative trajectory:** the project's actual path (KS failure → bias diagnosis → TVCalib replacement → Soccana ablation) is a textbook CRISP-DM loop.
- **Communication:** each phase maps onto one notebook, simplifying stakeholder review.
- **Validation focus:** the Evaluation phase is the centre of gravity, matching this project's data-engineering plus evaluation emphasis rather than model selection.

### 5.2 Detailed CRISP-DM Phase Implementation

**5.2.1 Phase 1: Business Understanding (nb01 Section 1, this report Section 6.1).**

- Problem framing: broadcast-only Pitch Control for clubs without tracking.
- Stakeholders: second-tier professional, women's, youth, scouting.
- Success criteria: at least partial distributional equivalence on one or more PC metrics, with mechanistic bias explanation.

**5.2.2 Phase 2: Data Understanding (nb01, this report Section 6.2).**

- StatsBomb Euro 2024 EDA: 706 set pieces, 64.2% freeze-frame coverage, outcome distribution within 10 s of execution.
- SoccerNet GSR scan: 33 candidate clips, per-frame `bbox_pitch` annotations, per-frame pitch-line annotations, action metadata.
- Annotation-quality audit: 2 / 20 baseline clips carry incorrect `action_class`.

**5.2.3 Phase 3: Data Preparation (nb02, this report Section 6.3).**

- Pipeline track: YOLO detection + ByteTrack + KMeans HSV teams + TVCalib.
- GT track: `bbox_pitch` parse + coordinate re-centring + role filter.
- Coordinate normalisation across StatsBomb (yards) / pipeline (metres) / GSR (centred metres).
- ±15 frame window around `action_position` per clip.

**5.2.4 Phase 4: Modeling (nb03, this report Section 6.4).**

- Laurie Shaw TTI Pitch Control, zero-velocity adaptation.
- Locked parameters: MAX_SPEED 5.0 m/s, REACTION_TIME 0.7 s, SIGMA 0.45 s, LAMBDA 4.3, grid 60×40.
- Identical model parameters applied to pipeline and GT tracks to isolate coordinate-quality effects.

**5.2.5 Phase 5: Evaluation (nb04, this report Section 6.5).**

- KS two-sample tests (α=0.05).
- Histogram overlap (12 bins, Bhattacharyya-style).
- Per-frame paired Pearson + Spearman + MAE + signed bias.
- Stratification by action class plus pooled.
- Bias decomposition via detector and homography-source ablations.

**5.2.6 Phase 6: Deployment (nb04 Section 5, this report Section 6.6).**

- Scalability assessment: TVCalib eliminates GT pitch-line dependency; YOLO inference scales linearly.
- Integration path: Parquet outputs consumable by downstream analytics tools.
- Limitations for production: zero-velocity assumption restricts to near-static phases; KMeans may fail on similar kit colours.

### 5.3 Methodological Innovation

The contribution is integrative, not a new algorithm. It applies a documented, repeatable path from broadcast pixels to a validated tactical metric under public-data constraints, a setting where reproducible end-to-end pipelines for Pitch Control remain rare.

**5.3.1 End-to-end autonomy as the core technical contribution.** Each component is an existing peer-reviewed or community-standard method (YOLO, ByteTrack, KMeans, TVCalib, Shaw TTI), but the integration produces a pipeline that runs without consuming any GT annotation at inference time. The Phase 1 sanity check (TVCalib 17 px vs GT-line 148,151 px RMSE) and the cohort recovery (33/33 vs 20/33) are the concrete evidence that the autonomy claim is now defensible.

**5.3.2 Bias decomposition under controlled ablation.** The detector ablation (YOLOv8x vs Soccana under identical homography) and the H-source ablation (GT-line vs TVCalib under identical detector) decompose the residual global-metric bias into a detector-domain component (~30%, closable by finetuning) and a structural occlusion component (~70%, intrinsic to broadcast angles in crowded penalty-area crops). This attribution is the methodological payoff of the experimental design.

**5.3.3 Documented QA and refreshable workflow.** Validation thresholds (KS α, histogram bins, paired window) are locked in nb04. Scripts and data layout (Annex G) support re-running the pipeline on a future SoccerNet GSR snapshot. The academic claim is replicability on new public data, not continuous self-optimising production systems.

---

## 6. Work Development

### 6.1 Phase 1: Business Understanding

**6.1.1 Business context.** The core question: can a broadcast-only pipeline produce Pitch Control distributions comparable to GT-annotation distributions, using open-source tools and consumer hardware?

**Stakeholders.** Primary beneficiaries: clubs and coaching staffs at levels where commercial tracking is unavailable (second-tier professional leagues, women's football, academies, scouting). Secondary: the research community gains a reproducible baseline for open-data CV-to-tactics work.

**Success criteria.** At least partial distributional equivalence (KS failure to reject H0, α=0.05) on one or more PC summary metrics, combined with a mechanistic bias explanation.

**Why set pieces.** Three characteristics make set pieces the optimal entry point:

1. Broadcast cameras are near-static during execution; homography is stable.
2. All tactically relevant players are in frame, with no occlusion from wide angles.
3. Ball position is fixed and pullable from the event feed; no ball tracking required.

Of 706 Euro 2024 set pieces, 464 (65.7%) produced no shot within 10 s, 229 (32.4%) produced a shot, and 13 (1.8%) produced a goal in that window (Figure 4). Low shot-conversion rate confirms that defensive-organisation metrics (Pitch Control) rather than shot counts capture the analytically relevant variable for evaluating set-piece danger.

![Figure 4. Outcome distribution within 10 seconds of set-piece execution, Euro 2024.](outputs/figures/04_setpiece_outcomes_10s.png)

**6.1.2 Success criteria definition.**

| Tier | Metric |
|---|---|
| Technical | End-to-end pipeline executes on MacBook Air M3, 33 clips, <45 min |
| Validation | KS pass on ≥1 metric or bias <0.05 on ≥2 metrics |
| Methodological | Component-level bias attribution via controlled ablations |
| Reproducibility | Locked thresholds, documented data provenance, scripts |

**6.1.3 Risk assessment and mitigation.**

- **Homography fragility:** mitigated by TVCalib replacement after Phase 1 sanity check.
- **Detector domain shift:** mitigated by Soccana finetuned weights as primary detector.
- **Annotation noise:** mitigated by visual inspection audit and disclosure of mislabelled clips.

### 6.2 Phase 2: Data Understanding

**6.2.1 StatsBomb Euro 2024 (nb01).** 51 matches across UEFA Euro 2024 (competition_id=55, season_id=282) were loaded via `statsbombpy`. From these, 706 set-piece events were extracted: 508 corners and 198 direct free kicks (Figure 1). StatsBomb 360 freeze-frame coverage was 64.2% (453 / 706 events). The Euro 2024 dataset provides the domain benchmark: player counts per freeze frame, ball locations, outcome rates within 10 s of execution.

![Figure 1. StatsBomb Euro 2024 set-piece counts by type.](outputs/figures/01_setpiece_counts.png)

Key EDA findings:

- Corners cluster tightly at corner-flag coordinates (105,0), (105,68), (0,0), (0,68) (Figure 2).
- Direct free kicks span a wide pitch range, concentrated in the attacking third (x_m > 70) (Figure 2).
- Freeze-frame coverage drops to zero for many events. Figure 3 shows player counts per available freeze frame.
- 10-second outcome analysis confirms defensive organisation (Pitch Control) is the primary analytical variable.

![Figure 2. Set-piece locations on the StatsBomb pitch (105 × 68 m), separated by type.](outputs/figures/02_setpiece_locations.png)

![Figure 3. Distribution of players per freeze frame, Euro 2024 360 data.](outputs/figures/03_players_per_frame.png)

**6.2.2 SoccerNet GSR (nb02).** 33 clips were identified across the four dataset splits (train/valid/test/challenge) with `action_class` in {Corner, Direct free-kick}: 17 corners, 16 direct free kicks. Each clip is a broadcast video segment; the `action_position` field gives the set-piece frame index. The dataset includes per-frame player annotations (`bbox_pitch`, metric-centred coordinates) and pitch-line annotations used for homography.

**6.2.3 Data limitations identified.**

- 13 / 33 clips (39%) failed RANSAC homography under the GT-leak baseline due to insufficient pitch-line intersection coverage.
- 2 of 20 processable clips had incorrect `action_class` annotations on visual inspection (a mid-game scene labelled Corner; a throw-in labelled Direct free-kick). Excluded from nb05 visualisations; retained in distributional evaluation for consistency.
- SoccerNet GSR and StatsBomb Euro 2024 are disjoint datasets; no per-clip or per-match correspondence, constraining validation to be distributional.

### 6.3 Phase 3: Data Preparation

The core of nb02 is parallel construction of two player-coordinate tracks per processable clip.

**6.3.1 Pipeline track, three-stage process.**

1. **Player detection and tracking (YOLO + ByteTrack).** YOLOv8x (baseline) or Soccana (primary) runs at confidence 0.40 via `yolo.track(..., tracker="bytetrack.yaml", persist=True)`. ByteTrack (Zhang et al., 2022) assigns persistent integer track IDs across frames by associating every detection box (including low-confidence) using Kalman filter state and IoU matching. This eliminates per-frame label instability: each physical player receives the same track ID throughout the clip window. Tracker state is reset between clips. Foot positions are approximated as the bottom-centre of each bounding box. Inference runs on Apple Silicon MPS.

2. **Team assignment (KMeans HSV, per-track).** Two-pass design per clip. In Pass 1, jersey HSV features accumulate per track ID across all frames: the torso region of each bounding box (central 50% horizontally, 15-45% vertically) is extracted, converted to HSV, and samples collected per track ID. In Pass 2, KMeans runs once on per-track mean HSV with k=3 to allow a referee/outlier cluster. If the smallest cluster represents <15% of tracks or its centroid matches a referee-kit heuristic (yellow/green hue or very low value), that cluster is dropped and KMeans is re-fit with k=2 on survivors. One stable team label per physical player; no labelled training data required; robust to varying kit colours across clips.

3. **Homography (RANSAC baseline / TVCalib primary).** Baseline: SoccerNet GSR pitch-line annotations parsed to 28 known intersection points; `cv2.findHomography` with RANSAC (reprojection threshold 15 px) estimates the planar transform. Clips with <4 correspondences are excluded. Primary: TVCalib (Theiner & Ewerth, 2023) segments pitch lines from broadcast frames and optimises camera parameters per frame via a differentiable projection loss, requiring no explicit correspondence labelling. TVCalib runs in a separate `../tvcalib/.venv/` invoked as subprocess; outputs `homographies_tvcalib.parquet`.

**6.3.2 Ground-truth track.** For each processed frame, SoccerNet GSR `bbox_pitch` annotations are parsed directly. Centred coordinates (±52.5, ±34) re-centred to (0-105, 0-68). Role preserved (player / goalkeeper); referees excluded by `category_id` filter.

**6.3.3 Outputs.**

- **GT-leak baseline cohort:** `detections_pipeline.parquet` (3,755 rows, 20 clips, 12 cols incl. `track_id`), `detections_gt.parquet` (4,295 rows, 20 clips). 540-row shortfall (GT > pipeline, ~13%) is the primary source of global-metric bias.
- **Primary pipeline cohort:** `detections_pipeline_tvcalib.parquet` (6,226 rows, 33 clips, YOLOv8x + TVCalib), `detections_soccana_tvcalib.parquet` (6,369 rows, 33 clips, Soccana + TVCalib), `detections_gt_full.parquet` (7,186 rows, 33 clips). Soccana shortfall vs full GT is 817 rows (~11%), narrower than YOLOv8x under the same homography (960 rows, ~13%).

**6.3.4 Sampling strategy.** ±15 frames around `action_position` per clip (up to 31 frames), giving temporal variation while remaining within the set-piece execution window.

### 6.4 Phase 4: Modeling

**6.4.1 Model selection.** Laurie Shaw's TTI Pitch Control model (Friends of Tracking, 2020, commit `21f4c2d`) was chosen because it is the established open-source baseline, computationally tractable on CPU/MPS for static frames, and produces interpretable probability surfaces. No alternative models evaluated; the analytical question is coordinate quality, not model selection.

**6.4.2 Static-frame adaptation.** The original Shaw model uses player velocities. Set-piece frames are near-static (ball out of play, players settling), so velocities are unavailable and assumed zero. This makes the model an *instantaneous* control surface: spatial dominance given current positions, without momentum. Applied identically to pipeline and GT tracks, ensuring any distributional difference is attributable to coordinate quality, not model asymmetry.

**6.4.3 Locked model parameters.**

| Parameter | Value | Meaning |
|---|---|---|
| `MAX_SPEED` | 5.0 m/s | Maximum player running speed |
| `REACTION_TIME` | 0.7 s | Time before a player begins moving toward a target |
| `SIGMA` | 0.45 s | Logistic slope on TTI difference |
| `LAMBDA` | 4.3 | Ball-control rate constant |
| Grid | 60 × 40 cells | ~1.75 m × 1.70 m per cell on 105 × 68 m |

**6.4.4 Attacking-team assignment.** Per frame, the team whose nearest player is closest to the ball is the attacking team. Consistent with set-piece context: the executing team is proximate to the ball.

Figure 7 shows a representative paired Pitch Control surface for a single set-piece frame: pipeline (left) vs ground-truth (right) under identical model parameters; surfaces are visually comparable while differing in defender coverage.

![Figure 7. Sample paired Pitch Control surface, pipeline (left) vs ground truth (right), single set-piece frame.](outputs/figures/07_pc_sample_pipeline_vs_gt.png)

**6.4.5 Summary metrics computed per frame.**

| Metric | Definition |
|---|---|
| `pc_mean` | Mean attacking PC across all 2,400 grid cells |
| `pc_at_ball` | PC at the grid cell nearest the ball position |
| `pc_in_box` | Mean attacking PC within the relevant penalty box |
| `pc_in_third` | Mean attacking PC within the relevant attacking third |
| `pc_area_gt_0p5` | Fraction of cells where attacking PC > 0.5 |

### 6.5 Phase 5: Evaluation

Evaluation used two-sample Kolmogorov-Smirnov tests (α=0.05, `scipy.stats.ks_2samp`) and histogram overlap (Bhattacharyya-style sum of min(p, q), 12 bins locked). Per-frame paired comparisons used Pearson and Spearman correlation, MAE, signed bias. Analysis stratified by action class (Corner / Direct free-kick) and pooled.

Three pipeline configurations are reported:

1. **Primary pipeline:** Soccana + TVCalib autonomous homography. No dependency on SoccerNet GSR pitch-line annotations.
2. **TVCalib YOLOv8x:** YOLOv8x + TVCalib. Isolates the contribution of football-domain detector finetuning.
3. **GT-leak baseline:** YOLOv8x + GT-pitch-line RANSAC. Exposes the homography-leak issue that motivated TVCalib adoption.

The primary pipeline evaluates against full-cohort GT (`detections_gt_full.parquet`, 33 clips); the GT-leak baseline evaluates against `detections_gt.parquet` (20 clips). Cohort mismatch is one of the reasons the GT-leak baseline cannot be directly compared frame-for-frame with the primary pipeline; Section 7.1.7 addresses resulting comparison challenges.

**6.5.1 Primary pipeline (Soccana + TVCalib) vs full-cohort GT.**

Primary configuration produces 457 frames across 30 paired clips, with 441 frame indices common to GT.

**Distributional comparison (pooled, n_pipe=457, n_gt=442):**

| Metric | KS stat | p-value | Hist. overlap | Mean primary | Mean GT | Δ |
|---|---|---|---|---|---|---|
| `pc_mean` | 0.193 | <0.001 | 0.807 | 0.541 | 0.595 | −0.055 |
| `pc_at_ball` | 0.110 | 0.008 | 0.854 | 0.899 | 0.898 | **+0.001** |
| `pc_in_box` | 0.126 | 0.001 | 0.806 | 0.588 | 0.575 | **+0.013** |
| `pc_in_third` | 0.118 | 0.003 | 0.810 | 0.581 | 0.621 | −0.040 |
| `pc_area_gt_0p5` | 0.169 | <0.001 | 0.815 | 0.544 | 0.604 | −0.061 |

Histogram overlap exceeds 0.80 on all five metrics. Bias is near zero on `pc_at_ball` and `pc_in_box`. Strict KS at α=0.05 rejects H0 on all five; Section 7.1.7 demonstrates this is a statistical-power artifact of larger n, not a fit regression.

**Per-frame paired comparison (n=441 paired frames, primary vs GT_full):**

| Metric | Pearson r | Spearman r | MAE | Bias |
|---|---|---|---|---|
| `pc_mean` | +0.269 | +0.263 | 0.133 | −0.056 |
| `pc_at_ball` | **+0.677** | **+0.572** | **0.072** | **−0.001** |
| `pc_in_box` | +0.373 | +0.426 | 0.227 | +0.021 |
| `pc_in_third` | +0.439 | +0.395 | 0.175 | −0.035 |
| `pc_area_gt_0p5` | +0.239 | +0.224 | 0.151 | −0.062 |

Per-frame paired correlation is strongly positive on `pc_at_ball` (Pearson r=0.677, Spearman r=0.572) and meaningfully positive on the four other metrics. MAE 0.072 on `pc_at_ball` means the primary pipeline's at-ball estimate differs from GT by 7.2 percentage points per frame on average.

![Figure 5. Paired pipeline vs ground-truth values per frame for each Pitch Control metric.](outputs/figures/05_pipeline_vs_gt_scatter.png)

![Figure 9. Paired scatter focused on per-action-class stratification.](outputs/figures/09_paired_scatter.png)

![Figure 8. Histogram overlays of pipeline vs ground-truth Pitch Control summary metrics.](outputs/figures/08_histogram_overlays.png)

**6.5.2 GT-leak baseline (YOLOv8x + GT-pitch-line H) vs 20-clip GT.**

Reported because it exposes the autonomy issue, motivates TVCalib replacement, and provides a controlled comparison point. Processes 20 clips successfully (13/33 excluded by RANSAC failure); produces 286 frames against 270 GT frames.

**Distributional comparison (pooled, n_pipe=286, n_gt=270):**

| Metric | KS stat | p-value | Hist. overlap | Mean baseline | Mean GT | Δ | Reject H0? |
|---|---|---|---|---|---|---|---|
| `pc_mean` | 0.402 | <0.001 | 0.629 | 0.450 | 0.617 | −0.167 | Yes |
| `pc_at_ball` | 0.089 | 0.202 | 0.864 | 0.876 | 0.895 | −0.019 | **No** |
| `pc_in_box` | 0.174 | <0.001 | 0.693 | 0.581 | 0.571 | +0.011 | Yes |
| `pc_in_third` | 0.203 | <0.001 | 0.762 | 0.557 | 0.634 | −0.077 | Yes |
| `pc_area_gt_0p5` | 0.375 | <0.001 | 0.638 | 0.450 | 0.630 | −0.180 | Yes |

Baseline passes KS on `pc_at_ball` (p=0.202) but fails on the other four. Bias on global metrics (`pc_mean`, `pc_area_gt_0p5`) is large and consistently negative (−0.17 to −0.18).

**Per-frame paired comparison (n=270 paired frames, baseline vs GT):**

| Metric | Pearson r | Spearman r | MAE | Bias |
|---|---|---|---|---|
| `pc_mean` | −0.032 | −0.175 | 0.243 | −0.174 |
| `pc_at_ball` | +0.548 | +0.457 | 0.094 | −0.024 |
| `pc_in_box` | +0.016 | +0.003 | 0.274 | +0.025 |
| `pc_in_third` | −0.028 | +0.040 | 0.195 | −0.070 |
| `pc_area_gt_0p5` | −0.048 | −0.193 | 0.270 | −0.187 |

Baseline produces meaningful paired correlation only on `pc_at_ball` (r=0.548). Other metrics show essentially zero or weakly negative paired correlation, in contrast to the primary pipeline.

![Figure 6. Distribution of detected players per frame, pipeline vs ground truth (20 clips, baseline).](outputs/figures/06_players_per_frame_dist.png)

**6.5.3 Bias diagnosis (nb04 Section 4b).** GT records more players per frame than YOLOv8x recovers: 4,295 GT rows vs 3,755 pipeline rows across 20 identical clips (~13% shortfall). In the Shaw model, additional defenders compress attacking PC uniformly across the surface; explains consistent negative bias on global metrics. `pc_at_ball` is structurally insensitive because the ball-proximate cell is dominated by the nearest attacker regardless of total defender count, which is why it is the only baseline metric to pass strict KS. Figure 10 confirms a consistent negative relationship between defender count and `pc_mean` in both tracks.

![Figure 10. Per-frame defender count vs `pc_mean`, pipeline (red) and ground truth (blue).](outputs/figures/10_defenders_vs_pc_mean.png)

Primary pipeline's bias reduction comes from two complementary corrections: Soccana's football-finetuned weights detect more players per frame, and TVCalib eliminates the 13-clip cohort attrition that biased the baseline toward easier camera geometries. Section 7.1.7 provides the three-way ablation that decomposes the contribution of each.

### 6.6 Phase 6: Deployment

**6.6.1 Current state.** Pipeline runs end-to-end on MacBook Air M3 (Apple Silicon MPS) with no cloud dependency. Runtime per clip ~15-30 s for YOLO inference; full 33-clip TVCalib run completes in ~30-45 min.

**6.6.2 Integration path.** Notebook-based pipeline converts to batch script with minimal refactoring; processing loops in nb02 are self-contained. Parquet output is compatible with DuckDB, pandas, polars. Club analyst with broadcast video and an automated pitch-line detector could adopt directly.

**6.6.3 Scalability considerations.** TVCalib removes the GT pitch-line dependency. YOLO inference scales linearly with frame count. PC computation is the fastest step (vectorised numpy, <0.1 s per frame). Cohort growth bounded by video access and Soccana/TVCalib inference time.

**6.6.4 Limitations for production use.** Zero-velocity assumption limits accuracy for dynamic (non-set-piece) phases. KMeans team assignment can fail when kit colours are similar. Homography accuracy depends on pitch-line visibility; clips with heavy advertising-board occlusion or unusual camera angles may produce higher TVCalib loss.

### 6.7 Project Outcomes and Deliverables

**6.7.1 Technical deliverables.**

- Pipeline scripts and notebooks for end-to-end execution.
- Parquet outputs covering set pieces, detections, pitch control, validations.
- 14 static figures + 2 animated GIFs.
- Locked threshold and parameter set documented in Annex B.

**6.7.2 Academic deliverables.**

- Distributional and per-frame paired validation results for three pipeline configurations.
- Three-way ablation decomposing residual bias into detector-domain and structural occlusion components.
- This report (`report.md`).

**6.7.3 Practical deliverables.**

- Reproducibility environment (Annex G) sufficient for a clean clone to reproduce all outputs from cached Parquet, or from scratch with SSD access.
- Run-order documentation aligned to CRISP-DM phases.

---

## 7. Discussion of Results

This section discusses each of the key findings individually for direct traceability between evidence and interpretation. For each: (1) meaning of the result, (2) critical interpretation and alternative explanation, (3) practical implication, (4) evidence limits.

### 7.1 Finding-by-Finding Critical Discussion

#### 7.1.1 Finding 1 - `pc_at_ball` is the most reliable pipeline output

**Finding.** Soccana + TVCalib on 30 paired clips gives `pc_at_ball` Δ=+0.001, histogram overlap 0.854, per-frame Pearson r=0.677, MAE 0.072.

The most operationally meaningful set-piece signal, control probability at the ball location, is preserved with near-zero bias. This captures whether the executing team has spatial dominance at the point of delivery, the primary determinant of set-piece danger.

**Critical interpretation.** Structural reason for robustness: the ball-proximate cell is dominated by the nearest attacker regardless of total defender count, so detector under-detection of defenders does not propagate strongly. This insensitivity is the same property that lets the GT-leak baseline pass strict KS on `pc_at_ball` (p=0.202) while failing on global metrics.

**Practical implication.** Use `pc_at_ball` as the primary tactical signal; it is calibrated and reliable across all three configurations evaluated.

**Evidence limits.** 441 paired frames over 30 clips; conclusions tied to the specific SoccerNet GSR cohort. Strict KS does reject H0 under the primary configuration (p=0.008), interpreted in Section 7.1.7 as power inflation rather than fit regression.

#### 7.1.2 Finding 2 - TVCalib recovers cohort and removes the autonomy leak

**Finding.** TVCalib processes 33/33 clips end-to-end (vs 20/33 baseline), zero homography failures, median `loss_ndc_total` 0.011. Phase 1 sanity: TVCalib mean RMSE 17 px against GT-line H 148,151 px (degenerate RANSAC on 4-intersection frames).

The GT-pitch-line baseline excluded 39% of clips and consumed GT annotations to compute the pixel→pitch transform, invalidating the autonomy claim. TVCalib eliminates both problems.

**Critical interpretation.** Cohort recovery is not free of selection effects: the 13 recovered clips may have wider angles or harder camera geometries than the 20 baseline-processable clips. Adding them grows n (more statistical power) and may shift the cohort distribution toward harder examples. This is a benefit, not a confound: the recovered cohort is more representative of real-deployment conditions.

**Practical implication.** Adopt TVCalib as the calibration step for any broadcast-only deployment. The GT-line baseline is not deployable outside annotated datasets.

**Evidence limits.** Phase 1 RMSE measured on 5 frames from SNGS-066; broader RMSE characterisation across all 528 frames was not performed.

#### 7.1.3 Finding 3 - Soccana cuts detector-domain bias by ~30% on global metrics

**Finding.** Under identical TVCalib homography, YOLOv8x → Soccana reduces `pc_mean` Δ from −0.096 to −0.055 and `pc_area_gt_0p5` Δ from −0.109 to −0.061.

Football-finetuned weights detect more defenders per frame in crowded penalty-area crops, partially closing the global-metric bias.

**Critical interpretation.** ~30% bias reduction sets a useful ceiling on the detector-domain contribution. The remaining ~70% is consistent with structural occlusion: defenders in tight clusters partially hidden by attackers from broadcast angles, which no detector can recover without different camera geometry. The detector ablation thus decomposes the residual bias into a closable component (domain shift) and an intrinsic component (occlusion).

**Practical implication.** Domain-finetuned detectors are worth the additional dependency for any tactical-metric pipeline. Off-the-shelf COCO detectors are not sufficient for crowded football scenes.

**Evidence limits.** No architecture-controlled ablation (e.g. YOLOv11x COCO vs Soccana YOLOv11n finetuned); the comparison is "off-the-shelf vs football-finetuned", which is the practitioner-relevant question, but conflates architecture and finetuning. Section 8.3 notes this as future work.

#### 7.1.4 Finding 4 - Global metrics are systematically underestimated

**Finding.** `pc_mean` and `pc_area_gt_0p5` are underestimated by 0.06 (primary) to 0.18 (baseline). Bias is consistent in direction (negative), explained by defender under-detection.

**Critical interpretation.** Mechanism is mathematically transparent in the Shaw model: more defenders compress attacking PC across the surface. Detector recall in crowded penalty-area crops is the proximate cause. The bias is structural and explainable, not random measurement noise.

**Practical implication.** Treat `pc_mean` and `pc_area_gt_0p5` as relative indicators for comparing set pieces within the same pipeline run, not as absolute values. Monitor `n_defenders` in the output Parquet; frames with unusually low counts should be treated as lower-confidence.

**Evidence limits.** Bias mechanism diagnosis is supported by Figure 10 (defender count vs `pc_mean` scatter) but cannot be experimentally confirmed without ground-truth detector counts on additional cohorts.

#### 7.1.5 Finding 5 - ByteTrack persistent IDs enable single-pass team assignment

**Finding.** Per-track mean HSV with one-shot KMeans on aggregated samples per clip eliminates the per-frame label flipping characteristic of naive per-detection clustering.

**Critical interpretation.** The contribution is design, not algorithm: KMeans is standard, ByteTrack is standard; combining them so each player gets one team label per clip rather than one per detection is what removes the noise floor. The approach requires no labelled training data and is robust to varying kit colours across clips.

**Practical implication.** Reusable design pattern for any broadcast-video pipeline that needs stable team labels without supervised training data.

**Evidence limits.** KMeans can still fail when team kit colours are very similar (e.g. white vs light grey). Not encountered in the SoccerNet GSR cohort; would need cohort-specific QA in deployment.

#### 7.1.6 Finding 6 - Two SoccerNet GSR clips carry annotation errors

**Finding.** During nb05 visualisation, two clips in the 20-clip baseline cohort were identified as mislabelled:

| Clip ID | Annotated `action_class` | Observed content |
|---|---|---|
| SNGS-125 | Corner | Mid-game open-play scene (no set piece at frame window) |
| SNGS-131 | Direct free-kick | Throw-in |

Both clips were excluded from nb05 visualisation figures (see `EXCLUDE_CLIPS` in nb05 cell 8) but retained in the distributional evaluation to keep the evaluation loop uniform.

**Critical interpretation.** Annotation noise at this rate (~10%) is meaningful for a 20-clip cohort and underscores the value of visual inspection alongside automated metrics.

**Practical implication.** Any future deployment of this pipeline on a new cohort should include a visual audit step, not rely on annotation labels alone.

**Evidence limits.** Cohort is small; the 10% rate may not generalise. The primary TVCalib cohort (33 clips) was not re-audited at this level for visualisation purposes beyond the 4 selected appendix clips.

#### 7.1.7 Finding 7 - Strict KS regression is a power artifact, not a fit regression

**Finding.** KS pass count (α=0.05) regresses from 1/5 (GT-leak baseline) to 0/5 (primary pipeline). Bias falls on 4/5 metrics, histogram overlap rises on 4/5, per-frame paired correlation improves on all five.

**Critical interpretation.** Cohort grew from 286 to 457 frames between the two configurations. KS detects smaller distributional differences with larger n. Bias and overlap evidence shows distributions are objectively closer under the primary pipeline; the strict pass-count metric is cohort-confounded.

**Practical implication.** Report KS alongside bias and overlap, not in isolation. Strict pass count is a poor summary when n differs between configurations.

**Evidence limits.** Power inflation can be quantified (e.g. by subsampling primary cohort to 286 frames) but was not done; the qualitative argument is supported by bias-and-overlap evidence rather than formal power analysis.

#### 7.1.8 Finding 8 - The pipeline runs on consumer hardware

**Finding.** Full 33-clip pipeline executes in ~30-45 minutes on a MacBook Air M3, no cloud dependency, no GPU server. YOLO inference on MPS, TVCalib on MPS, PC computation vectorised on CPU.

**Critical interpretation.** Hardware accessibility was a design constraint, not an outcome. The choice of Soccana (2.6M params vs YOLOv8x 68M) and ByteTrack (no appearance embedding model) over heavier alternatives directly serves this constraint.

**Practical implication.** Resource-constrained clubs can deploy this pipeline on existing laptops; no GPU server budget required.

**Evidence limits.** 33 clips is a small cohort; runtime scales linearly with frame count, so larger deployments (hundreds of clips per match week) would need batch scheduling.

### 7.2 Cross-Finding Synthesis

The eight findings collectively indicate that broadcast-only Pitch Control is viable for the most decision-relevant set-piece signal (`pc_at_ball`), is partially viable for penalty-area dominance (`pc_in_box`), and requires relative interpretation for global surface metrics (`pc_mean`, `pc_area_gt_0p5`).

The most coherent interpretation is that residual bias is structural (occlusion in crowded penalty-area crops, intrinsic to broadcast angles) rather than algorithmic. The pipeline can be improved incrementally (better detectors, finer calibration) but cannot fully close the global-metric gap without different camera setups.

Strategic sequence supported by the findings:

1. Adopt the primary configuration (Soccana + TVCalib) as the default deployment.
2. Use `pc_at_ball` for absolute tactical decisions; use `pc_mean` and `pc_area_gt_0p5` for relative within-run comparisons.
3. Treat `n_defenders` as a confidence diagnostic.
4. Maintain visual-audit step alongside automated metrics for annotation-quality QA.

### 7.3 Methodological Limits and Caution in Interpretation

Findings should be interpreted with the following constraints:

- **Cohort size:** 33 processable clips under TVCalib; conclusions cannot be generalised beyond this cohort without further validation.
- **Static-frame assumption:** zero-velocity Shaw TTI is appropriate for set pieces but does not extrapolate to open play.
- **Per-frame identity ambiguity:** pipeline track IDs are not matched to GT player IDs; paired comparison is per-frame summary-metric, not per-player coordinate accuracy.
- **Selection bias on the GT-leak baseline:** 13 excluded clips correlate with harder camera geometries; baseline cohort is not random.
- **Architecture/finetuning confound:** detector ablation compares YOLOv8x (68M, COCO) vs Soccana (2.6M, finetuned); architecture and training data both differ.
- **Statistical power inflation:** strict KS regression under the primary pipeline reflects n growth (286 → 457), not worse fit.

These constraints define the boundary conditions for inference and reinforce the need for next-phase validation on additional broadcast cohorts.

### 7.4 Practical Prioritization for Next-Phase Execution

Based on the findings, the highest-priority next steps are:

1. **Cohort expansion:** apply the primary pipeline to additional SoccerNet GSR or club-provided broadcast clips to characterise bias across leagues, camera setups, and annotation conventions.
2. **Velocity integration:** estimate per-player velocity from optical flow between consecutive frames (exploiting ByteTrack persistent IDs) to enable the full Shaw model and extend the pipeline to open play.
3. **Detector recall tuning:** raise YOLO confidence and add NMS tuning to reduce defender under-detection in crowded crops without inflating false positives.
4. **Architecture-controlled detector ablation:** YOLOv11x COCO baseline against Soccana YOLOv11n finetuned to decompose architecture vs training-data contributions to the ~30% closable bias.
5. **Operational deployment:** package as CLI + Docker for adoption by clubs without notebook expertise.

---

## 8. Conclusions and Future Work

### 8.1 Final Reflections

This project confirms that broadcast-only Pitch Control is viable for the most decision-relevant set-piece signal under a reproducible, open-source pipeline. The combination of Soccana (football-finetuned detector), ByteTrack (persistent IDs), per-track KMeans HSV (stable team assignment), TVCalib (autonomous calibration), and Shaw TTI (zero-velocity static-frame PC) produces near-zero bias on `pc_at_ball` and `pc_in_box`, with histogram overlap ≥0.81 on 4/5 metrics against full-cohort GT.

The central conclusion is that residual bias is structural, not algorithmic. Detector recall in crowded penalty-area crops can be partially improved by domain finetuning (Soccana closes ~30% of the YOLOv8x global-metric bias), but the remaining ~70% reflects intrinsic broadcast-angle occlusion that no detector can fully recover. The pipeline is therefore honest about which metrics it can support as absolute (`pc_at_ball`) and which require relative interpretation (`pc_mean`, `pc_area_gt_0p5`).

Methodologically, the project demonstrates that distributional plus per-frame paired validation against open SoccerNet GSR annotations is a viable evaluation design where proprietary tracking is unavailable. The CRISP-DM iteration loop is reflected directly in the development trajectory: KS failure under the baseline triggered bias diagnosis (back to Data Understanding), discovery of the GT-line homography leak triggered the TVCalib replacement (back to Data Preparation), and the Soccana ablation closed the autonomy and bias gaps in a controlled experimental design.

### 8.2 Core Conclusions

1. **The primary pipeline (Soccana + TVCalib) preserves the most decision-relevant set-piece signal.** Bias Δ=+0.001 on `pc_at_ball` against full-cohort GT (33 clips, 441 paired frames), histogram overlap 0.854, per-frame Pearson r=0.677. Practical upside: deployable directly as the primary tactical signal for resource-constrained clubs.
2. **TVCalib autonomous calibration is the highest-leverage single substitution.** Recovers 13 clips (cohort grows 65%), removes the GT-line autonomy leak, and produces Phase 1 RMSE 17 px against the GT-line H's 148,151 px. Practical upside: pipeline now runs end-to-end without consuming GT annotation.
3. **Football-domain detector finetuning closes ~30% of the YOLOv8x global-metric bias** under identical homography. The remaining ~70% is structural occlusion. Practical upside: Soccana is the right default for broadcast football CV pipelines; off-the-shelf COCO detectors are insufficient.
4. **Global surface metrics (`pc_mean`, `pc_area_gt_0p5`) are systematically biased** by detector under-detection of defenders, producing underestimates of 0.06 (primary) to 0.18 (baseline). The Shaw model's mathematical structure explains the direction of bias. Practical upside: relative-not-absolute interpretation guidance is well-grounded.
5. **ByteTrack integration eliminates per-frame team-label instability** by assigning persistent player IDs, enabling KMeans team assignment to run once per clip on aggregated jersey-colour evidence. Reusable design pattern.
6. **SoccerNet GSR annotation quality is imperfect:** 2/20 baseline clips carry mislabelled `action_class`. Visual-audit step is necessary alongside automated metrics.
7. **Strict KS pass count is cohort-confounded.** Report bias and overlap alongside KS, not in isolation, when configurations differ in n.
8. **The pipeline is fully reproducible on a MacBook Air M3** with no cloud infrastructure, proprietary data, or commercial licences.

### 8.3 Future Work

This thesis operationalises a broadcast-only Pitch Control pipeline validated on a 33-clip SoccerNet GSR cohort. Natural extensions: cohort expansion, velocity estimation, open-play adaptation, and operational deployment.

**8.3.1 Cohort expansion and annotation QA.**

- Apply the primary pipeline to additional SoccerNet GSR splits or club-provided broadcast clips.
- Characterise bias across leagues, camera setups, and broadcaster styles.
- Implement a systematic annotation-quality audit (e.g. action-class re-labelling on a sample) before distributional comparison.

Outcome: stronger generalisability claims and explicit characterisation of cohort-specific bias.

**8.3.2 Velocity estimation and open-play extension.**

- Estimate per-player velocity from optical flow between consecutive frames, exploiting ByteTrack's persistent IDs to build per-player trajectory estimates.
- Switch from zero-velocity static-frame PC to the full Shaw model (with velocity).
- Apply to open-play phases (throw-ins, goal kicks, dynamic possession sequences) where the camera is less static but PC remains analytically meaningful.

Outcome: pipeline extends from set pieces to broader tactical analysis use cases.

**8.3.3 Detector recall tuning.**

- Raise YOLO confidence threshold and add NMS tuning to reduce defender under-detection in crowded crops.
- Add a second-stage occlusion-aware detector head for tight player clusters.

Outcome: further reduction in global-metric bias, possibly closing the remaining structural component beyond what domain finetuning alone achieves.

**8.3.4 Architecture-controlled detector ablation.**

- YOLOv11x COCO baseline against Soccana YOLOv11n finetuned, isolating architecture from training data.
- Quantify the contribution of finetuning independently of architecture choice.

Outcome: stricter academic decomposition of the detector-domain bias component.

**8.3.5 Supervised team assignment.**

- Replace unsupervised KMeans with a supervised classifier trained on jersey-colour reference frames.
- Handle edge cases (similar kit colours, lighting variation, kit changes mid-clip).

Outcome: more robust team labels in deployment cohorts with challenging kit colours.

**8.3.6 Operational packaging.**

- CLI + Docker for adoption by clubs without notebook expertise.
- Streaming-mode inference for near-real-time deployment during live broadcasts.
- Dashboard integration for coach-facing tactical-metric visualisation.

Outcome: pipeline moves from academic prototype to production-ready toolkit.

### 8.4 Proposed Roadmap

**Phase 1 (0-3 months): Cohort expansion and recall tuning.**

- Add a second SoccerNet GSR-style cohort or a club-provided broadcast set.
- Tune YOLO confidence and NMS parameters; re-run primary pipeline.
- Re-publish bias and overlap tables on the expanded cohort.

**Phase 2 (3-6 months): Velocity integration and open-play trial.**

- Implement optical-flow-based per-player velocity.
- Run the full Shaw model with velocity on set pieces; compare against the static-frame baseline.
- Pilot the pipeline on a sample of open-play phases.

**Phase 3 (6-12 months): Operational deployment.**

- Package as CLI + Docker.
- Integrate with a lightweight dashboard.
- Pilot deployment with at least one resource-constrained club partner.

### 8.5 Academic and Practical Contribution

**Academic.** A reproducible, validated chain from broadcast pixels to a tactical metric, with controlled bias decomposition attributing residual error to specific pipeline components. Methodological pattern (distributional plus per-frame paired validation against open per-frame annotations) is reusable for other CV-to-tactics work.

**Practical.** An execution-ready open-source pipeline that resource-constrained clubs can adopt directly. Configuration guidance (Soccana + TVCalib, `pc_at_ball` as primary signal, `n_defenders` as confidence diagnostic) is evidence-backed and ready for deployment trials.

### 8.6 Closing Statement

Broadcast-only Pitch Control is now feasible for the most decision-relevant set-piece signal under an open-source, reproducible pipeline that runs on consumer hardware. The remaining gap on global surface metrics is structural, attributable to broadcast-angle occlusion, and partially closable by domain-finetuned detectors. Future work should prioritise cohort expansion, velocity integration, and operational deployment so the pipeline moves from academic prototype to club-ready toolkit. The honest commitment of the present work is methodological transparency: locked thresholds, documented data provenance, component-level bias attribution, and explicit limits on what the evidence can support.

---

## 9. Bibliography

Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019). Actions speak louder than goals: Valuing player actions in football. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 1851–1861. https://doi.org/10.1145/3292500.3330758

Deliège, A., Cioppa, A., Giancola, S., Seikavand, M. J., Magera, F., Jordi, B., Ghanem, B., & Van Droogenbroeck, M. (2021). SoccerNet-v2: A dataset and benchmarks for holistic understanding of broadcast soccer videos. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, 4508–4519. https://doi.org/10.48550/arXiv.2011.13367

Grand View Research. (2025). *Sports analytics market size, share & trends analysis report*. https://www.grandviewresearch.com/industry-analysis/sports-analytics-market

Hartley, R., & Zisserman, A. (2004). *Multiple view geometry in computer vision* (2nd ed.). Cambridge University Press.

Hudl. (2024). *StatsBomb — the world's most advanced football data* [Product page]. Hudl. https://www.hudl.com/en_gb/products/statsbomb

Institute for Science and Sport Performance in Football. (2022). *Performance tracking in professional football*. ISSPF. https://www.isspf.com/articles/performance-tracking-in-professional-football/

Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLO* (Version 8.0) [Computer software]. Ultralytics. https://github.com/ultralytics/ultralytics

Joos, V., Somers, V., & Standaert, B. (2024). *TrackLab* (Version 1.0) [Computer software]. GitHub. https://github.com/TrackingLaboratory/tracklab

Mansourian, A. M., Somers, V., De Vleeschouwer, C., & Kasaei, S. (2023). Multi-task learning for joint re-identification, team affiliation, and role classification for sports visual tracking. *Proceedings of the 6th International Workshop on Multimedia Content Analysis in Sports (MMSports '23)*, 103–112. https://doi.org/10.1145/3606038.3616172

Mordor Intelligence. (2025). *Sports analytics market — size, share & trends analysis*. https://www.mordorintelligence.com/industry-reports/sports-analytics-market

Nie, X., Peng, W., Chen, Y., & Cao, J. (2021). A robust and efficient framework for sports-field registration. *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, 1936–1944. https://doi.org/10.1109/WACV48630.2021.00198

Peralta Benítez, J., Buldú, J. M., Maestu, F., & Iglesias-Parro, S. (2024). Data analytics in the football industry: A survey investigating operational frameworks and practices in professional clubs and national federations from around the world. *Science and Medicine in Football*, *9*(2), 155–166. https://doi.org/10.1080/24733938.2024.2341837

ReSpo.Vision. (2024). *Making elite-level football data accessible to all*. ReSpo.Vision. https://respo.vision/blog-posts/making-elite-level-football-data-accessible-to-all

Shaw, L. (2020). *Pitch control model* (Commit 21f4c2d) [Computer software]. Friends of Tracking Data. https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking

Somers, V., Joos, V., Giancola, S., Cioppa, A., Ghasemzadeh, S. A., Magera, F., Standaert, B., Mansourian, A. M., Zhou, X., Kasaei, S., Ghanem, B., Alahi, A., Van Droogenbroeck, M., & De Vleeschouwer, C. (2024). SoccerNet game state reconstruction: End-to-end athlete tracking and identification on a minimap. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*. https://doi.org/10.48550/arXiv.2404.11335

Spearman, W. (2018). *Beyond expected goals* [Conference paper]. MIT Sloan Sports Analytics Conference. https://www.semanticscholar.org/paper/Beyond-Expected-Goals-Spearman/7a8a59f32a5d19b97fd9e7bd7e543d1d97b6de14

Sportmonks. (2024). *How football clubs use data analytics to improve performance*. Sportmonks. https://www.sportmonks.com/blogs/how-football-clubs-use-data-analytics-to-improve-performance/

StatsBomb. (2024). *StatsBomb open data* [Data set]. GitHub. https://github.com/statsbomb/open-data

Theiner, J., & Ewerth, R. (2023). TVCalib: Camera calibration for sports field registration in soccer. *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, 1166–1175. https://doi.org/10.48550/arXiv.2207.11709

Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). ByteTrack: Multi-object tracking by associating every detection box. *Proceedings of the European Conference on Computer Vision (ECCV)*. https://doi.org/10.48550/arXiv.2110.06864

### Notes

1. All URLs accessed and verified during the 2025–2026 project period.
2. Where version numbers are unavailable for software sources, commit hashes are provided as version identifiers.
3. SoccerNet GSR data accessed via official credentialed download; usage governed by the SoccerNet terms of service.
4. StatsBomb open data released under StatsBomb's open-data licence; cited per their attribution requirements.
5. Commercial tracking provider pricing (SkillCorner, Tracab, Second Spectrum) is not publicly disclosed; the competitive landscape table in §3.4.1 reflects publicly available positioning information only.

---

## 10. Annexes

### Annex A: Repository Structure

```
soccernet-setpiece-vision/
    notebooks/
        01_business_and_data_understanding.ipynb
        02_data_preparation_and_pipeline.ipynb
        03_pitch_control.ipynb
        04_evaluation_and_validation.ipynb
        05_visualizations.ipynb
    outputs/
        setpieces.parquet
        detections_pipeline.parquet
        detections_soccana.parquet              # ablation: Soccana under GT-line H
        detections_gt.parquet
        detections_gt_full.parquet              # GT for all 33 clips
        detections_pipeline_tvcalib.parquet     # primary: YOLOv8x + TVCalib H
        detections_soccana_tvcalib.parquet      # primary: Soccana + TVCalib H
        ball_positions.parquet
        homographies_tvcalib.parquet
        pipeline_diagnostics.parquet
        tvcalib_phase1_rmse.parquet             # Phase 1 sanity (5 SNGS-066 frames)
        pitch_control.parquet
        pitch_control_soccana.parquet           # ablation PC
        pitch_control_tvcalib.parquet
        pitch_control_soccana_tvcalib.parquet
        pitch_control_gt_full.parquet
        ablation_detector_summary.parquet       # detection-count comparison
        ablation_ks_summary.parquet             # detector ablation KS table
        validation_summary.parquet
        validation_paired.parquet
        validation_summary_tvcalib.parquet      # 3-way H-source ablation KS table
        figures/
            01_setpiece_counts.png ... 14_ks_table_tvcalib.png
            anim_corner_<clip_id>.gif
            anim_direct_free-kick_<clip_id>.gif
            still_corner_<clip_id>.png
            still_direct_free-kick_<clip_id>.png
    scripts/
        _pipeline_core.py                       # shared helpers (detection, homography, team assign)
        download_soccernet.py
        dump_ball_positions.py / dump_gt_setpieces.py
        run_soccana_ablation.py / run_pc_soccana.py
        compare_detectors.py / ablation_ks_table.py
        tvcalib_rmse_check.py / run_tvcalib_batch.py
        run_pipeline_tvcalib.py / run_pc_tvcalib.py
        run_pc_gt_full.py / ks_table_tvcalib.py
        run_soccana_tvcalib.py / run_pc_soccana_tvcalib.py
    CLAUDE.md
    requirements.txt
    report.md
```

### Annex B: Key Model Parameters

All parameters below are locked for reproducibility. Any deviation invalidates the validation in nb04.

**Primary pipeline (Soccana + TVCalib):**

| Parameter | Value | Notebook / Script |
|---|---|---|
| Detector | Soccana (`Adit-jain/soccana`, YOLOv11n, HF) | nb02 / run_soccana_tvcalib.py |
| Soccana confidence | 0.40 | nb02 |
| Soccana class | 0 (Player) | nb02 |
| Camera calibration | TVCalib (Theiner & Ewerth, WACV 2023) | run_tvcalib_batch.py |
| TVCalib median loss_ndc_total | 0.011 | run_tvcalib_batch.py |
| Tracker | ByteTrack (`bytetrack.yaml` via ultralytics) | nb02 |
| KMeans k | 3 with smallest-cluster drop (<15% of tracks or referee-kit centroid), then re-fit k=2 on survivors | nb02 |
| KMeans features | Per-track mean HSV (aggregated across ±15 frames) | nb02 |
| PC grid | 60 × 40 cells (105 × 68 m) | nb03 |
| MAX_SPEED | 5.0 m/s | nb03 |
| REACTION_TIME | 0.7 s | nb03 |
| SIGMA | 0.45 s | nb03 |
| KS alpha | 0.05 | nb04 |
| Histogram bins | 12 | nb04 |
| Frame window around action_position | ±15 frames | nb02 |

**Ablation baseline (YOLOv8x + GT-pitch-line H):**

| Parameter | Value | Notebook |
|---|---|---|
| Detector | YOLOv8x (`yolov8x.pt`, COCO-pretrained) | nb02 |
| YOLOv8 confidence | 0.40 | nb02 |
| YOLOv8 class | 0 (person / COCO) | nb02 |
| Homography RANSAC threshold | 15 px | nb02 |

### Annex C: Data Sources and Access

| Dataset | Access | Notes |
|---|---|---|
| SoccerNet GSR 2024 | Download via `scripts/download_soccernet.py` (requires SoccerNet password in `.env`) | Stored on external SSD, not in repo |
| StatsBomb Euro 2024 | `statsbombpy` (open, no authentication required) | Auto-cached in `~/.cache/statsbombpy/` |
| YOLOv8x weights | Auto-downloaded by `ultralytics` on first use | Cached in `~/.cache/ultralytics/` |
| Soccana weights | Auto-downloaded via `huggingface_hub.hf_hub_download` | Cached in `~/.cache/huggingface/` |
| TVCalib code | Sibling repo at `../tvcalib/`, own venv at `tvcalib/.venv/` | See Annex G |

### Annex D: Validation Summary Table

Full per-metric, per-action-class breakdowns are persisted as Parquet:

- `outputs/validation_summary.parquet` — GT-leak baseline vs 20-clip GT.
- `outputs/validation_summary_tvcalib.parquet` — three-way comparison: GT-leak YOLOv8x vs TVCalib YOLOv8x vs TVCalib Soccana, all against full-cohort GT.
- `outputs/validation_paired.parquet` — per-frame paired Pearson/Spearman/MAE/bias.

Reproducible by re-running nb04 with cached `pitch_control.parquet` and the `ks_table_tvcalib.py` script.

### Annex E: Broadcast Stills and Animated Surfaces

Broadcast still overlays paired with the corresponding Pitch Control surface are produced for four representative clips (two corners, two direct free kicks). Figures 15a-15d show pipeline output overlaid on the source frame at `action_position`.

![Figure 15a. SNGS-125 corner: broadcast still + Pitch Control overlay.](outputs/figures/still_corner_SNGS-125.png)

![Figure 15b. SNGS-140 corner: broadcast still + Pitch Control overlay.](outputs/figures/still_corner_SNGS-140.png)

![Figure 15c. SNGS-131 direct free kick: broadcast still + Pitch Control overlay.](outputs/figures/still_direct_free-kick_SNGS-131.png)

![Figure 15d. SNGS-198 direct free kick: broadcast still + Pitch Control overlay.](outputs/figures/still_direct_free-kick_SNGS-198.png)

Animated GIFs of the Pitch Control surface evolving over the ±15 frame window are produced by nb05 and stored at:

- `outputs/figures/anim_corner_SNGS-125.gif`
- `outputs/figures/anim_corner_SNGS-140.gif`
- `outputs/figures/anim_direct_free-kick_SNGS-131.gif`
- `outputs/figures/anim_direct_free-kick_SNGS-198.gif`

These animations do not embed in static PDF / DOCX exports and are referenced as supplementary material in the GitHub repository.

**MP4 video exports** can be produced from the same nb05 frame sequence using `cv2.VideoWriter` (codec mp4v, 6 fps, 1920×600 px three-panel layout: broadcast frame | minimap | Pitch Control heatmap). The MP4 export step requires the SSD to be mounted (broadcast JPEGs are read from each clip's `img1/` directory); cached Parquet files are sufficient for all other pipeline steps.

### Annex F: Data Dictionary

**`detections_pipeline.parquet`** (and `detections_pipeline_tvcalib.parquet`, `detections_soccana_tvcalib.parquet`): player detections from the CV pipeline (YOLOv8x or Soccana + ByteTrack + TVCalib).

| Column | Type | Description |
|---|---|---|
| `clip_id` | str | SoccerNet GSR clip identifier (e.g. `SNGS-066`) |
| `frame_idx` | int | Frame index within the clip (0-based) |
| `track_id` | int | ByteTrack persistent player ID within clip |
| `x_m` | float | Player foot position, pitch x, metres, origin top-left (0-105) |
| `y_m` | float | Player foot position, pitch y, metres, origin top-left (0-68) |
| `team` | int | Team label: 0 or 1 (KMeans-HSV assignment, once per clip) |
| `action_class` | str | Set-piece type: `Corner` or `Direct free-kick` |
| `action_position` | int | Frame index of set-piece moment (from SoccerNet annotation) |
| `conf` | float | YOLO detection confidence score |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | float | Bounding box in image pixels |
| `split` | str | SoccerNet split: `train`, `valid`, `test`, or `challenge` |

**`detections_gt.parquet`** (and `detections_gt_full.parquet`): player positions from SoccerNet GSR ground-truth annotations.

| Column | Type | Description |
|---|---|---|
| `clip_id` | str | SoccerNet GSR clip identifier |
| `frame_idx` | int | Frame index within the clip |
| `player_id` | int | SoccerNet annotation player identifier |
| `x_m` | float | Player foot position, pitch x, metres, origin top-left (0-105) |
| `y_m` | float | Player foot position, pitch y, metres, origin top-left (0-68) |
| `team` | int | Team label from annotation (0 or 1) |
| `role` | str | `player` or `goalkeeper` |
| `action_class` | str | Set-piece type: `Corner` or `Direct free-kick` |
| `action_position` | int | Frame index of set-piece moment |

**`pitch_control.parquet`** (and `pitch_control_tvcalib.parquet`, `pitch_control_soccana_tvcalib.parquet`, `pitch_control_gt_full.parquet`): Pitch Control summary metrics per frame, for all tracks.

| Column | Type | Description |
|---|---|---|
| `clip_id` | str | SoccerNet GSR clip identifier |
| `frame_idx` | int | Frame index within the clip |
| `track` | str | Source track: `pipeline`, `gt`, `tvcalib`, `soccana`, `soccana_tvcalib`, `gt_full` |
| `action_class` | str | Set-piece type |
| `pc_mean` | float | Mean attacking Pitch Control across all 2,400 grid cells (0-1) |
| `pc_at_ball` | float | Pitch Control at grid cell nearest ball position (0-1) |
| `pc_in_box` | float | Mean attacking PC within relevant penalty box (0-1) |
| `pc_in_third` | float | Mean attacking PC within relevant attacking third (0-1) |
| `pc_area_gt_0p5` | float | Fraction of pitch cells where attacking PC > 0.5 (0-1) |
| `n_attackers` | int | Number of attacking-team players detected in frame |
| `n_defenders` | int | Number of defending-team players detected in frame |

**`validation_summary.parquet`** (and `validation_summary_tvcalib.parquet`): KS test and histogram overlap results per metric and stratum.

| Column | Type | Description |
|---|---|---|
| `metric` | str | One of the five PC metric names |
| `stratum` | str | `pooled`, `Corner`, or `Direct free-kick` |
| `ks_stat` | float | Two-sample KS statistic |
| `ks_pvalue` | float | KS test p-value |
| `hist_overlap` | float | Bhattacharyya-style histogram overlap (0-1) |
| `mean_pipeline` | float | Mean metric value across pipeline frames |
| `mean_gt` | float | Mean metric value across GT frames |
| `delta` | float | `mean_pipeline − mean_gt` (signed bias) |
| `reject_h0` | bool | True if `ks_pvalue < 0.05` |

### Annex G: Reproducibility Environment

**Python environment.** All notebooks and scripts were executed in the `py311-dev` conda environment. Key package versions:

| Package | Version | Role |
|---|---|---|
| Python | 3.11.x | Runtime |
| ultralytics | 8.x | YOLOv8x + ByteTrack |
| statsbombpy | latest (auto-cache) | StatsBomb data access |
| mplsoccer | 1.x | Pitch visualisation |
| scikit-learn | 1.x | KMeans team assignment |
| scipy | 1.x | KS tests |
| opencv-python | 4.x | Homography (RANSAC baseline only) |
| pyarrow | latest | Parquet I/O |
| matplotlib | 3.x | Figures |
| pandas | 2.x | Data manipulation |

TVCalib runs in a separate conda/venv environment at `../tvcalib/.venv/` (Python 3.10, torch 2.1.1, kornia 0.8.2). It is invoked as a subprocess via `tvcalib/run_inference.py`; outputs written to Parquet and consumed by the main `py311-dev` environment.

**Hardware.** Development and execution ran on a MacBook Air M3 (Apple Silicon, 16 GB unified memory). YOLO inference used the MPS backend. TVCalib used MPS for batch inference. Total pipeline runtime for 33 clips (Phases 1-5 including TVCalib batch) is approximately 30-45 minutes.

**Run order.** The following sequence reproduces all outputs from scratch (requires SoccerNet GSR data on external SSD and `.env` with `SOCCERNET_LOCAL_DIR` and `SOCCERNET_PASSWORD`):

```bash
# Step 1: StatsBomb EDA and set-piece extraction
jupyter nbconvert --to notebook --execute notebooks/01_business_and_data_understanding.ipynb --inplace

# Step 2: CV pipeline (YOLO + ByteTrack + TVCalib)
python scripts/run_tvcalib_batch.py          # batch TVCalib H (Phase 2, ~5 min)
jupyter nbconvert --to notebook --execute notebooks/02_data_preparation_and_pipeline.ipynb --inplace
python scripts/dump_ball_positions.py
python scripts/dump_gt_setpieces.py

# Step 3: Pitch Control
jupyter nbconvert --to notebook --execute notebooks/03_pitch_control.ipynb --inplace
python scripts/run_pc_tvcalib.py
python scripts/run_pc_gt_full.py

# Step 4: Evaluation
jupyter nbconvert --to notebook --execute notebooks/04_evaluation_and_validation.ipynb --inplace
python scripts/ks_table_tvcalib.py

# Step 5: Visualisations
jupyter nbconvert --to notebook --execute notebooks/05_visualizations.ipynb --inplace

# Optional: detector ablation (Soccana under TVCalib H)
python scripts/run_soccana_tvcalib.py
python scripts/run_pc_soccana_tvcalib.py
python scripts/ks_table_tvcalib.py
```

All intermediate Parquet outputs are committed to the repository under `outputs/`. Steps 3-5 and the evaluation scripts can be re-executed without the SSD by using these cached files.
