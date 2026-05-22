# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# soccernet-setpiece-vision — Master's Final Project

## Candidate
Moritz Philipp Haaf, MSc AI Applied to Sports, Sports Data Campus
Deadline: 30 June 2026

## Data Storage
External SSD: `/Volumes/MPH-ExternalStorage/soccernet-gsr` (Mac) — path on Windows will differ.
All SoccerNet GSR video clips and annotations stored here.
Never store video data inside the project repo.

## Goal
Computer Vision pipeline for set-piece analysis in football.
Derive Pitch Control from SoccerNet GSR broadcast video.
Validate distributionally against StatsBomb 360 Euro 2024.

## Five Notebooks (CRISP-DM) — all run locally on Mac
1. notebooks/01_business_and_data_understanding.ipynb
2. notebooks/02_data_preparation_and_pipeline.ipynb
3. notebooks/03_pitch_control.ipynb
4. notebooks/04_evaluation_and_validation.ipynb
5. notebooks/05_visualizations.ipynb

## Stack (keep it simple)
- Python 3.11, conda env py311-dev
- No Poetry, no CI, no pre-commit (project rule overrides global CLAUDE.md; `pre-commit` may appear in `pip freeze` from global env but is not used here)
- `requirements.txt` is full `pip freeze` of `py311-dev`, not curated. Treat as lock snapshot, not source of truth.
- statsbombpy: Euro 2024 (competition_id=55, season_id=282)
- YOLOv8x (ultralytics): player detection on SoccerNet GSR frames
- Soccana (HuggingFace `Adit-jain/soccana`, YOLOv11n football-finetuned): ablation detector for section 6
- ByteTrack (ultralytics built-in): persistent player IDs across frames
  - Requires `lap` package; `pip install lap` if `yolo.track()` fails on import
- OpenCV: RANSAC homography (legacy GT-leak baseline only)
- TVCalib (Theiner & Ewerth, WACV 2023, MM4SPA/tvcalib): autonomous camera calibration replacing GT pitch lines. Sibling dir `../tvcalib/` with own venv at `tvcalib/.venv/` (torch 2.11, kornia 0.8.2). Patches applied for torch 2.x: `tvcalib/sncalib_dataset.py:13` (`string_classes`), `tvcalib/inference.py:89` (`weights_only=False`).
- KMeans (scikit-learn): two-pass team assignment on per-track mean HSV jersey colour (once per clip)
- Pitch Control: Laurie Shaw Friends of Tracking implementation
- mplsoccer: all pitch visualisations
- Parquet: all intermediate outputs saved to outputs/

## Coordinate Systems

Three systems are in play — always be explicit about which one you are in:

| System | Convention |
|---|---|
| StatsBomb | 120 yards × 80 yards, origin top-left |
| Pipeline / mplsoccer | 105 m × 68 m, origin top-left |
| SoccerNet GSR `bbox_pitch` | centred origin (±52.5 m, ±34 m) |

StatsBomb → Pipeline: `x_m = x_sb * (105/120)`, `y_m = y_sb * (68/80)`
GSR centred → Pipeline: `x_m = x_gsr + 52.5`, `y_m = y_gsr + 34`

## Validation Strategy
Distributional, not frame-matched.
SoccerNet GSR clips != StatsBomb Euro 2024 matches.
Compare Pitch Control distributions across comparable set-piece types.

## Key Paths
| Resource | Path |
|---|---|
| Notebooks | ./notebooks |
| Outputs (Parquet) | ./outputs |
| Figures | ./outputs/figures |
| Scripts | ./scripts |
| Thesis source | ./report.md |
| SoccerNet data (Mac) | /Volumes/MPH-ExternalStorage/soccernet-gsr |

SSD root resolved via `SOCCERNET_LOCAL_DIR` env var (fallback = Mac path above). Set in `.env` on Windows clone to point at local mount; all scripts and nb02/nb03/nb05 honour it.

## Repo State (as of 2026-05-06)
All five notebooks exist and have been executed. Outputs directory is populated. Pipeline is functionally complete through validation.

**Notebooks:**
- `notebooks/01_business_and_data_understanding.ipynb` — complete
- `notebooks/02_data_preparation_and_pipeline.ipynb` — complete
- `notebooks/03_pitch_control.ipynb` — complete
- `notebooks/04_evaluation_and_validation.ipynb` — complete (incl. section 6 detector ablation)
- `notebooks/05_visualizations.ipynb` — complete (animated PC + minimap + broadcast stills)

**Scripts:**
- `scripts/download_soccernet.py` — SoccerNet GSR download (idempotent, needs SSD mounted)
- `scripts/dump_ball_positions.py` — reads `Labels-GameState.json` from the SSD and writes `outputs/ball_positions.parquet`; run this once after nb02 so that nb03/nb04 can execute offline without the SSD
- `scripts/run_soccana_ablation.py` — re-runs nb02 detection stage with Soccana (HF: `Adit-jain/soccana`) holding all other stages constant; writes `outputs/detections_soccana.parquet`. Needs SSD.
- `scripts/run_pc_soccana.py` — mirrors nb03 PC model on Soccana detections; writes `outputs/pitch_control_soccana.parquet`. SSD-free.
- `scripts/compare_detectors.py` — detection-count ablation (YOLOv8x vs Soccana vs GT); writes `outputs/ablation_detector_summary.parquet` + figure 11.
- `scripts/ablation_ks_table.py` — standalone KS table for the PC ablation; writes `outputs/ablation_ks_summary.parquet` + figure 13.
- `scripts/repair_setpieces_freeze_frames.py` — re-fetches StatsBomb 360 freeze frames per match if nb01 produced 0% coverage. Use only if `outputs/setpieces.parquet` shows empty `freeze_frame` arrays.
- `scripts/tvcalib_rmse_check.py` — Phase 1 sanity: project GT player `bbox_pitch` foot points through TVCalib H vs GT-pitch-line H, measure pixel RMSE on 5 SNGS-066 frames. Writes `outputs/tvcalib_phase1_rmse.parquet`. Needs SSD + `tvcalib/.venv/` set up.
- `scripts/run_tvcalib_batch.py` — Phase 2 batch: stage all set-piece frames (33 clips × 16 = 528) into `/tmp/tvcalib_batch/`, invoke `tvcalib/run_inference.py` once, parse `calib.json` -> `outputs/homographies_tvcalib.parquet`. Idempotent; reuses `/tmp/tvcalib_batch_out/calib.json` if present.
- `scripts/run_pipeline_tvcalib.py` — Phase 3 pipeline run: mirrors nb02 cell 8 with TVCalib H lookup replacing `homography_from_pitch_lines`. Writes `outputs/detections_pipeline_tvcalib.parquet`. Needs SSD.
- `scripts/run_pc_tvcalib.py` — PC model on TVCalib detections; writes `outputs/pitch_control_tvcalib.parquet` (track='tvcalib'). SSD-free.
- `scripts/dump_gt_setpieces.py` — extracts GT player positions for ALL 33 clips' set-piece windows (not the 20-clip subset that survived GT-line homography in nb02). Writes `outputs/detections_gt_full.parquet`. Needs SSD.
- `scripts/run_pc_gt_full.py` — PC over `detections_gt_full.parquet`; writes `outputs/pitch_control_gt_full.parquet`. SSD-free.
- `scripts/ks_table_tvcalib.py` — KS comparison: GT-leak YOLOv8x vs TVCalib YOLOv8x vs TVCalib Soccana, all against full-cohort GT. Writes `outputs/validation_summary_tvcalib.parquet` and `outputs/figures/14_ks_table_tvcalib.png`.
- `scripts/run_soccana_tvcalib.py` — Soccana detector under TVCalib H (pure ablation isolation). Writes `outputs/detections_soccana_tvcalib.parquet`. Needs SSD.
- `scripts/run_pc_soccana_tvcalib.py` — PC for Soccana+TVCalib detections; writes `outputs/pitch_control_soccana_tvcalib.parquet`. SSD-free.
- `scripts/render_annotated_clips.py` — renders team-colored player bbox overlays on broadcast frames; writes MP4s to `outputs/figures/annotated/`. Reads `x1_px/y1_px/x2_px/y2_px` columns from a detection parquet (produced by `run_pipeline_tvcalib.py` or `run_soccana_tvcalib.py`). Accepts `--clip SNGS-066` and `--detector soccana` flags. Needs SSD.

**Outputs (all Parquet):**
- `outputs/ball_positions.parquet`
- `outputs/detections_gt.parquet`
- `outputs/detections_pipeline.parquet`
- `outputs/detections_soccana.parquet`             # ablation detections
- `outputs/pipeline_diagnostics.parquet`
- `outputs/pitch_control.parquet`
- `outputs/pitch_control_soccana.parquet`          # ablation PC surfaces
- `outputs/setpieces.parquet`
- `outputs/validation_paired.parquet`
- `outputs/validation_summary.parquet`
- `outputs/ablation_detector_summary.parquet`      # detection-count comparison
- `outputs/ablation_ks_summary.parquet`            # PC KS table per detector
- `outputs/homographies_tvcalib.parquet`           # TVCalib H per (split, clip, frame), 528 rows × 33 clips
- `outputs/detections_pipeline_tvcalib.parquet`    # YOLOv8x under TVCalib H, 6226 rows × 33 clips
- `outputs/detections_soccana_tvcalib.parquet`     # Soccana under TVCalib H, 6369 rows × 33 clips
- `outputs/detections_gt_full.parquet`             # GT for all 33 clips (not the GT-leak 20)
- `outputs/pitch_control_tvcalib.parquet`          # PC for YOLOv8x+TVCalib
- `outputs/pitch_control_soccana_tvcalib.parquet`  # PC for Soccana+TVCalib
- `outputs/pitch_control_gt_full.parquet`          # GT PC over 33-clip cohort
- `outputs/validation_summary_tvcalib.parquet`     # KS table: 3-way H-source ablation
- `outputs/tvcalib_phase1_rmse.parquet`            # Phase 1 sanity result (5 frames, SNGS-066)

**Figures** (`outputs/figures/`): 14 static PNGs + 2 animated GIFs (corner, direct free-kick). Figures 11-13 cover the detector ablation (count distributions, histogram overlays, KS table). Figure 14 covers the TVCalib H-source ablation (3-way KS table: GT-leak baseline vs TVCalib YOLOv8x vs TVCalib Soccana).

**Clip cohort:**
- GT-leak baseline: 33 clips → 20 end-to-end (18 paired with GT in PC validation), 13 excluded by homography failure.
- TVCalib autonomous: 33/33 clips processed end-to-end (zero homography failures); 30 paired with full-cohort GT in PC. **+13 clips recovered.**

**Thesis source:** `report.md` — Markdown with LaTeX/pandoc front-matter, renders to PDF via `pandoc report.md -o report.pdf`.

**Key validated results (for thesis context):**

GT-leak baseline (20 clips): `pc_at_ball` passes KS at pooled level (p=0.202, hist overlap 0.864); global metrics (`pc_mean`, `pc_area_gt_0p5`) underestimate by ~−0.17 to −0.19. Soccana ablation under same H reduces global-metric bias ~30%.

TVCalib autonomous H (33 clips, full cohort vs `gt_full`):

| metric | GT-leak YOLOv8x Δ | TVCalib YOLOv8x Δ | TVCalib Soccana Δ | overlap GT-leak→TV-Soccana |
|---|---|---|---|---|
| pc_mean | −0.167 | −0.096 | −0.055 | 0.629 → 0.807 |
| pc_at_ball | −0.019 | −0.004 | **+0.001** | 0.864 → 0.854 |
| pc_in_box | +0.011 | +0.100 | **+0.013** | 0.693 → 0.806 |
| pc_in_third | −0.077 | +0.012 | −0.040 | 0.762 → 0.810 |
| pc_area_gt_0p5 | −0.181 | −0.109 | −0.061 | 0.638 → 0.815 |

KS pass count: GT-leak 1/5, TVCalib YOLOv8x 0/5, TVCalib Soccana 0/5. **Strict KS regresses because n grew (286 → 457 frames) and small effects become statistically detectable, but bias and overlap improve everywhere.** Best end-to-end combination = TVCalib + Soccana: bias near-zero on `pc_at_ball` and `pc_in_box`, hist overlap ≥0.81 on 4/5 metrics. The narrative is autonomy + cohort recovery + bias reduction; honest reporting of KS power inflation.

Phase 1 RMSE sanity (5 SNGS-066 frames): TVCalib mean 16.99 px, GT-pitch-line H mean 148,151 px (degenerate on these frames — only 4 intersections each, RANSAC unstable). Decisive gate pass.

**Next milestone:** written thesis/report ahead of 30 June 2026 deadline.

`scripts/download_soccernet.py` downloads task `gamestate-2024`, splits `[train, valid, test, challenge]`. Idempotent: skips splits already on disk (zip >0B or extracted dir non-empty). Honors `SOCCERNET_LOCAL_DIR` env var; defaults to SSD path.

## Commands

```bash
# Activate env
conda activate py311-dev

# Launch JupyterLab
jupyter lab

# Run a notebook non-interactively
jupyter nbconvert --to notebook --execute notebooks/01_business_and_data_understanding.ipynb --inplace

# Lint / format
ruff check .
black .

# Download SoccerNet GSR data (idempotent, needs .env with SOCCERNET_PASSWORD)
python scripts/download_soccernet.py

# Detector ablation pipeline (after nb02 has run)
python scripts/run_soccana_ablation.py
python scripts/run_pc_soccana.py
python scripts/ablation_ks_table.py

# TVCalib autonomous H pipeline (replaces GT-pitch-line homography leak)
# Phase 1: RMSE sanity on 5 frames (~1 min)
python scripts/tvcalib_rmse_check.py

# Phase 2: batch H over all 33 set-piece clips' frames (~5 min on MPS)
python scripts/run_tvcalib_batch.py

# Phase 3: pipeline detection under TVCalib H (~10-15 min, SSD)
python scripts/run_pipeline_tvcalib.py
python scripts/dump_ball_positions.py     # picks up new TVCalib clip frames
python scripts/dump_gt_setpieces.py       # GT for all 33 clips

# Phase 4: PC + KS comparison
python scripts/run_pc_tvcalib.py
python scripts/run_pc_gt_full.py
python scripts/ks_table_tvcalib.py

# Phase 5 (optional): Soccana under TVCalib H — best end-to-end combo
python scripts/run_soccana_tvcalib.py
python scripts/run_pc_soccana_tvcalib.py
python scripts/ks_table_tvcalib.py        # auto-includes Soccana+TVCalib row

# Render annotated broadcast clips (MP4, needs SSD)
python scripts/render_annotated_clips.py                     # all clips, yolov8x detections
python scripts/render_annotated_clips.py --clip SNGS-066     # single clip
python scripts/render_annotated_clips.py --detector soccana  # soccana detections

# Register py311-dev as Jupyter kernel (one-time, after fresh env)
python -m ipykernel install --user --name py311-dev --display-name "Python (py311-dev)"
```

## External Caches
- StatsBomb: `~/.cache/statsbombpy/` (statsbombpy auto-cache, offline-friendly)
- YOLOv8 weights: `~/.cache/ultralytics/` (auto-download on first use; pin checkpoint name in notebook 02)
- `notebooks/yolov8x.pt` and `yolov8x.pt` (project root) — local weight copies checked in for offline use; ultralytics also resolves from `~/.cache/ultralytics/` automatically
- HuggingFace (if used): `~/.cache/huggingface/`

## Editable Packages
- `setpiece-pipeline` — this repo itself, installed as `setpiece-pipeline==0.1.0` (`pip install -e .`)
- `broadcast-to-tactics` — sibling repo at `/Users/mph/Dev/itzmore-mph/MAIS-projects/final-master-project/broadcast-to-tactics` (Mac path). If imports from `broadcast_to_tactics` fail: `pip install -e ../broadcast-to-tactics`.

## Pipeline Architecture
Each notebook is a self-contained CRISP-DM phase:

1. **01_business_and_data_understanding** — StatsBomb Euro 2024 EDA; set-piece type distribution; establish validation benchmarks.
2. **02_data_preparation_and_pipeline** — SoccerNet GSR frame extraction (OpenCV), YOLOv8x player detection, ByteTrack for persistent IDs, two-pass KMeans team assignment on per-track mean HSV (once per clip), RANSAC homography to pitch coordinates, output Parquet to `outputs/`.
3. **03_pitch_control** — Load Parquet detections, run Laurie Shaw pitch control model, aggregate control surfaces per set-piece type, save results to `outputs/`.
4. **04_evaluation_and_validation** — Distributional comparison (KS test / histogram overlap) of pipeline pitch control vs StatsBomb 360 Euro 2024 ground truth, plus bias diagnostics.
5. **05_visualizations** — Animated pitch-control GIFs, broadcast stills overlay, minimap renders. All mplsoccer outputs land in `outputs/figures/`.

## Detector Ablation
A second detection pass swaps YOLOv8x (COCO-pretrained) for Soccana (`Adit-jain/soccana`, YOLOv11n finetuned on SoccerNet + match footage). All other pipeline stages are held constant: ByteTrack, KMeans-HSV teams, RANSAC homography, Laurie Shaw PC model, and locked nb04 thresholds. Soccana weights live at `Model/weights/best.pt` inside the HF repo and are fetched via `huggingface_hub.hf_hub_download` (cached at `~/.cache/huggingface/`). Soccana class IDs: 0=Player, 1=Ball, 2=Referee — pipeline filters to class 0 only.

Run order:
1. `python scripts/run_soccana_ablation.py`   (~30-60 min, SSD required)
2. `python scripts/run_pc_soccana.py`         (seconds)
3. `python scripts/ablation_ks_table.py`      (or re-execute nb04 section 6)

## H-Source Ablation (TVCalib)
Replaces GT-pitch-line homography (`homography_from_pitch_lines` in nb02 cell 7) with autonomous TVCalib (Theiner & Ewerth, WACV 2023, MM4SPA/tvcalib). The original GT-line approach was a methodological leak: it used SoccerNet GSR ground-truth pitch annotations to compute the pixel→pitch transform, invalidating the autonomy claim of the proposal.

TVCalib produces world (centred metres ±52.5/±34) → image (1920×1080 pixels) homography per frame via segmentation + per-frame self-supervised camera optimisation. To match nb02's expected image→pitch-topleft H: `H_image_to_topleft = T_centred_to_topleft @ inv(H_world_to_image_centred)` where `T = [[1,0,52.5],[0,1,34],[0,0,1]]`.

Phase results:
- Phase 1 sanity (5 SNGS-066 frames): TVCalib mean 17 px, GT-line H 148 K px (4-intersection degenerate). Decisive pass.
- Phase 2 batch: 528 H over 33/33 clips × 16 frames each. Median `loss_ndc_total` = 0.011.
- Phase 3 pipeline: 33/33 clips processed (vs 20/33 baseline). 6226 detection rows (vs 3755). Zero homography failures.
- Phase 4 KS: bias improved on 4/5 metrics, hist overlap improved on 4/5; KS strict pass count drops 1→0 due to n=457 vs 286 (more power detects smaller effects).
- Phase 5 Soccana+TVCalib: best end-to-end. Bias near-zero on `pc_at_ball` (Δ=+0.001) and `pc_in_box` (Δ=+0.013).

Run order:
1. `python scripts/tvcalib_rmse_check.py`     (Phase 1, ~1 min, SSD)
2. `python scripts/run_tvcalib_batch.py`      (Phase 2, ~5 min on MPS, SSD)
3. `python scripts/run_pipeline_tvcalib.py`   (Phase 3, ~10-15 min, SSD)
4. `python scripts/dump_ball_positions.py && python scripts/dump_gt_setpieces.py`
5. `python scripts/run_pc_tvcalib.py && python scripts/run_pc_gt_full.py`
6. `python scripts/ks_table_tvcalib.py`       (Phase 4 KS comparison)
7. `python scripts/run_soccana_tvcalib.py && python scripts/run_pc_soccana_tvcalib.py` (Phase 5)
8. `python scripts/ks_table_tvcalib.py`       (auto-detects Soccana+TVCalib parquet)

TVCalib repo lives at `../tvcalib/` (sibling dir, separate venv). Smoke-test wrapper is `tvcalib/run_inference.py`. See memory `project_tvcalib_resume.md` for setup details.

## Detector Choice (settled 2026-05-06)

**Primary detector:** Soccana (YOLOv11n, football-finetuned via SoccerNet GSR + match footage). Best end-to-end results under TVCalib H: bias near zero on `pc_at_ball` and `pc_in_box`, hist overlap ≥0.81 on 4/5 metrics.

**Baseline ablation:** YOLOv8x (COCO-pretrained). Kept as off-the-shelf comparison to quantify the value of football-domain finetuning. Bias decomposition shows ~30% detector-domain / ~70% structural occlusion contribution.

**Rejected:** Three-way YOLOv11x arm. Would isolate architecture vs finetuning variables, but: (a) Soccana already wins decisively, (b) v11n is nano-size (2.6M params) — smaller model than v8x (68M), so v11n COCO probably worse-recall baseline than v8x COCO, (c) thesis deadline (2026-06-30) better spent writing than re-running. Frame current ablation as "off-the-shelf vs football-finetuned" — practical deployment question, examiner-defendable.

## Key Design Decisions
- **No frame-level ground truth:** validation is distributional only. Never claim per-frame accuracy.
- **Coordinate systems:** StatsBomb is yards (120x80); pipeline works in metres (105x68). Conversion is `x_m = x_sb * (105/120)`, `y_m = y_sb * (68/80)`. Apply before any geometry.
- **Team assignment:** KMeans on HSV values of bounding-box crops, not jersey numbers. Works for broadcast clips where numbers aren't reliably resolved.
- **outputs/ discipline:** only `.parquet` files land here. Any intermediate frame images or video go to the external SSD or `/tmp`.
- **Pitch Control source:** vendor or pin Laurie Shaw's "Friends of Tracking" repo + commit hash in notebook 03 to lock implementation behaviour.
- **Validation thresholds:** lock KS test alpha + histogram bin count in notebook 04 for reproducibility.

## Rules
- Everything inline in notebooks, no separate src/ modules
- Use `py311-dev` conda env exclusively. Do NOT use `.venv` — observed kernel/version drift between the two breaks ultralytics weight loading on PyTorch ≥2.6.
- On Mac: verify `which python` resolves to `/Users/mph/miniconda3/envs/py311-dev/bin/python` before running scripts.
- All paths relative to project root except SoccerNet data on external SSD
- .env for secrets (SOCCERNET_PASSWORD), never hardcoded
- Each notebook runs independently
- All five notebooks run locally on MacBook Air M3; repo is also cloned on Windows (c:\Users\morhaaf\Dev\Git\) but execution is Mac-primary
- outputs/ contains only Parquet files, never raw video
- Never hardcode `Path("/Volumes/...")`. SSD root must be `Path(os.getenv("SOCCERNET_LOCAL_DIR", "/Volumes/MPH-ExternalStorage/soccernet-gsr")) / "gamestate-2024"` so the Windows clone stays runnable via `.env`.
- Notebook print statements display paths via `Path.relative_to(PROJECT_ROOT)`. Keeps committed cell outputs portable and clean of `/Users/mph/...` leaks.