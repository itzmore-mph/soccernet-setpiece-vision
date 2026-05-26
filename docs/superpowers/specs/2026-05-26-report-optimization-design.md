# Report Optimization Design — `report.md`

**Date:** 2026-05-26
**Owner:** Moritz Philipp Haaf
**Target deliverable:** Submission-ready MSc Final Project report (Sports Data Campus, Module 9)

## Goal

Bring `report.md` into full alignment with the official Module 9 documentation structure and academic conventions, optimised for an excellent grade. The report is already substantive and well-written; this pass is about closing structural gaps, embedding visual evidence, and tightening academic polish.

## Scope (confirmed with user)

- Full pass: structural + polish
- Embed all key figures inline with numbered captions
- No strict length limit; quality first
- Timeline: illustrative (Week 1–N), not calendar-dated
- Bibliography: APA 7 consistency pass (current style retained, formatting normalised, URLs/DOIs added where missing)
- Delivery: single commit at the end

## Gap analysis vs Sports Data Campus rubric

| Rubric requirement | Current state | Action |
|---|---|---|
| 1. Executive Summary | Present, strong | Polish: stronger opening sentence |
| 2. Introduction | Present, strong | No change |
| 3. Objectives | Present, well-structured | No change |
| 4. Timeline / Project Planning | Missing | Insert new Section 4 |
| 5. Conceptual + Technological Architecture | Present as §4 | Renumber to §5 |
| 6. Methodologies and Techniques (CRISP-DM) | Present as §5 | Renumber to §6 |
| 7. Project Development | Present as §6 | Renumber to §7 |
| 8. Results Discussion | Present as §7 | Renumber to §8 |
| 9. Conclusions and Future Work | Present as §8 | Renumber to §9 |
| 10. Bibliography | Present as §9 | Renumber to §10, APA polish |
| 11. Appendices | Present as §10 (Annexes) | Renumber to §11 |

## Section-by-section changes

### Section 1 (Executive Summary)
- Open with a one-sentence problem+solution lead before any metrics.
- Keep all factual content as-is.
- Sweep em-dashes (replace with commas or restructure per user feedback memory).

### Section 4 (NEW) — Timeline
Insert between Objectives (§3) and Architecture (§5). Content:
- Brief intro paragraph framing project planning approach.
- Illustrative phase table mapping CRISP-DM phases to weeks (W1–W16).
- Key milestones list including the `action_position` discovery and fix.
- Constraints (data access, hardware, mid-project pivots).
- Dependencies (TVCalib → detections → PC → validation).

### Section 5 (was §4) — Architecture
- Add Table N captions to all tables.
- Embed Figure 6 (`11_multiclass_detections.png`) showing Soccana multiclass output.
- Otherwise no content change.

### Section 6 (was §5) — Methodology
- Add intro paragraph to §6.2 (currently jumps into the phase table).
- Otherwise no content change.

### Section 7 (was §6) — Work Development
- Embed Figures 1, 2, 4, 5 in §7.1 / §7.2 (Business + Data Understanding).
- Embed Figure 3 (`03_players_per_frame.png`) in §7.3 (Data Preparation).
- Embed Figures 7, 8, 9, 11 in §7.5 (Evaluation).
- Embed Figures 12, 13 (three-panel stills) in §7.6 (Deployment).
- All tables get Table N captions.

### Section 8 (was §7) — Discussion
- Embed Figure 10 (`10_defenders_vs_pc_mean.png`) in §8.3 (underestimation discussion).
- Tighten cross-references ("Section 6.5" instead of bare "Section 6").

### Section 9 (was §8) — Conclusions
- No content change, just renumbering.

### Section 10 (was §9) — Bibliography
- Standardise to APA 7: italics on journal/proceedings names rendered via markdown emphasis, consistent author punctuation, add URLs/DOIs where missing.

### Section 11 (was §10) — Annexes
- Renumber.
- Update Annex A repository tree to reflect any path changes.

### Cross-cutting
- Numbered tables: Table 1, Table 2, … with descriptive captions above each table.
- Numbered figures: Figure 1, Figure 2, … with captions below each embedded image (markdown alt-text + bold caption line).
- Replace remaining em-dashes throughout.
- Sweep cross-references for explicit section numbers.

## Out of scope

- Changes to notebooks, scripts, or output data.
- Re-running the pipeline or regenerating any figure.
- Changes to validation methodology or results.
- Translation to German or any other language.

## Success criteria

- All 11 rubric sections present and correctly numbered.
- At least 12 numbered figures embedded with captions.
- All tables numbered with captions.
- Bibliography internally consistent in APA 7 style.
- No em-dashes in the final text.
- Report renders cleanly with pandoc to PDF (syntax-valid markdown).
- Single git commit on `main` covering the full change.
