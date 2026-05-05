"""Patch nb01: replace cell 15 with diagnostic-rich version that surfaces silent
sb.frames() failures and rate-limit hiccups. Idempotent.

Run once:
    python scripts/_patch_nb01_cell15.py
"""

from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "01_business_and_data_understanding.ipynb"

NEW_SRC = '''# Set-piece extraction loop. Robust: explicit per-match diagnostics on
# events/frames failures, retry with backoff for transient statsbombpy errors,
# and a summary of frame-attach success at the end.

import time

rows = []
skipped = []
diag_frames_empty = []
match_ids = matches["match_id"].tolist()


def _fetch_with_retry(fn, *, attempts: int = 3, delay: float = 1.0, **kwargs):
    """Call statsbombpy fn(**kwargs) up to N times with linear backoff."""
    last = None
    for k in range(attempts):
        try:
            return fn(**kwargs)
        except Exception as e:
            last = e
            time.sleep(delay * (k + 1))
    raise last


for i, mid in enumerate(match_ids, 1):
    mid = int(mid)
    try:
        events = _fetch_with_retry(sb.events, match_id=mid)
    except Exception as e:
        skipped.append((mid, f"events: {type(e).__name__}: {e}"))
        continue

    if events is None or events.empty:
        skipped.append((mid, "events empty"))
        continue

    corners = events[(events["type"] == "Pass") & (events.get("pass_type") == "Corner")].copy()
    corners["event_type"] = "corner"

    pass_type_col = events["pass_type"] if "pass_type" in events.columns else pd.Series([None] * len(events), index=events.index)
    shot_type_col = events["shot_type"] if "shot_type" in events.columns else pd.Series([None] * len(events), index=events.index)
    start_x = events["location"].apply(lambda l: l[0] if isinstance(l, list) and len(l) >= 1 else np.nan)
    fk_shot = (events["type"] == "Shot") & (shot_type_col == "Free Kick")
    fk_pass_attacking = (events["type"] == "Pass") & (pass_type_col == "Free Kick") & (start_x >= 80)
    free_kicks = events[fk_shot | fk_pass_attacking].copy()
    free_kicks["event_type"] = "free_kick"

    set_pieces = pd.concat([corners, free_kicks], ignore_index=True)
    if set_pieces.empty:
        continue

    # Frames: retry, log empty/None reason explicitly
    try:
        frames = _fetch_with_retry(sb.frames, match_id=mid)
    except Exception as e:
        diag_frames_empty.append((mid, f"frames raised: {type(e).__name__}: {e}"))
        frames = None

    if frames is None:
        frames_by_event = {}
    elif frames.empty:
        diag_frames_empty.append((mid, "frames returned empty df"))
        frames_by_event = {}
    else:
        ff_key = "event_uuid" if "event_uuid" in frames.columns else "id"
        frames_by_event = {eid: grp for eid, grp in frames.groupby(ff_key)}

    for _, ev in set_pieces.iterrows():
        loc = ev.get("location")
        if loc is None or (isinstance(loc, float) and np.isnan(loc)):
            continue
        ball_x_m, ball_y_m = statsbomb_to_metres(loc[0], loc[1])
        ev_id = ev.get("id")
        ff_rows = frames_by_event.get(ev_id)
        ff = convert_freeze_frame(ff_rows) if ff_rows is not None else []
        rows.append({
            "match_id": int(mid),
            "event_id": ev_id,
            "event_type": ev["event_type"],
            "period": int(ev.get("period", 0) or 0),
            "minute": int(ev.get("minute", 0) or 0),
            "second": int(ev.get("second", 0) or 0),
            "team": ev.get("team"),
            "ball_x_m": float(ball_x_m),
            "ball_y_m": float(ball_y_m),
            "freeze_frame": ff,
        })

    if i % 10 == 0:
        n_with_ff_so_far = sum(1 for r in rows if r["freeze_frame"])
        print(f"  processed {i}/{len(match_ids)} matches  |  set pieces: {len(rows)}  |  with FF: {n_with_ff_so_far}")

n_with_ff_total = sum(1 for r in rows if r["freeze_frame"])
print(f"\\nDone. Total set pieces: {len(rows)}  |  With FF: {n_with_ff_total}  |  Coverage: {100*n_with_ff_total/max(len(rows),1):.1f} %")
print(f"Matches with frames issue: {len(diag_frames_empty)}  |  Matches fully skipped: {len(skipped)}")
for mid, why in diag_frames_empty[:5]:
    print(f"  frames-issue {mid}: {why}")
for mid, why in skipped[:5]:
    print(f"  skipped      {mid}: {why}")
'''


def main():
    nb = json.loads(NB.read_text())
    cell = nb["cells"][15]
    assert cell["cell_type"] == "code", "cell 15 not code"
    cell["source"] = NEW_SRC.splitlines(keepends=True)
    cell["outputs"] = []
    cell["execution_count"] = None
    NB.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"Patched cell 15 in {NB}")


if __name__ == "__main__":
    main()
