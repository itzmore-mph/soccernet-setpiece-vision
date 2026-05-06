"""Compute pitch control surfaces from Soccana detections, mirroring nb03 exactly.

Reads:
    outputs/detections_soccana.parquet
    outputs/ball_positions.parquet

Writes:
    outputs/pitch_control_soccana.parquet  (track='soccana', schema matches pitch_control.parquet)

All Laurie Shaw / Spearman model parameters identical to nb03, locked for
reproducibility. Any change here invalidates the nb04 ablation comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
GRID_NX = 60
GRID_NY = 40
MAX_SPEED = 5.0
REACTION_TIME = 0.7
SIGMA = 0.45
TIME_TO_INTERCEPT_SIGMOID_K = np.pi / (np.sqrt(3.0) * SIGMA)


def time_to_intercept(player_xy, target_xy):
    diff = player_xy[:, None, :] - target_xy[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return REACTION_TIME + dist / MAX_SPEED


def pitch_control_surface(att_xy, def_xy, ball_xy, nx=GRID_NX, ny=GRID_NY):
    xs = np.linspace(0.0, PITCH_LENGTH_M, nx)
    ys = np.linspace(0.0, PITCH_WIDTH_M, ny)
    grid = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
    if len(att_xy) == 0 or len(def_xy) == 0:
        return np.full((ny, nx), 0.5)
    tti_att = time_to_intercept(att_xy, grid).min(axis=0)
    tti_def = time_to_intercept(def_xy, grid).min(axis=0)
    delta = tti_att - tti_def
    return (1.0 / (1.0 + np.exp(TIME_TO_INTERCEPT_SIGMOID_K * delta))).reshape(ny, nx)


def split_attack_defend(players_xy, team_labels, ball_xy):
    bx, by = ball_xy
    teams = [t for t in np.unique(team_labels)
             if t is not None and t != -1 and not (isinstance(t, float) and np.isnan(t))]
    if len(teams) < 2:
        return players_xy, np.zeros((0, 2)), teams[0] if teams else None
    min_d = {}
    for t in teams:
        mask = team_labels == t
        if mask.sum() == 0:
            continue
        d = np.linalg.norm(players_xy[mask] - np.array([bx, by]), axis=1)
        min_d[t] = d.min()
    att_team = min(min_d, key=min_d.get)
    def_team = [t for t in min_d if t != att_team][0]
    return players_xy[team_labels == att_team], players_xy[team_labels == def_team], att_team


def summarise_surface(pc, ball_xy):
    bx, by = ball_xy
    ny, nx = pc.shape
    xs = np.linspace(0.0, PITCH_LENGTH_M, nx)
    ys = np.linspace(0.0, PITCH_WIDTH_M, ny)
    ix = int(np.clip(round((bx / PITCH_LENGTH_M) * (nx - 1)), 0, nx - 1))
    iy = int(np.clip(round((by / PITCH_WIDTH_M) * (ny - 1)), 0, ny - 1))
    pc_at_ball = float(pc[iy, ix])
    XX, YY = np.meshgrid(xs, ys)
    pen_left = (XX <= 16.5) & (YY >= 13.84) & (YY <= 54.16)
    pen_right = (XX >= PITCH_LENGTH_M - 16.5) & (YY >= 13.84) & (YY <= 54.16)
    if bx < PITCH_LENGTH_M / 2:
        att_third = XX <= PITCH_LENGTH_M / 3
        att_box = pen_left
    else:
        att_third = XX >= 2 * PITCH_LENGTH_M / 3
        att_box = pen_right
    return {
        "pc_mean": float(pc.mean()),
        "pc_at_ball": pc_at_ball,
        "pc_in_box": float(pc[att_box].mean()) if att_box.any() else np.nan,
        "pc_in_third": float(pc[att_third].mean()) if att_third.any() else np.nan,
        "pc_area_gt_0p5": float((pc > 0.5).mean()),
    }


def process_track(df, track_name, team_col, balls):
    out = []
    for (split, clip_id, frame_idx), g in df.groupby(["split", "clip_id", "frame_idx"]):
        ball_row = balls[(balls["split"] == split) & (balls["clip_id"] == clip_id) & (balls["frame_idx"] == frame_idx)]
        if ball_row.empty or ball_row["ball_x_m"].isna().any():
            continue
        ball_xy = (float(ball_row["ball_x_m"].iloc[0]), float(ball_row["ball_y_m"].iloc[0]))
        players_xy = g[["x_m", "y_m"]].to_numpy()
        teams = g[team_col].to_numpy()
        att_xy, def_xy, _ = split_attack_defend(players_xy, teams, ball_xy)
        if len(att_xy) == 0 or len(def_xy) == 0:
            continue
        pc = pitch_control_surface(att_xy, def_xy, ball_xy)
        m = summarise_surface(pc, ball_xy)
        action_class = g["action_class"].iloc[0] if "action_class" in g.columns else None
        out.append({
            "split": split, "clip_id": clip_id, "frame_idx": int(frame_idx),
            "track": track_name, "action_class": action_class,
            "n_attackers": int(len(att_xy)), "n_defenders": int(len(def_xy)),
            "ball_x_m": ball_xy[0], "ball_y_m": ball_xy[1],
            **m,
        })
    return pd.DataFrame(out)


def main():
    soc = pd.read_parquet(OUTPUTS_DIR / "detections_soccana.parquet")
    balls = pd.read_parquet(OUTPUTS_DIR / "ball_positions.parquet")
    print(f"detections_soccana: {soc.shape}  |  ball_positions: {balls.shape}")

    pc = process_track(soc, track_name="soccana", team_col="team_kmeans", balls=balls)
    out = OUTPUTS_DIR / "pitch_control_soccana.parquet"
    pc.to_parquet(out, engine="pyarrow", index=False)
    print(f"pitch_control_soccana frames: {len(pc)}  |  saved: {out}")
    print(pc[["pc_mean", "pc_at_ball", "pc_in_box", "pc_in_third", "pc_area_gt_0p5"]].mean().round(3))


if __name__ == "__main__":
    main()
