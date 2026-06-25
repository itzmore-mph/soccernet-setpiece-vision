"""Unit tests for the core Pitch Control and team-assignment functions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from _pipeline_core import (  # noqa: E402
    GRID_NX,
    GRID_NY,
    REACTION_TIME,
    _is_ref_like,
    assign_teams_kmeans,
    jersey_hsv,
    pitch_control_surface,
    split_attack_defend,
    summarise_surface,
    time_to_intercept,
)

# ---------------------------------------------------------------- time_to_intercept


def test_time_to_intercept_output_shape():
    players = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])  # P=3
    grid = np.array([[0.0, 0.0], [5.0, 5.0]])  # G=2
    tti = time_to_intercept(players, grid)
    assert tti.shape == (3, 2)


def test_time_to_intercept_zero_distance_is_reaction_time():
    players = np.array([[25.0, 17.0]])
    grid = np.array([[25.0, 17.0]])  # same point → distance 0
    tti = time_to_intercept(players, grid)
    assert tti[0, 0] == pytest.approx(REACTION_TIME)


# ---------------------------------------------------------------- pitch_control_surface


def test_pitch_control_surface_shape():
    att = np.array([[30.0, 34.0]])
    deff = np.array([[75.0, 34.0]])
    pc = pitch_control_surface(att, deff, (52.5, 34.0))
    assert pc.shape == (GRID_NY, GRID_NX)


def test_pitch_control_surface_values_in_unit_interval():
    att = np.array([[20.0, 30.0], [40.0, 50.0]])
    deff = np.array([[80.0, 30.0], [60.0, 10.0]])
    pc = pitch_control_surface(att, deff, (50.0, 34.0))
    assert pc.min() >= 0.0
    assert pc.max() <= 1.0


def test_pitch_control_surface_empty_team_returns_half():
    deff = np.array([[80.0, 34.0]])
    pc = pitch_control_surface(np.zeros((0, 2)), deff, (50.0, 34.0))
    assert np.array_equal(pc, np.full((GRID_NY, GRID_NX), 0.5))

    att = np.array([[20.0, 34.0]])
    pc2 = pitch_control_surface(att, np.zeros((0, 2)), (50.0, 34.0))
    assert np.array_equal(pc2, np.full((GRID_NY, GRID_NX), 0.5))


def test_pitch_control_surface_symmetric_mean_near_half():
    # Attacker and defender mirrored about the half-way line → balanced surface.
    att = np.array([[30.0, 34.0]])
    deff = np.array([[75.0, 34.0]])  # 105 - 30 = 75
    pc = pitch_control_surface(att, deff, (52.5, 34.0))
    assert pc.mean() == pytest.approx(0.5, abs=0.05)


def test_pitch_control_surface_closer_attacker_gives_higher_control():
    deff = np.array([[90.0, 34.0]])  # defender fixed, far right
    target = (20.0, 34.0)
    ix = int(round(target[0] / 105.0 * (GRID_NX - 1)))
    iy = int(round(target[1] / 68.0 * (GRID_NY - 1)))

    pc_close = pitch_control_surface(np.array([[20.0, 34.0]]), deff, (52.5, 34.0))
    pc_far = pitch_control_surface(np.array([[50.0, 34.0]]), deff, (52.5, 34.0))
    assert pc_close[iy, ix] > pc_far[iy, ix]


# ---------------------------------------------------------------- split_attack_defend


def test_split_attack_defend_nearest_to_ball_is_attacker():
    players = np.array([[10.0, 34.0], [90.0, 34.0]])
    teams = np.array([0, 1])
    att_xy, def_xy, att_team = split_attack_defend(players, teams, (5.0, 34.0))
    assert att_team == 0
    assert np.array_equal(att_xy, np.array([[10.0, 34.0]]))
    assert np.array_equal(def_xy, np.array([[90.0, 34.0]]))


def test_split_attack_defend_single_team_all_attack():
    players = np.array([[10.0, 34.0], [20.0, 40.0]])
    teams = np.array([0, 0])
    att_xy, def_xy, _ = split_attack_defend(players, teams, (5.0, 34.0))
    assert np.array_equal(att_xy, players)
    assert def_xy.shape == (0, 2)


def test_split_attack_defend_excludes_referees():
    players = np.array([[10.0, 34.0], [90.0, 34.0], [50.0, 34.0]])
    teams = np.array([0, 1, -1])  # last is a referee
    att_xy, def_xy, _ = split_attack_defend(players, teams, (5.0, 34.0))
    combined = np.vstack([att_xy, def_xy])
    assert not np.any(np.all(combined == np.array([50.0, 34.0]), axis=1))
    assert len(combined) == 2


# ---------------------------------------------------------------- summarise_surface


def test_summarise_surface_pc_mean_matches():
    rng = np.random.default_rng(0)
    pc = rng.random((GRID_NY, GRID_NX))
    out = summarise_surface(pc, (52.5, 34.0))
    assert out["pc_mean"] == float(pc.mean())


def test_summarise_surface_area_gt_half_matches():
    rng = np.random.default_rng(1)
    pc = rng.random((GRID_NY, GRID_NX))
    out = summarise_surface(pc, (52.5, 34.0))
    assert out["pc_area_gt_0p5"] == float((pc > 0.5).mean())


def test_summarise_surface_single_cell_outside_zones_is_nan():
    pc = np.array([[0.5]])  # nx=1, ny=1; single cell sits at x=0
    out = summarise_surface(pc, (60.0, 34.0))  # bx >= 52.5 → right-side zones
    assert np.isnan(out["pc_in_box"])
    assert np.isnan(out["pc_in_third"])


# ---------------------------------------------------------------- assign_teams_kmeans


def test_assign_teams_kmeans_too_few_valid_returns_all_minus_one():
    features = np.array([[10.0, 10.0, 10.0], [np.nan, np.nan, np.nan]])  # n_valid = 1
    labels = assign_teams_kmeans(features)
    assert np.all(labels == -1)


def test_assign_teams_kmeans_drops_small_ref_like_cluster():
    # Two large team clusters (blue, red) plus a small ref-like (yellow) cluster.
    blue = np.tile([120.0, 180.0, 180.0], (5, 1))
    red = np.tile([0.0, 180.0, 180.0], (5, 1))
    yellow = np.tile([40.0, 150.0, 200.0], (2, 1))  # small AND ref-like → dropped
    features = np.vstack([blue, red, yellow])
    labels = assign_teams_kmeans(features)

    yellow_labels = labels[-2:]
    team_labels = labels[:-2]
    assert np.all(yellow_labels == -1)
    assert set(np.unique(team_labels)) == {0, 1}


# ---------------------------------------------------------------- jersey_hsv


def test_jersey_hsv_zero_area_box_returns_nan():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = jersey_hsv(image, np.array([10, 10, 0, 20]))  # width 0 → zero area
    assert result.shape == (3,)
    assert np.all(np.isnan(result))


# ---------------------------------------------------------------- _is_ref_like


def test_is_ref_like_yellow_centroid_true():
    assert _is_ref_like(np.array([40.0, 150.0, 200.0])) is True


def test_is_ref_like_blue_centroid_false():
    assert _is_ref_like(np.array([120.0, 180.0, 180.0])) is False
