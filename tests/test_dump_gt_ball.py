"""Unit tests for the GT ball extraction helper in dump_gt_ball.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from dump_gt_ball import parse_gt_ball  # noqa: E402
from _pipeline_core import PITCH_LENGTH_M, PITCH_WIDTH_M  # noqa: E402


def _labels(annotations: list[dict]) -> dict:
    """Minimal Labels-GameState dict with one image (frame 5) and given annotations."""
    return {"images": [{"file_name": "000005.jpg", "image_id": "img5"}], "annotations": annotations}


def test_parse_gt_ball_converts_centred_to_topleft_metres():
    labels = _labels(
        [{"image_id": "img5", "category_id": 4, "bbox_pitch": {"x_bottom_middle": 0.0, "y_bottom_middle": 0.0}}]
    )
    assert parse_gt_ball(labels, 5) == (PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2)


def test_parse_gt_ball_returns_none_when_no_ball_annotation():
    # Only a player (category_id 1), no ball (category_id 4).
    labels = _labels(
        [{"image_id": "img5", "category_id": 1, "bbox_pitch": {"x_bottom_middle": 1.0, "y_bottom_middle": 2.0}}]
    )
    assert parse_gt_ball(labels, 5) is None


def test_parse_gt_ball_returns_none_for_unknown_frame():
    labels = _labels(
        [{"image_id": "img5", "category_id": 4, "bbox_pitch": {"x_bottom_middle": 0.0, "y_bottom_middle": 0.0}}]
    )
    assert parse_gt_ball(labels, 999) is None


def test_parse_gt_ball_returns_none_when_bbox_pitch_missing_coords():
    labels = _labels(
        [{"image_id": "img5", "category_id": 4, "bbox_pitch": {"x_bottom_middle": None, "y_bottom_middle": None}}]
    )
    assert parse_gt_ball(labels, 5) is None


@pytest.mark.parametrize(
    "x_centred,y_centred,expected",
    [
        (-52.5, -34.0, (0.0, 0.0)),
        (52.5, 34.0, (PITCH_LENGTH_M, PITCH_WIDTH_M)),
    ],
)
def test_parse_gt_ball_corner_conversions(x_centred, y_centred, expected):
    labels = _labels(
        [
            {
                "image_id": "img5",
                "category_id": 4,
                "bbox_pitch": {"x_bottom_middle": x_centred, "y_bottom_middle": y_centred},
            }
        ]
    )
    assert parse_gt_ball(labels, 5) == expected
