"""Validasi config (trust boundary)."""

import json

import pytest

from disposal.config import load_config

BASE = {
    "reference_width": 640,
    "reference_height": 480,
    "areas": {
        "main_area": {"points": [[1, 479], [639, 479], [639, 1], [1, 1]]},
        "violation_area": {"points": [[100, 100], [300, 100], [300, 200], [100, 200]]},
    },
    "target_classes": ["Truck"],
    "occluder_classes": ["Truck"],
}


def write(tmp_path, raw):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    return p


def test_valid_load(tmp_path):
    rc = load_config(write(tmp_path, BASE))
    assert rc.reference_width == 640 and rc.violation_area.name == "violation_area"
    assert rc.target_classes == ["Truck"] and rc.device == "auto"


def test_concave_violation_area_rejected(tmp_path):
    raw = json.loads(json.dumps(BASE))
    raw["areas"]["violation_area"]["points"] = [[100, 100], [300, 100], [300, 200], [150, 150], [100, 200]]
    with pytest.raises(ValueError, match="convex"):
        load_config(write(tmp_path, raw))


def test_too_few_points_rejected(tmp_path):
    raw = json.loads(json.dumps(BASE))
    raw["areas"]["violation_area"]["points"] = [[100, 100], [300, 100]]
    with pytest.raises(ValueError, match="3 titik"):
        load_config(write(tmp_path, raw))


def test_zero_area_rejected(tmp_path):
    raw = json.loads(json.dumps(BASE))
    raw["areas"]["violation_area"]["points"] = [[0, 0], [10, 10], [20, 20]]
    with pytest.raises(ValueError, match="luas"):
        load_config(write(tmp_path, raw))


def test_points_outside_reference_rejected(tmp_path):
    raw = json.loads(json.dumps(BASE))
    raw["areas"]["violation_area"]["points"] = [[100, 100], [700, 100], [700, 200], [100, 200]]
    with pytest.raises(ValueError, match="di luar frame"):
        load_config(write(tmp_path, raw))


def test_empty_targets_rejected(tmp_path):
    raw = json.loads(json.dumps(BASE))
    raw["target_classes"] = []
    with pytest.raises(ValueError, match="target_classes"):
        load_config(write(tmp_path, raw))


def test_bad_threshold_rejected(tmp_path):
    raw = json.loads(json.dumps(BASE))
    raw["confidence_threshold"] = 1.5
    with pytest.raises(ValueError, match="confidence_threshold"):
        load_config(write(tmp_path, raw))
