"""Matriks perilaku wajib AC01–AC14 (bagian geometri & episode) — tanpa video/model."""

import numpy as np
import pytest

from disposal.area_rules import (
    IGNORED_NO_ID,
    IGNORED_OCCLUDED,
    IGNORED_TRUNCATED,
    OUTSIDE,
    SKIPPED,
    VIOLATION,
    AreaRules,
    Detection,
)
from disposal.config import Area

AREA = Area("violation_area", np.array([[100, 100], [300, 100], [300, 200], [100, 200]], dtype=np.float32))
FS = (640, 480)


def rules(missing: int = 30) -> AreaRules:
    return AreaRules(
        AREA,
        target_classes={"Truck"},
        occluder_classes={"Truck", "Light Vehicle"},
        margin_px=2,
        max_missing_frames=missing,
    )


def det(cls="Truck", conf=0.9, tid=1, bbox=(120, 110, 220, 190)) -> Detection:
    return Detection(class_name=cls, confidence=conf, track_id=tid, bbox=tuple(bbox))


def test_ac01_bbox_fully_outside():
    decisions, events = rules().evaluate([det(bbox=(340, 110, 440, 190))], 0, FS, 0.0)
    assert decisions[0].status == OUTSIDE and not events


def test_ac02_partial_overlap_no_event_and_ratio_not_iou():
    decisions, events = rules().evaluate([det(bbox=(250, 110, 350, 190))], 0, FS, 0.0)
    assert decisions[0].status == OUTSIDE and not events
    assert abs(decisions[0].coverage_ratio - 0.5) < 1e-4


def test_ac03_fully_inside_violation_one_event():
    decisions, events = rules().evaluate([det(bbox=(120, 110, 220, 190))], 0, FS, 0.0)
    assert decisions[0].status == VIOLATION
    assert len(events) == 1 and events[0].frame_index == 0


def test_ac04_touching_polygon_boundary_from_inside():
    decisions, events = rules().evaluate([det(bbox=(100, 110, 300, 190))], 0, FS, 0.0)
    assert decisions[0].status == VIOLATION and len(events) == 1


def test_ac05_truncated_by_frame_edge():
    decisions, events = rules().evaluate([det(bbox=(0, 110, 220, 190))], 0, FS, 0.0)
    assert decisions[0].status == IGNORED_TRUNCATED and not events


def test_ac05_margin_within_8px_counts_truncated():
    rs = AreaRules(AREA, target_classes={"Truck"}, occluder_classes={"Truck"}, margin_px=8, max_missing_frames=30)
    decisions, events = rs.evaluate([det(tid=1, bbox=(120, 110, 637, 190))], 0, FS, 0.0)
    assert decisions[0].status == IGNORED_TRUNCATED and not events


def test_subpixel_leak_still_inside_and_real_exit_outside():
    decisions, events = rules().evaluate([det(tid=1, bbox=(120, 99.7, 220, 190))], 0, FS, 0.0)
    assert decisions[0].status == VIOLATION and len(events) == 1
    decisions, events = rules().evaluate([det(tid=2, bbox=(120, 96, 220, 190))], 1, FS, 1 / 25.0)
    assert decisions[0].status == OUTSIDE and not events


def test_ac06_overlapping_bboxes_both_occluded():
    rs = rules()
    a = det(tid=1, bbox=(120, 110, 220, 190))
    b = det(cls="Light Vehicle", tid=2, bbox=(200, 110, 300, 190))
    decisions, events = rs.evaluate([a, b], 0, FS, 0.0)
    assert [d.status for d in decisions] == [IGNORED_OCCLUDED, IGNORED_OCCLUDED]
    assert not events


def test_ac07_edge_touch_only_no_occlusion():
    a = det(tid=1, bbox=(120, 110, 220, 190))
    b = det(cls="Light Vehicle", tid=2, bbox=(220, 110, 320, 190))
    decisions, events = rules().evaluate([a, b], 0, FS, 0.0)
    assert decisions[0].status == VIOLATION and decisions[1].status == SKIPPED
    assert len(events) == 1


def test_ac08_single_event_per_episode():
    rs = rules()
    for idx in range(10):
        _, events = rs.evaluate([det(bbox=(120, 110, 220, 190))], idx, FS, idx / 25.0)
        assert len(events) == (1 if idx == 0 else 0)


def test_ac09_event_written_when_exception_clears():
    rs = rules()
    truck = det(tid=1, bbox=(120, 110, 220, 190))
    occluder = det(cls="Light Vehicle", tid=2, bbox=(200, 110, 300, 190))
    _, events = rs.evaluate([truck, occluder], 0, FS, 0.0)
    assert not events
    _, events = rs.evaluate([truck], 2, FS, 2 / 25.0)
    assert len(events) == 1 and events[0].frame_index == 2

    rs2 = rules()
    rs2.evaluate([truck], 0, FS, 0.0)
    rs2.evaluate([truck, occluder], 1, FS, 1 / 25.0)
    _, events = rs2.evaluate([truck], 2, FS, 2 / 25.0)
    assert len(events) == 0


def test_ac10_reentry_same_id_new_event():
    rs = rules()
    rs.evaluate([det(tid=7, bbox=(120, 110, 220, 190))], 0, FS, 0.0)
    rs.evaluate([det(tid=7, bbox=(330, 110, 430, 190))], 10, FS, 10 / 25.0)
    _, events = rs.evaluate([det(tid=7, bbox=(120, 110, 220, 190))], 20, FS, 20 / 25.0)
    assert len(events) == 1 and events[0].frame_index == 20


def test_ac11_first_seen_already_inside():
    _, events = rules().evaluate([det(tid=3, bbox=(120, 110, 220, 190))], 5, FS, 5 / 25.0)
    assert len(events) == 1


def test_ac12_missing_frames_then_return_no_duplicate():
    rs = rules(missing=30)
    rs.evaluate([det(tid=9, bbox=(120, 110, 220, 190))], 0, FS, 0.0)
    for idx in range(1, 11):
        _, events = rs.evaluate([], idx, FS, idx / 25.0)
        assert not events
    _, events = rs.evaluate([det(tid=9, bbox=(120, 110, 220, 190))], 11, FS, 11 / 25.0)
    assert len(events) == 0


def test_state_pruned_after_max_missing():
    rs = rules(missing=5)
    rs.evaluate([det(tid=2, bbox=(120, 110, 220, 190))], 0, FS, 0.0)
    rs.evaluate([], 6, FS, 6 / 25.0)
    assert 2 not in rs.states


def test_no_track_id_never_events():
    decisions, events = rules().evaluate([det(tid=None, bbox=(120, 110, 220, 190))], 0, FS, 0.0)
    assert decisions[0].status == IGNORED_NO_ID and not events
    assert not rules().states


def test_non_target_class_skipped():
    decisions, events = rules().evaluate([det(cls="Person", tid=4, bbox=(120, 110, 220, 190))], 0, FS, 0.0)
    assert decisions[0].status == SKIPPED and not events


def test_invalid_bbox_never_violation():
    decisions, events = rules().evaluate([det(tid=5, bbox=(100, 100, 100, 200))], 0, FS, 0.0)
    assert decisions[0].status == SKIPPED and not events
