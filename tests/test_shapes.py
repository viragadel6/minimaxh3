import math

import pytest

import shapes


def test_frames_for_duration_respects_fps():
    assert shapes.frames_for_duration(1.0, 24) == 24
    assert shapes.frames_for_duration(1.0, 30) == 30
    with pytest.raises(ValueError):
        shapes.frames_for_duration(1.0, 0)
    with pytest.raises(ValueError):
        shapes.frames_for_duration(float("nan"))
    with pytest.raises(ValueError):
        shapes.frames_for_duration(float("inf"))
    with pytest.raises(ValueError):
        shapes.frames_for_duration(-1)


def test_normalize_num_frames_is_on_grid_and_bounded():
    for frames in (1, 5, 124, 345, 1000):
        normalized = shapes.normalize_num_frames(frames)
        assert shapes.validate_num_frames(normalized) == normalized
        assert normalized >= shapes.NATIVE_MIN_FRAMES
    with pytest.raises(ValueError):
        shapes.normalize_num_frames(0)
    with pytest.raises(ValueError):
        shapes.normalize_num_frames(float("nan"))


def test_validate_num_frames_rejects_bad_grid():
    assert shapes.validate_num_frames(124) == 124
    for bad in (0, 4, 5, 6, 123, True, 1.5):
        with pytest.raises(ValueError):
            shapes.validate_num_frames(bad)


def test_plan_segments_covers_requested_frames():
    for frames in (124, 200, 345, 690, 1200):
        segments, final = shapes.plan_segments(frames)
        assert final == frames
        coverage = 1 + sum(seg - 1 for seg in segments)
        assert coverage >= frames
        for seg in segments:
            assert (seg - 5) % 17 == 0
            assert shapes.NATIVE_MIN_FRAMES <= seg <= shapes.MAX_SEGMENT_FRAMES


def test_closest_aspect_requires_positive_dimensions():
    assert shapes.closest_aspect(1344, 768) == "16:9"
    with pytest.raises(ValueError):
        shapes.closest_aspect(0, 768)
    with pytest.raises(ValueError):
        shapes.closest_aspect(10, -1)


def test_native_bounds_constants():
    assert shapes.NATIVE_MIN_FRAMES == 17 * shapes.NATIVE_MIN_N + 5
    assert shapes.NATIVE_MAX_FRAMES == 17 * shapes.NATIVE_MAX_N + 5
    assert math.isfinite(float(shapes.FPS))
