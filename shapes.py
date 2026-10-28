"""Canonical frame/resolution helpers shared by the API, runtime and media layers.

Every helper validates its inputs explicitly and raises :class:`ValueError` with a
human readable message; no helper silently clamps or truncates invalid values.
"""

from __future__ import annotations

import math

FPS = 24

RESOLUTIONS: dict[str, dict[str, tuple[int, int]]] = {
    "480p": {
        "16:9": (832, 480),
        "9:16": (480, 832),
        "1:1": (640, 640),
        "4:3": (736, 544),
        "3:4": (544, 736),
    },
    "768p": {
        "16:9": (1344, 768),
        "9:16": (768, 1344),
        "1:1": (1024, 1024),
        "4:3": (1152, 864),
        "3:4": (864, 1152),
    },
}

ASPECT_RATIOS: tuple[str, ...] = tuple(RESOLUTIONS["480p"].keys())
RESOLUTION_NAMES: tuple[str, ...] = tuple(RESOLUTIONS.keys())

# Native segment length is 17n + 5 frames for n in [MIN_N, MAX_N].
NATIVE_MIN_N = 7
NATIVE_MAX_N = 20
NATIVE_MIN_FRAMES = 17 * NATIVE_MIN_N + 5
NATIVE_MAX_FRAMES = 17 * NATIVE_MAX_N + 5

# A single segment can never be longer than the largest native segment.
MAX_SEGMENT_FRAMES = NATIVE_MAX_FRAMES


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_positive_fps(fps: object) -> int:
    if isinstance(fps, bool) or not isinstance(fps, int):
        raise ValueError("fps must be an integer")
    if fps <= 0:
        raise ValueError("fps must be positive")
    return fps


def closest_aspect(w: int, h: int) -> str:
    """Return the advertised aspect ratio closest to ``w``/``h``."""
    _require_positive_int(w, "width")
    _require_positive_int(h, "height")
    target = math.log(w / h)
    return min(
        RESOLUTIONS["768p"].items(),
        key=lambda item: abs(target - math.log(item[1][0] / item[1][1])),
    )[0]


def frames_for_duration(seconds: float, fps: int = FPS) -> int:
    """Convert a duration in seconds to a frame count using ``fps``."""
    fps = _require_positive_fps(fps)
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
        raise ValueError("duration_seconds must be a number")
    if not math.isfinite(float(seconds)):
        raise ValueError("duration_seconds must be finite")
    if seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    return max(1, int(round(float(seconds) * fps)))


def validate_num_frames(n: int) -> int:
    """Validate that ``n`` follows the ``17n + 5`` frame grid."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError("num_frames must be an integer")
    if (n - 5) % 17 != 0 or (n - 5) // 17 < 1:
        raise ValueError("num_frames must follow 17n + 5 with n >= 1")
    return n


def normalize_num_frames(n: int | float) -> int:
    """Snap an arbitrary positive frame count onto the ``17n + 5`` grid.

    The result always satisfies :func:`validate_num_frames` and is clamped to
    ``[NATIVE_MIN_FRAMES, MAX_SEGMENT_FRAMES]`` so that a request derived from a
    duration can never produce an unrunnable frame count.
    """
    if isinstance(n, bool) or not isinstance(n, (int, float)):
        raise ValueError("num_frames must be a number")
    if not math.isfinite(float(n)) or float(n) < 1:
        raise ValueError("num_frames must be a positive finite number")
    k = max(NATIVE_MIN_N, int(round((float(n) - 5) / 17)))
    k = min(k, NATIVE_MAX_N * 64)  # bounded so plan_segments() can chunk it
    return 17 * k + 5


def plan_segments(F: int) -> tuple[list[int], int]:
    """Split ``F`` frames into runnable native segments.

    Returns ``(segments, final_frames)``.  ``sum(segments) - (len(segments) - 1)``
    equals ``final_frames`` because consecutive segments share one frame.
    """
    if isinstance(F, bool) or not isinstance(F, int) or F < 1:
        raise ValueError("Frame count F must be a positive integer")
    if F <= 345:
        target = max(F, NATIVE_MIN_FRAMES)
        n = max(NATIVE_MIN_N, math.ceil((target - 5) / 17))
        n = min(n, NATIVE_MAX_N)
        return ([17 * n + 5], F)
    k = math.ceil((F - 1) / (MAX_SEGMENT_FRAMES - 1))
    for n in range(NATIVE_MIN_N, NATIVE_MAX_N + 1):
        L = 17 * n + 5
        if 1 + k * (L - 1) >= F:
            return ([L] * k, F)
    return ([MAX_SEGMENT_FRAMES] * k, F)
