import math


FPS = 24

RESOLUTIONS = {
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

NATIVE_MIN_FRAMES = 124
NATIVE_MAX_FRAMES = 345


def closest_aspect(w: int, h: int) -> str:
    if w <= 0 or h <= 0:
        raise ValueError("Image dimensions must be positive")
    target = math.log(w / h)
    return min(
        RESOLUTIONS["768p"],
        key=lambda key: abs(
            target
            - math.log(
                int(key.split(":")[0]) / int(key.split(":")[1])
            )
        ),
    )


def frames_for_duration(seconds: float) -> int:
    if not math.isfinite(seconds):
        raise ValueError("Duration must be finite")
    return max(1, round(seconds * FPS))


def validate_num_frames(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError("num_frames must be an integer")
    if (n - 5) % 17 != 0 or (n - 5) // 17 < 1:
        raise ValueError("num_frames must satisfy num_frames = 17*n + 5, n >= 1")
    return n


def _native_ceiling(value: int) -> int:
    index = max(7, math.ceil((value - 5) / 17))
    if index > 20:
        raise ValueError("Requested segment exceeds the native frame range")
    return 17 * index + 5


def plan_segments(F: int) -> tuple[list[int], int]:
    if isinstance(F, bool) or not isinstance(F, int) or F < 1:
        raise ValueError("The final frame count must be a positive integer")
    if F <= NATIVE_MAX_FRAMES:
        return [_native_ceiling(max(F, NATIVE_MIN_FRAMES))], F
    count = math.ceil((F - 1) / 344)
    minimum_length = 1 + math.ceil((F - 1) / count)
    length = _native_ceiling(minimum_length)
    return [length] * count, F
