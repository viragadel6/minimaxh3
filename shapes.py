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

NATIVE_MIN_N = 7
NATIVE_MAX_N = 20
NATIVE_MIN_FRAMES = 17 * NATIVE_MIN_N + 5
NATIVE_MAX_FRAMES = 17 * NATIVE_MAX_N + 5


def closest_aspect(w: int, h: int) -> str:
    if w <= 0 or h <= 0:
        raise ValueError("Width and height must be positive")
    target = math.log(w / h)
    return min(
        RESOLUTIONS["768p"].items(),
        key=lambda item: abs(target - math.log(item[1][0] / item[1][1])),
    )[0]


def frames_for_duration(seconds: float) -> int:
    return max(1, round(seconds * FPS))


def validate_num_frames(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError("num_frames must be an integer")
    if (n - 5) % 17 != 0 or (n - 5) // 17 < 1:
        raise ValueError("num_frames must follow 17n + 5 with n >= 1")
    return n


def plan_segments(F: int) -> tuple[list[int], int]:
    if isinstance(F, bool) or not isinstance(F, int) or F < 1:
        raise ValueError("Frame count F must be a positive integer")
    if F <= 345:
        target = max(F, 124)
        n = max(7, math.ceil((target - 5) / 17))
        return ([17 * n + 5], F)
    k = math.ceil((F - 1) / 344)
    for n in range(7, 21):
        L = 17 * n + 5
        if 1 + k * (L - 1) >= F:
            return ([L] * k, F)
    return ([345] * k, F)
