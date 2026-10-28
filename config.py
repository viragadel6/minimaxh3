"""Runtime settings.

Settings are a frozen dataclass.  Every value can be overridden through an
environment variable of the same name; ``load_settings`` performs strict,
type aware coercion (notably for booleans and floats) and raises ``ValueError``
instead of silently mis-parsing values such as ``"0"`` as ``True``.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, fields

from shapes import FPS, NATIVE_MAX_FRAMES

PROFILE_NAMES: tuple[str, ...] = ("quality", "fast", "turbo")


def _coerce_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"cannot interpret {raw!r} as a boolean")


def _coerce_value(default, raw: str):
    if isinstance(default, bool):
        return _coerce_bool(raw)
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass(frozen=True)
class Settings:
    APP_NAME: str = "video-audio-diffusion-server"
    PUBLIC_BASE_URL: str = ""
    GPU_TYPE: str = "B200"
    MODEL_REPO_ID: str = "MiniMaxAI/MiniMax-H3"
    MODEL_BLOCKS_MODULE: str = "diffusers.modular_pipelines.minimax_h3"
    FL_SUBFOLDER: str = "FL2VA"
    REF_SUBFOLDER: str = "Ref2VA"
    TURBO_TRANSFORMER_REPO: str = ""
    TURBO_TRANSFORMER_SUBFOLDER: str = "transformer"
    TURBO_STEPS: int = 8
    QUALITY_STEPS: int = 20
    FAST_STEPS: int = 20
    FAST_CACHE_THRESHOLD: float = 0.12
    CACHE_WARMUP_STEPS: int = 3
    CACHE_TAIL_STEPS: int = 2
    CACHE_MIN_SKIPPABLE_STEPS: int = 1
    QUALITY_REPAIR_TAU: float = 0.5
    FAST_REPAIR_TAU: float = -1.0
    TURBO_REPAIR_TAU: float = -1.0
    DEFAULT_PROFILE: str = "fast"
    WEIGHT_PRECISION: str = "fp8"
    FP8_EXCLUDE_PATTERNS: str = "embed,norm,adaln,modulation,time,proj_out,final,head"
    FP8_KEEP_EDGE_BLOCKS: int = 2
    ATTENTION_BACKEND: str = "fp8"
    ATTN_FP8_MIN_SEQ: int = 4096
    COMPILE_MODE: str = "max-autotune-no-cudagraphs"
    VAE_TILING: int = 0
    RESIDENCY_RESERVE_GIB: int = 40
    LORA_GPU_CACHE_SIZE: int = 8
    LORA_CPU_CACHE_SIZE: int = 32
    FUSED_LORAS: str = ""
    WARMUP_BUCKETS: str = "768p:16:9:124,480p:16:9:124"
    WARMUP_STEPS: int = 2
    MIN_CONTAINERS: int = 1
    SCALEDOWN_WINDOW: int = 600
    GPU_MAX_INPUTS: int = 4
    GPU_TARGET_INPUTS: int = 1
    HOST_MEMORY_MIB: int = 262144
    CPU_CORES: int = 16
    FUNCTION_TIMEOUT: int = 3600
    DOWNLOAD_TIMEOUT: int = 6 * 3600
    WEB_TIMEOUT: int = 3600
    SYNC_REQUEST_TIMEOUT: float = 1800.0
    POLL_TIMEOUT_SECONDS: float = 2.0
    POLL_MAX_TRANSIENT_FAILURES: int = 5
    MEDIA_CONCURRENCY: int = 4
    ENCODE_QUEUE_LIMIT: int = 4
    MAX_SYNC_BYTES: int = 512 * 1024 * 1024
    MAX_REFERENCES: int = 8
    MAX_LORAS: int = 8
    MAX_DURATION_SECONDS: float = 3600.0
    OUTPUT_RETENTION_HOURS: float = 0.0
    MODELS_VOLUME: str = "va-models"
    OUTPUTS_VOLUME: str = "va-outputs"
    JOBS_DICT: str = "va-jobs"
    MODELS_DIR: str = "/models"
    OUTPUTS_DIR: str = "/outputs"
    OUTPUT_STAGING_DIR: str = ""
    FPS: int = FPS
    X264_PRESET: str = "veryfast"
    X264_CRF: int = 17
    AUDIO_BITRATE: str = "192k"
    DIFFUSERS_SPEC: str = "diffusers==0.35.1"
    FFMPEG_TIMEOUT: float = 600.0

    def __post_init__(self) -> None:
        errors: list[str] = []
        if self.GPU_TYPE not in ("B200", "B300", "H200"):
            errors.append(f"Unsupported GPU_TYPE: {self.GPU_TYPE}")
        if self.DEFAULT_PROFILE not in PROFILE_NAMES:
            errors.append(f"Unsupported DEFAULT_PROFILE: {self.DEFAULT_PROFILE}")
        if self.WEIGHT_PRECISION not in ("fp8", "bf16"):
            errors.append(f"Unsupported WEIGHT_PRECISION: {self.WEIGHT_PRECISION}")
        if self.ATTENTION_BACKEND not in ("fp8", "native"):
            errors.append(f"Unsupported ATTENTION_BACKEND: {self.ATTENTION_BACKEND}")
        if not isinstance(self.FPS, int) or self.FPS <= 0:
            errors.append("FPS must be a positive integer")
        if self.FP8_KEEP_EDGE_BLOCKS < 0:
            errors.append("FP8_KEEP_EDGE_BLOCKS must be >= 0")
        if self.ATTN_FP8_MIN_SEQ < 1:
            errors.append("ATTN_FP8_MIN_SEQ must be >= 1")
        if self.GPU_MAX_INPUTS < 1:
            errors.append("GPU_MAX_INPUTS must be >= 1")
        if not (0 < self.GPU_TARGET_INPUTS <= self.GPU_MAX_INPUTS):
            errors.append("GPU_TARGET_INPUTS must be in [1, GPU_MAX_INPUTS]")
        for name in ("QUALITY_STEPS", "FAST_STEPS", "TURBO_STEPS"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                errors.append(f"{name} must be a positive integer")
        if not math.isfinite(self.FAST_CACHE_THRESHOLD) or self.FAST_CACHE_THRESHOLD < 0:
            errors.append("FAST_CACHE_THRESHOLD must be a finite non-negative number")
        if self.CACHE_WARMUP_STEPS < 0 or self.CACHE_TAIL_STEPS < 0:
            errors.append("CACHE_WARMUP_STEPS/CACHE_TAIL_STEPS must be >= 0")
        if self.CACHE_MIN_SKIPPABLE_STEPS < 1:
            errors.append("CACHE_MIN_SKIPPABLE_STEPS must be >= 1")
        if self.LORA_GPU_CACHE_SIZE < 1 or self.LORA_CPU_CACHE_SIZE < 1:
            errors.append("LoRA cache sizes must be >= 1")
        if self.MEDIA_CONCURRENCY < 1:
            errors.append("MEDIA_CONCURRENCY must be >= 1")
        if self.ENCODE_QUEUE_LIMIT < 1:
            errors.append("ENCODE_QUEUE_LIMIT must be >= 1")
        if self.MAX_REFERENCES < 0 or self.MAX_LORAS < 0:
            errors.append("MAX_REFERENCES/MAX_LORAS must be >= 0")
        if not math.isfinite(self.MAX_DURATION_SECONDS) or self.MAX_DURATION_SECONDS <= 0:
            errors.append("MAX_DURATION_SECONDS must be a positive finite number")
        if self.OUTPUT_RETENTION_HOURS < 0:
            errors.append("OUTPUT_RETENTION_HOURS must be >= 0")
        if self.FFMPEG_TIMEOUT <= 0:
            errors.append("FFMPEG_TIMEOUT must be positive")
        if errors:
            raise ValueError("; ".join(errors))

    # Convenience helpers -------------------------------------------------
    def max_frames(self) -> int:
        """Hard upper bound on a single request's frame count."""
        seconds = min(self.MAX_DURATION_SECONDS, 24 * 60 * 60)
        return max(NATIVE_MAX_FRAMES, int(seconds * self.FPS))


ENV_KEYS: list[str] = [f.name for f in fields(Settings)]


def load_settings() -> Settings:
    kwargs = {}
    for f in fields(Settings):
        raw = os.environ.get(f.name)
        if raw is None:
            continue
        try:
            kwargs[f.name] = _coerce_value(f.default, raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid value for {f.name}: {raw!r} ({exc})") from exc
    return Settings(**kwargs)


def settings_env(settings: Settings | None = None) -> dict[str, str]:
    active = settings if settings is not None else load_settings()
    return {k: str(v) for k, v in asdict(active).items()}


def profile_steps(settings: Settings, profile: str) -> int:
    mapping = {
        "quality": settings.QUALITY_STEPS,
        "fast": settings.FAST_STEPS,
        "turbo": settings.TURBO_STEPS,
    }
    try:
        return mapping[profile]
    except KeyError as exc:
        raise ValueError(
            f"Unknown profile {profile!r}; expected one of {sorted(mapping)}"
        ) from exc


def profile_repair_tau(settings: Settings, profile: str) -> float:
    mapping = {
        "quality": settings.QUALITY_REPAIR_TAU,
        "fast": settings.FAST_REPAIR_TAU,
        "turbo": settings.TURBO_REPAIR_TAU,
    }
    try:
        return float(mapping[profile])
    except KeyError as exc:
        raise ValueError(
            f"Unknown profile {profile!r}; expected one of {sorted(mapping)}"
        ) from exc
