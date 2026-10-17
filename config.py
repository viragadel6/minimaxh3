import math
import os
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True, slots=True)
class Settings:
    APP_NAME: str = "videoaudiodiffusionserver"
    GPU_TYPE: str = "B200"
    MODEL_REPO_ID: str = "MiniMaxAI/MiniMaxH3"
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
    QUALITY_REPAIR_TAU: float = 0.5
    FAST_REPAIR_TAU: float = 1.0
    TURBO_REPAIR_TAU: float = 1.0
    DEFAULT_PROFILE: str = "fast"
    WEIGHT_PRECISION: str = "fp8"
    FP8_EXCLUDE_PATTERNS: str = "embed,norm,adaln,modulation,time,proj_out,final,head"
    FP8_KEEP_EDGE_BLOCKS: int = 2
    ATTENTION_BACKEND: str = "fp8"
    ATTN_FP8_MIN_SEQ: int = 4096
    COMPILE_MODE: str = "maxautotunenocudagraphs"
    VAE_TILING: int = 0
    RESIDENCY_RESERVE_GIB: float = 40.0
    LORA_GPU_CACHE_SIZE: int = 8
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
    MODELS_VOLUME: str = "vamodels"
    OUTPUTS_VOLUME: str = "vaoutputs"
    JOBS_DICT: str = "vajobs"
    MODELS_DIR: str = "/models"
    OUTPUTS_DIR: str = "/outputs"
    FPS: int = 24
    X264_PRESET: str = "veryfast"
    X264_CRF: int = 17
    AUDIO_BITRATE: str = "192k"

    def __post_init__(self):
        choices = {
            "GPU_TYPE": {"B200", "B300", "H200"},
            "DEFAULT_PROFILE": {"quality", "fast", "turbo"},
            "WEIGHT_PRECISION": {"bf16", "fp8"},
            "ATTENTION_BACKEND": {"native", "fp8"},
            "VAE_TILING": {0, 1},
            "X264_PRESET": {
                "ultrafast", "superfast", "veryfast", "faster", "fast",
                "medium", "slow", "slower", "veryslow",
            },
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}")

        positive = (
            "TURBO_STEPS", "QUALITY_STEPS", "FAST_STEPS",
            "LORA_GPU_CACHE_SIZE", "GPU_MAX_INPUTS", "GPU_TARGET_INPUTS",
            "HOST_MEMORY_MIB", "CPU_CORES", "FUNCTION_TIMEOUT",
            "SCALEDOWN_WINDOW",
        )
        nonnegative = (
            "FAST_CACHE_THRESHOLD", "CACHE_WARMUP_STEPS",
            "CACHE_TAIL_STEPS", "FP8_KEEP_EDGE_BLOCKS",
            "ATTN_FP8_MIN_SEQ", "RESIDENCY_RESERVE_GIB",
            "WARMUP_STEPS", "MIN_CONTAINERS",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in nonnegative:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{item.name} must be finite")
        if self.FPS != 24:
            raise ValueError("MiniMaxH3 frame planning requires FPS=24")
        if self.GPU_TARGET_INPUTS > self.GPU_MAX_INPUTS:
            raise ValueError("GPU_TARGET_INPUTS must not exceed GPU_MAX_INPUTS")
        if not 0 <= self.X264_CRF <= 51:
            raise ValueError("X264_CRF must be between 0 and 51")
        if self.DEFAULT_PROFILE == "turbo" and not self.TURBO_TRANSFORMER_REPO:
            raise ValueError("The turbo default requires TURBO_TRANSFORMER_REPO")
        if not os.path.isabs(self.MODELS_DIR) or not os.path.isabs(self.OUTPUTS_DIR):
            raise ValueError("MODELS_DIR and OUTPUTS_DIR must be absolute")
        if self.MODELS_DIR == self.OUTPUTS_DIR:
            raise ValueError("Model and output volumes must have different mount points")


ENV_KEYS = [item.name for item in fields(Settings)]

assert len(ENV_KEYS) == 46


def load_settings() -> Settings:
    values = {}
    for item in fields(Settings):
        raw = os.environ.get(item.name)
        if raw is None:
            continue
        try:
            values[item.name] = type(item.default)(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid environment value for {item.name}") from exc
    return Settings(**values)


def settings_env(settings: Settings | None = None) -> dict[str, str]:
    effective = settings if settings is not None else load_settings()
    result = {}
    for name, value in asdict(effective).items():
        if isinstance(value, float) and value.is_integer():
            result[name] = str(int(value))
        else:
            result[name] = str(value)
    return result
