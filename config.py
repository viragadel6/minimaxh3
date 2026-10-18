import os
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class Settings:
    APP_NAME: str = "video-audio-diffusion-server"
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
    MODELS_VOLUME: str = "va-models"
    OUTPUTS_VOLUME: str = "va-outputs"
    JOBS_DICT: str = "va-jobs"
    MODELS_DIR: str = "/models"
    OUTPUTS_DIR: str = "/outputs"
    FPS: int = 24
    X264_PRESET: str = "veryfast"
    X264_CRF: int = 17
    AUDIO_BITRATE: str = "192k"

    def __post_init__(self) -> None:
        if self.GPU_TYPE not in ("B200", "B300", "H200"):
            raise ValueError(f"Unsupported GPU_TYPE: {self.GPU_TYPE}")
        if self.DEFAULT_PROFILE not in ("quality", "fast", "turbo"):
            raise ValueError(f"Unsupported DEFAULT_PROFILE: {self.DEFAULT_PROFILE}")
        if self.WEIGHT_PRECISION not in ("fp8", "bf16"):
            raise ValueError(f"Unsupported WEIGHT_PRECISION: {self.WEIGHT_PRECISION}")
        if self.ATTENTION_BACKEND not in ("fp8", "native"):
            raise ValueError(f"Unsupported ATTENTION_BACKEND: {self.ATTENTION_BACKEND}")


ENV_KEYS: list[str] = [f.name for f in fields(Settings)]


def load_settings() -> Settings:
    kwargs = {}
    for f in fields(Settings):
        raw = os.environ.get(f.name)
        if raw is not None:
            target_type = type(f.default)
            kwargs[f.name] = target_type(raw)
    return Settings(**kwargs)


def settings_env(settings: Settings | None = None) -> dict[str, str]:
    active = settings if settings is not None else load_settings()
    return {k: str(v) for k, v in asdict(active).items()}
