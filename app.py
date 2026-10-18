import base64
import json
import mimetypes
import tempfile
from pathlib import Path

import modal

from config import load_settings, settings_env


settings = load_settings()

app = modal.App(settings.APP_NAME)

models_volume = modal.Volume.from_name(
    settings.MODELS_VOLUME, create_if_missing=True
)
outputs_volume = modal.Volume.from_name(
    settings.OUTPUTS_VOLUME, create_if_missing=True
)

_env = {
    **settings_env(settings),
    "HF_HOME": f"{settings.MODELS_DIR}/hf",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
}

_local_sources = (
    "config",
    "shapes",
    "schemas",
    "media_io",
    "api",
    "kernels",
    "runtime",
)

gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.9.*",
        "torchaudio==2.9.*",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install(
        "triton>=3.5",
        "git+https://github.com/huggingface/diffusers.git@main",
        "transformers",
        "accelerate",
        "safetensors",
        "huggingface_hub[hf_transfer]",
        "sentencepiece",
        "protobuf",
        "einops",
        "ftfy",
        "pillow",
        "numpy",
        "httpx",
        "fastapi[standard]",
        "pydantic>=2",
    )
    .env(_env)
    .add_local_python_source(*_local_sources)
)

api_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]", "pydantic>=2", "httpx")
    .env(_env)
    .add_local_python_source(*_local_sources)
)


@app.function(
    image=gpu_image,
    volumes={settings.MODELS_DIR: models_volume},
    timeout=settings.FUNCTION_TIMEOUT,
)
def download_weights():
    from runtime.weights import commit_if_changed, ensure_weights

    paths = ensure_weights(settings)
    commit_if_changed(models_volume)
    return {k: str(v) for k, v in paths.items()}


@app.cls(
    gpu=settings.GPU_TYPE,
    image=gpu_image,
    volumes={
        settings.MODELS_DIR: models_volume,
        settings.OUTPUTS_DIR: outputs_volume,
    },
    timeout=settings.FUNCTION_TIMEOUT,
    scaledown_window=settings.SCALEDOWN_WINDOW,
    min_containers=settings.MIN_CONTAINERS,
    memory=settings.HOST_MEMORY_MIB,
    cpu=settings.CPU_CORES,
)
@modal.concurrent(
    max_inputs=settings.GPU_MAX_INPUTS,
    target_inputs=settings.GPU_TARGET_INPUTS,
)
class VideoAudioGenerator:
    @modal.enter()
    def start(self):
        from runtime.generator import Generator
        from runtime.jobs import JobStore
        from runtime.model_manager import ModelManager
        from runtime.weights import commit_if_changed

        self.manager = ModelManager(settings)
        self.manager.load()
        commit_if_changed(models_volume)
        self.generator = Generator(self.manager, settings)
        self.job_store = JobStore(settings)

    @modal.method()
    async def generate(self, request_json: str, job_id: str):
        from schemas import GenerationRequest

        await self.job_store.mark_running(job_id)
        out_path = Path(settings.OUTPUTS_DIR) / f"{job_id}.mp4"
        try:
            req = GenerationRequest.model_validate_json(request_json)
            params = await self.generator.render_mp4(req, out_path)
            outputs_volume.commit()
            await self.job_store.mark_succeeded(job_id, params, str(out_path))
            return params.model_dump()
        except Exception as exc:
            out_path.unlink(missing_ok=True)
            await self.job_store.mark_failed(job_id, str(exc))
            raise

    @modal.method()
    async def generate_bytes(self, request_json: str):
        from schemas import GenerationRequest

        req = GenerationRequest.model_validate_json(request_json)
        with tempfile.TemporaryDirectory(prefix="va-bytes-") as tmpdir:
            out_path = Path(tmpdir) / "output.mp4"
            params = await self.generator.render_mp4(req, out_path)
            mp4_bytes = out_path.read_bytes()
            return mp4_bytes, params.model_dump()


@app.function(
    image=api_image,
    volumes={settings.OUTPUTS_DIR: outputs_volume},
)
@modal.concurrent(max_inputs=200)
@modal.asgi_app()
def web():
    from api import create_api
    from runtime.jobs import JobStore

    return create_api(
        VideoAudioGenerator,
        JobStore(settings),
        settings,
        outputs_volume,
    )


def _encode_media_arg(value: str) -> str:
    if value.startswith(("http://", "https://", "data:")):
        return value
    p = Path(value).expanduser()
    mime, _ = mimetypes.guess_type(p.name)
    mime = mime or "application/octet-stream"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


@app.local_entrypoint()
def main(
    prompt: str,
    out: str = "output.mp4",
    mode: str = "",
    resolution: str = "768p",
    aspect_ratio: str = "16:9",
    duration_seconds: float = 0.0,
    num_frames: int = 0,
    profile: str = "",
    steps: int = 0,
    seed: int = -1,
    first_frame: str = "",
    last_frame: str = "",
    references: str = "[]",
    loras: str = "[]",
    include_audio: bool = True,
):
    from schemas import GenerationRequest

    ref_list = json.loads(references)
    for item in ref_list:
        if "source" in item:
            item["source"] = _encode_media_arg(item["source"])

    req = GenerationRequest(
        prompt=prompt,
        mode=mode or None,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        duration_seconds=duration_seconds if duration_seconds > 0 else None,
        num_frames=num_frames if num_frames > 0 else None,
        profile=profile or None,
        num_inference_steps=steps if steps > 0 else None,
        seed=seed if seed >= 0 else None,
        first_frame={"source": _encode_media_arg(first_frame)} if first_frame else None,
        last_frame={"source": _encode_media_arg(last_frame)} if last_frame else None,
        references=ref_list,
        loras=json.loads(loras),
        include_audio=include_audio,
    )
    mp4_bytes, params = VideoAudioGenerator().generate_bytes.remote(
        req.model_dump_json()
    )
    out_path = Path(out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(mp4_bytes)
    print(json.dumps(params, indent=2))
