import asyncio
import base64
import json
import logging
import mimetypes
import tempfile
from pathlib import Path
from uuid import UUID

import modal

from config import load_settings, settings_env


settings = load_settings()

UPSTREAM_REQUIREMENT = (
    "This deployment requires the real MiniMaxH3 checkpoints and a diffusers build "
    "exporting the configured ModularPipeline blocks and reference constructors. "
    "Availability, numerical quality, and GPU performance must be validated against "
    "that upstream implementation before production rollout."
)

app = modal.App(settings.APP_NAME)

models_volume = modal.Volume.from_name(
    settings.MODELS_VOLUME, create_if_missing=True
)
outputs_volume = modal.Volume.from_name(
    settings.OUTPUTS_VOLUME, create_if_missing=True
)

environment = {
    **settings_env(settings),
    "HF_HOME": f"{settings.MODELS_DIR}/hf",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
}

source_modules = (
    "config", "shapes", "schemas", "media_io", "api", "kernels", "runtime"
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
    .env(environment)
    .add_local_python_source(*source_modules)
)

api_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]", "pydantic>=2", "httpx")
    .env(environment)
    .add_local_python_source(*source_modules)
)


@app.function(
    image=gpu_image,
    volumes={settings.MODELS_DIR: models_volume},
    timeout=settings.FUNCTION_TIMEOUT,
    memory=settings.HOST_MEMORY_MIB,
    cpu=settings.CPU_CORES,
)
def download_weights():
    from runtime.weights import commit_if_changed, ensure_weights

    paths = ensure_weights(settings)
    commit_if_changed(models_volume)
    return {name: str(path) for name, path in paths.items()}


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

        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).warning(UPSTREAM_REQUIREMENT)
        self.manager = ModelManager(settings)
        self.manager.load()
        commit_if_changed(models_volume)
        self.engine = Generator(self.manager, settings)
        self.jobs = JobStore(settings)

    @modal.exit()
    def stop(self):
        if hasattr(self, "engine"):
            self.engine.close()

    @modal.method()
    async def generate(self, request_json, job_id):
        from runtime.weights import commit_if_changed
        from schemas import GenerationRequest

        parsed = UUID(job_id)
        if parsed.version != 4 or str(parsed) != job_id:
            raise ValueError("job_id must be a canonical UUID4")
        request = GenerationRequest.model_validate_json(request_json)
        output = Path(settings.OUTPUTS_DIR) / f"{job_id}.mp4"
        if await self.jobs.get(job_id) is None:
            return {"cancelled": True}
        await self.jobs.mark_running(job_id)

        try:
            params = await self.engine.render_mp4(request, output)
            await asyncio.to_thread(commit_if_changed, models_volume)
            if await self.jobs.get(job_id) is None:
                output.unlink(missing_ok=True)
                await outputs_volume.commit.aio()
                return {"cancelled": True}
            await outputs_volume.commit.aio()
            saved = await self.jobs.mark_succeeded(job_id, params, output)
            if not saved:
                output.unlink(missing_ok=True)
                await outputs_volume.commit.aio()
                return {"cancelled": True}
            return {"params": params.model_dump(), "video_path": str(output)}
        except asyncio.CancelledError:
            output.unlink(missing_ok=True)
            try:
                await asyncio.shield(self.jobs.mark_failed(job_id, "Generation cancelled"))
                await asyncio.shield(outputs_volume.commit.aio())
            except Exception:
                pass
            raise
        except Exception as exc:
            output.unlink(missing_ok=True)
            try:
                await self.jobs.mark_failed(job_id, str(exc))
                await outputs_volume.commit.aio()
            except Exception:
                logging.getLogger(__name__).exception("Could not persist job failure")
            raise

    @modal.method()
    async def generate_bytes(self, request_json):
        from runtime.weights import commit_if_changed
        from schemas import GenerationRequest

        request = GenerationRequest.model_validate_json(request_json)
        with tempfile.TemporaryDirectory(prefix="va-sync-") as directory:
            output = Path(directory) / "video.mp4"
            params = await self.engine.render_mp4(request, output)
            data = await asyncio.to_thread(output.read_bytes)
            await asyncio.to_thread(commit_if_changed, models_volume)
            return data, params.model_dump()


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


def _cli_source(value):
    if value.startswith(("http://", "https://", "data:")):
        return value
    path = Path(value).expanduser()
    if not path.is_file():
        raise ValueError(f"Local media file does not exist: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


@app.local_entrypoint()
def main(
    prompt: str,
    output: str = "output.mp4",
    mode: str = "",
    resolution: str = "768p",
    aspect_ratio: str = "16:9",
    duration_seconds: float = 5.0,
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

    reference_values = json.loads(references)
    if not isinstance(reference_values, list):
        raise ValueError("references must be a JSON array")
    for item in reference_values:
        item["source"] = _cli_source(item["source"])

    request = GenerationRequest(
        prompt=prompt,
        mode=mode or None,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        duration_seconds=None if num_frames else duration_seconds,
        num_frames=num_frames or None,
        profile=profile or None,
        num_inference_steps=steps or None,
        seed=None if seed < 0 else seed,
        first_frame={"source": _cli_source(first_frame)} if first_frame else None,
        last_frame={"source": _cli_source(last_frame)} if last_frame else None,
        references=reference_values,
        loras=json.loads(loras),
        include_audio=include_audio,
    )
    data, params = VideoAudioGenerator().generate_bytes.remote(
        request.model_dump_json()
    )
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    print(json.dumps(
        {"output": str(destination), "params": params},
        ensure_ascii=False,
        indent=2,
    ))
