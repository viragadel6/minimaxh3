"""Modal deployment: GPU worker, weight downloader and HTTP API.

Image build notes
-----------------
* ``diffusers`` is pinned (``DIFFUSERS_SPEC``) instead of tracking ``git@main`` so an
  image rebuild is reproducible.
* Triton is *not* installed explicitly: the PyTorch wheel pins a matching version and
  overriding it can break the ABI.
* Application code is published under ``va_kernels`` (not ``kernels``) so it can never
  shadow the PyPI ``kernels`` package imported by diffusers/transformers.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import mimetypes
import os
import shutil
import tempfile
import time
from pathlib import Path

import modal

from config import load_settings, settings_env

settings = load_settings()

app = modal.App(settings.APP_NAME)

models_volume = modal.Volume.from_name(settings.MODELS_VOLUME, create_if_missing=True)
outputs_volume = modal.Volume.from_name(settings.OUTPUTS_VOLUME, create_if_missing=True)

_gpu_env = {
    **settings_env(settings),
    "HF_HOME": f"{settings.MODELS_DIR}/hf",
}
# The API container never loads weights: it must not advertise a HF cache that is not
# mounted, and it needs no GPU side environment at all.
_api_env = {**settings_env(settings), "HF_HOME": "/tmp/hf"}

_gpu_sources = ("config", "shapes", "schemas", "media_io", "va_kernels", "runtime")
# ``runtime.generator``/``media_io`` require torch, but ``web()`` only imports
# ``runtime.jobs`` (modal + stdlib), so the API image deliberately ships fewer sources.
_api_sources = ("config", "shapes", "schemas", "api", "runtime")

gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.9.*",
        "torchaudio==2.9.*",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install(
        settings.DIFFUSERS_SPEC,
        "modal",
        "transformers",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
        "einops",
        "ftfy",
        "pillow",
        "numpy",
        "httpx",
        "pydantic>=2",
    )
    .env(_gpu_env)
    .add_local_python_source(*_gpu_sources)
)

api_image = (
    modal.Image.debian_slim(python_version="3.11")
    # ``runtime.jobs`` talks to a Modal Dict, so the API container needs ``modal`` too.
    .pip_install("modal", "fastapi[standard]", "pydantic>=2", "httpx")
    .env(_api_env)
    .add_local_python_source(*_api_sources)
)


# ---------------------------------------------------------------------------
# volume helpers
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _volume_lock(name: str, timeout: float = 1800.0):
    """Cross-container exclusive lock backed by the shared models volume."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non POSIX hosts
        yield
        return
    lock_path = Path(settings.MODELS_DIR) / f".{name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("w")
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"timed out waiting for the {name} volume lock"
                    ) from None
                time.sleep(1.0)
        yield
    finally:
        with contextlib.suppress(Exception):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _commit_volume(volume) -> None:
    if volume is not None and hasattr(volume, "commit"):
        volume.commit()


def _prune_outputs() -> int:
    """Delete rendered videos older than ``OUTPUT_RETENTION_HOURS``."""
    hours = float(settings.OUTPUT_RETENTION_HOURS)
    if hours <= 0:
        return 0
    cutoff = time.time() - hours * 3600
    removed = 0
    for path in Path(settings.OUTPUTS_DIR).glob("*.mp4"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
                removed += 1
        except OSError:
            continue
    return removed


@app.function(
    image=gpu_image,
    volumes={settings.MODELS_DIR: models_volume},
    timeout=settings.DOWNLOAD_TIMEOUT,
)
def download_weights():
    from runtime.weights import commit_if_changed, ensure_weights

    with _volume_lock("weights"):
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

        # Only one cold-starting container may download/commit the shared volume.
        with _volume_lock("model-load"):
            self.manager = ModelManager(settings)
            self.manager.load()
            commit_if_changed(models_volume)
        self.generator = Generator(self.manager, settings)
        self.job_store = JobStore(settings)
        self._publish_lock = asyncio.Lock()
        self._run_lock = asyncio.Lock()

    @modal.exit()
    def stop(self):
        generator = getattr(self, "generator", None)
        if generator is not None:
            generator.shutdown_nowait()

    # -- internals -------------------------------------------------------
    def _staging_path(self, job_id: str) -> Path:
        base = settings.OUTPUT_STAGING_DIR or tempfile.gettempdir()
        directory = Path(base) / "va-staging" / job_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory / "output.mp4"

    async def _publish(self, staged: Path, final: Path) -> None:
        """Move a finished render onto the volume and commit, off the event loop."""
        loop = asyncio.get_running_loop()

        def _publish_sync() -> None:
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, final)
            # Rendering happened outside the volume, so a commit can only ever see
            # finished files - never another request's partial render.
            _commit_volume(outputs_volume)
            removed = _prune_outputs()
            if removed:
                _commit_volume(outputs_volume)

        async with self._publish_lock:
            await loop.run_in_executor(None, _publish_sync)

    async def _cleanup(self, job_id: str, staging: Path, final: Path) -> None:
        loop = asyncio.get_running_loop()

        def _unlink_all() -> None:
            shutil.rmtree(staging.parent, ignore_errors=True)
            try:
                final.unlink(missing_ok=True)
                _commit_volume(outputs_volume)
            except OSError:
                pass

        with contextlib.suppress(Exception):
            await loop.run_in_executor(None, _unlink_all)
        with contextlib.suppress(Exception):
            await self.job_store.mark_failed(job_id, "generation failed")

    # -- methods ---------------------------------------------------------
    @modal.method()
    async def generate(self, request_json: str, job_id: str):
        from schemas import GenerationRequest

        await self.job_store.mark_running(job_id)
        final_path = Path(settings.OUTPUTS_DIR) / f"{job_id}.mp4"
        staged_path = self._staging_path(job_id)
        try:
            req = GenerationRequest.model_validate_json(request_json)
            async with self._run_lock:
                params = await self.generator.render_mp4(req, staged_path)
            await self._publish(staged_path, final_path)
            updated = await self.job_store.mark_succeeded(
                job_id, params, str(final_path)
            )
            if updated is None:
                raise RuntimeError(f"job {job_id} disappeared before it was recorded")
            # The API expects this shape; ``params`` is validated against the schema
            # on the polling side so a corrupted result can never be reported.
            return {"params": params.model_dump(), "video_path": str(final_path)}
        except BaseException as exc:  # includes CancelledError / worker timeouts
            await self._cleanup(job_id, staged_path, final_path)
            raise exc

    @modal.method()
    async def generate_bytes(self, request_json: str):
        from schemas import GenerationRequest

        req = GenerationRequest.model_validate_json(request_json)
        loop = asyncio.get_running_loop()
        with tempfile.TemporaryDirectory(prefix="va-bytes-") as tmpdir:
            out_path = Path(tmpdir) / "output.mp4"
            async with self._run_lock:
                params = await self.generator.render_mp4(req, out_path)
            mp4_bytes = await loop.run_in_executor(None, out_path.read_bytes)
            if len(mp4_bytes) > int(settings.MAX_SYNC_BYTES):
                raise ValueError(
                    "rendered video exceeds MAX_SYNC_BYTES; use the asynchronous API"
                )
            return mp4_bytes, params.model_dump()


@app.function(
    image=api_image,
    volumes={settings.OUTPUTS_DIR: outputs_volume},
    timeout=settings.WEB_TIMEOUT,
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


_SCHEMES = ("http", "https", "data")


def _encode_media_arg(value: str) -> str:
    """Inline a local media file as a data URI (schemes are case insensitive)."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("media path must be a non-empty string")
    head, sep, _ = value.partition(":")
    if sep and head.strip().lower() in _SCHEMES:
        return value
    path = Path(value).expanduser()
    if not path.is_file():
        raise ValueError(f"media file not found: {value}")
    mime, encoding = mimetypes.guess_type(path.name)
    if encoding is not None:
        raise ValueError(
            f"compressed media files are not supported ({encoding}): {value}"
        )
    mime = mime or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _parse_references(raw: str) -> list[dict]:
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("--references must be a JSON list")
    references: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"reference entries must be objects, got {type(item).__name__}")
        if "source" in item:
            item = dict(item)
            item["source"] = _encode_media_arg(item["source"])
        references.append(item)
    return references


@app.local_entrypoint()
def main(
    prompt: str,
    out: str = "output.mp4",
    mode: str = "",
    resolution: str = "768p",
    aspect_ratio: str = "16:9",
    duration_seconds: float | None = None,
    num_frames: int | None = None,
    profile: str = "",
    steps: int | None = None,
    seed: int | None = None,
    first_frame: str = "",
    last_frame: str = "",
    references: str = "[]",
    loras: str = "[]",
    include_audio: bool = True,
):
    from schemas import GenerationRequest

    # Negative/invalid values are deliberately passed through so the schema raises a
    # descriptive validation error instead of being silently dropped.
    req = GenerationRequest(
        prompt=prompt,
        mode=mode or None,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
        duration_seconds=duration_seconds,
        num_frames=num_frames,
        profile=profile or None,
        num_inference_steps=steps,
        seed=seed,
        first_frame={"source": _encode_media_arg(first_frame)} if first_frame else None,
        last_frame={"source": _encode_media_arg(last_frame)} if last_frame else None,
        references=_parse_references(references),
        loras=json.loads(loras),
        include_audio=include_audio,
    )
    mp4_bytes, params = VideoAudioGenerator().generate_bytes.remote(
        req.model_dump_json()
    )

    out_path = Path(out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replace: an interrupted run can never truncate an existing file.
    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    tmp_path.write_bytes(mp4_bytes)
    os.replace(tmp_path, out_path)
    print(json.dumps(params, indent=2, default=str))
