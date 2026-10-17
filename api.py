import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from uuid import UUID, uuid4

import modal
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.background import BackgroundTask

from schemas import ErrorBody, GenerationRequest, JobStatus
from shapes import RESOLUTIONS


def create_api(generator_cls, job_store, settings, outputs_volume) -> FastAPI:
    api = FastAPI(title=settings.APP_NAME, version="1.0.0")
    volume_lock = asyncio.Lock()

    def error_response(status, code, message):
        body = ErrorBody(error={"code": code, "message": str(message)})
        return JSONResponse(status_code=status, content=body.model_dump())

    @api.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        messages = [
            f"{'.'.join(map(str, item['loc']))}: {item['msg']}"
            for item in exc.errors()
        ]
        return error_response(400, "validation_error", "; ".join(messages))

    @api.exception_handler(ValueError)
    async def value_error(request: Request, exc: ValueError):
        return error_response(400, "invalid_request", str(exc))

    @api.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        codes = {
            400: "invalid_request",
            404: "not_found",
            409: "not_ready",
            500: "generation_failed",
        }
        return error_response(
            exc.status_code, codes.get(exc.status_code, "http_error"), exc.detail
        )

    @api.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception):
        return error_response(500, "internal_error", "The request could not be completed")

    def canonical_id(value):
        try:
            parsed = UUID(value)
        except (ValueError, AttributeError) as exc:
            raise HTTPException(404, "Job not found") from exc
        if parsed.version != 4 or str(parsed) != value:
            raise HTTPException(404, "Job not found")
        return value

    def status_model(record):
        succeeded = record["status"] == "succeeded"
        return JobStatus(
            id=record["id"],
            status=record["status"],
            created_at=record["created_at"],
            updated_at=record["updated_at"],
            result=record.get("result"),
            video_url=(
                f"/v1/generations/{record['id']}/video" if succeeded else None
            ),
            error=record.get("error"),
        )

    async def read_job(job_id, poll=True):
        canonical_id(job_id)
        record = await job_store.get(job_id)
        if record is None:
            raise HTTPException(404, "Job not found")
        call_id = record.get("call_id")
        if poll and call_id and record["status"] in {"queued", "running"}:
            call = modal.FunctionCall.from_id(call_id)
            try:
                result = await call.get.aio(timeout=0)
            except (TimeoutError, modal.exception.TimeoutError):
                pass
            except Exception as exc:
                latest = await job_store.get(job_id)
                if latest and latest["status"] in {"queued", "running"}:
                    await job_store.mark_failed(job_id, str(exc))
            else:
                if isinstance(result, dict) and "params" in result:
                    latest = await job_store.get(job_id)
                    if latest and latest["status"] in {"queued", "running"}:
                        await job_store.mark_succeeded(
                            job_id, result["params"], result["video_path"]
                        )
            record = await job_store.get(job_id)
            if record is None:
                raise HTTPException(404, "Job not found")
        return record

    @api.get("/health")
    async def health():
        return {"status": "ok"}

    @api.get("/v1/models")
    async def models():
        profiles = ["quality", "fast"]
        if settings.TURBO_TRANSFORMER_REPO:
            profiles.append("turbo")
        return {
            "id": settings.MODEL_REPO_ID,
            "modes": ["t2v", "i2v", "fl2v", "ref2v"],
            "profiles": profiles,
            "profile_modes": {
                profile: (
                    ["t2v", "i2v", "fl2v"]
                    if profile == "turbo"
                    else ["t2v", "i2v", "fl2v", "ref2v"]
                )
                for profile in profiles
            },
            "resolutions": RESOLUTIONS,
            "fps": settings.FPS,
            "frame_grid": {"formula": "17*n + 5", "minimum_n": 1},
            "native_segment_range": {
                "minimum_n": 7,
                "maximum_n": 20,
                "minimum_frames": 124,
                "maximum_frames": 345,
            },
            "default_profile": settings.DEFAULT_PROFILE,
        }

    @api.post("/v1/generations/sync")
    async def generate_sync(request: GenerationRequest):
        try:
            data, params = await generator_cls().generate_bytes.remote.aio(
                request.model_dump_json()
            )
        except ValueError:
            raise
        except Exception as exc:
            raise HTTPException(500, f"Generation failed: {exc}") from exc
        headers = {
            "XGenerationSeed": str(params["seed"]),
            "XWidth": str(params["width"]),
            "XHeight": str(params["height"]),
            "XNumFrames": str(params["num_frames"]),
            "XSteps": str(params["steps"]),
            "XProfile": str(params["profile"]),
        }
        return Response(content=data, media_type="video/mp4", headers=headers)

    @api.post("/v1/generations", response_model=JobStatus, status_code=202)
    async def generate(request: GenerationRequest):
        profile = request.profile or settings.DEFAULT_PROFILE
        if profile == "turbo":
            if not settings.TURBO_TRANSFORMER_REPO:
                raise ValueError("The turbo profile is not configured")
            if request.mode == "ref2v":
                raise ValueError("The turbo profile does not support ref2v")
        if len(request.loras) > settings.LORA_GPU_CACHE_SIZE:
            raise ValueError("The request exceeds LORA_GPU_CACHE_SIZE")

        job_id = str(uuid4())
        await job_store.create(job_id, request.model_dump_json())
        try:
            call = await generator_cls().generate.spawn.aio(
                request.model_dump_json(), job_id
            )
            await job_store.set_call(job_id, call.object_id)
        except Exception as exc:
            await job_store.mark_failed(job_id, str(exc))
            raise HTTPException(500, "The generation could not be scheduled") from exc
        return status_model(await job_store.get(job_id))

    @api.get("/v1/generations/{job_id}", response_model=JobStatus)
    async def get_generation(job_id: str):
        record = await read_job(job_id)
        status = status_model(record)
        if status.status == "failed":
            return JSONResponse(status_code=500, content=status.model_dump())
        return status

    @api.get("/v1/generations/{job_id}/video")
    async def get_video(job_id: str):
        record = await read_job(job_id)
        if record["status"] == "failed":
            raise HTTPException(500, record.get("error") or "Generation failed")
        if record["status"] != "succeeded":
            raise HTTPException(409, "The generation has not completed")
        source = Path(settings.OUTPUTS_DIR) / f"{job_id}.mp4"

        async with volume_lock:
            await outputs_volume.reload.aio()
            if not source.is_file():
                raise HTTPException(404, "The video file does not exist")
            descriptor, name = tempfile.mkstemp(prefix="va-download-", suffix=".mp4")
            os.close(descriptor)
            try:
                await asyncio.to_thread(shutil.copyfile, source, name)
            except BaseException:
                Path(name).unlink(missing_ok=True)
                raise
        return FileResponse(
            name,
            media_type="video/mp4",
            filename=f"{job_id}.mp4",
            background=BackgroundTask(Path(name).unlink, missing_ok=True),
        )

    @api.delete("/v1/generations/{job_id}", status_code=204)
    async def delete_generation(job_id: str):
        record = await read_job(job_id, poll=False)
        await job_store.delete(job_id)
        if record.get("call_id") and record["status"] in {"queued", "running"}:
            try:
                await modal.FunctionCall.from_id(record["call_id"]).cancel.aio()
            except Exception:
                pass
        async with volume_lock:
            await outputs_volume.reload.aio()
            path = Path(settings.OUTPUTS_DIR) / f"{job_id}.mp4"
            path.unlink(missing_ok=True)
            await outputs_volume.commit.aio()
        return Response(status_code=204)

    return api
