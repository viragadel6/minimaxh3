import asyncio
import inspect
from pathlib import Path
from uuid import uuid4

import modal
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response

from schemas import ErrorBody, GenerationRequest, JobStatus, ResolvedParams
from shapes import NATIVE_MAX_FRAMES, NATIVE_MAX_N, NATIVE_MIN_FRAMES, NATIVE_MIN_N, RESOLUTIONS


async def _call_maybe_async(fn, *args, **kwargs):
    aio = getattr(fn, "aio", None)
    if callable(aio):
        return await aio(*args, **kwargs)
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    res = await asyncio.to_thread(fn, *args, **kwargs)
    if inspect.isawaitable(res):
        return await res
    return res


def create_api(generator_cls, job_store, settings, outputs_volume) -> FastAPI:
    api = FastAPI(title=settings.APP_NAME)

    def _error(status_code: int, code: str, message: str) -> JSONResponse:
        body = ErrorBody(error={"code": code, "message": message})
        return JSONResponse(status_code=status_code, content=body.model_dump())

    @api.exception_handler(RequestValidationError)
    async def _handle_validation_error(request: Request, exc: RequestValidationError):
        detail = "; ".join(
            f"{'.'.join(str(p) for p in err.get('loc', ()))}: {err.get('msg', '')}"
            for err in exc.errors()
        )
        return _error(400, "validation_error", detail or str(exc))

    @api.exception_handler(ValueError)
    async def _handle_value_error(request: Request, exc: ValueError):
        return _error(400, "invalid_request", str(exc))

    @api.exception_handler(HTTPException)
    async def _handle_http_exception(request: Request, exc: HTTPException):
        code_map = {
            400: "invalid_request",
            404: "not_found",
            409: "conflict",
            500: "generation_failed",
        }
        return _error(
            exc.status_code,
            code_map.get(exc.status_code, "error"),
            str(exc.detail),
        )

    @api.exception_handler(Exception)
    async def _handle_generic_exception(request: Request, exc: Exception):
        return _error(500, "generation_failed", str(exc))

    def _to_job_status(record: dict) -> JobStatus:
        vid_id = record["id"]
        succeeded = record["status"] == "succeeded"
        res = record.get("result")
        if isinstance(res, dict):
            res = ResolvedParams(**res)
        return JobStatus(
            id=vid_id,
            status=record["status"],
            created_at=record["created_at"],
            updated_at=record["updated_at"],
            result=res,
            video_url=f"/v1/generations/{vid_id}/video" if succeeded else None,
            error=record.get("error"),
        )

    async def _sync_job(job_id: str) -> dict:
        record = await job_store.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        cid = record.get("call_id")
        if cid and record["status"] in ("queued", "running"):
            try:
                fc = modal.FunctionCall.from_id(cid)
                outcome = await _call_maybe_async(fc.get, timeout=0)
            except TimeoutError:
                return record
            except Exception as exc:
                if type(exc).__name__ == "TimeoutError":
                    return record
                await job_store.mark_failed(job_id, str(exc))
                record = await job_store.get(job_id) or record
                return record
            else:
                latest = await job_store.get(job_id)
                if latest and latest["status"] in ("queued", "running"):
                    params_data = (
                        outcome["params"]
                        if isinstance(outcome, dict) and "params" in outcome
                        else outcome
                    )
                    video_path = str(Path(settings.OUTPUTS_DIR) / f"{job_id}.mp4")
                    await job_store.mark_succeeded(job_id, params_data, video_path)
                record = await job_store.get(job_id) or record
        return record

    @api.get("/health")
    async def health():
        return {"status": "ok"}

    @api.get("/v1/models")
    async def list_models():
        profiles = ["quality", "fast"]
        if settings.TURBO_TRANSFORMER_REPO:
            profiles.append("turbo")
        return {
            "model_repo_id": settings.MODEL_REPO_ID,
            "modes": ["t2v", "i2v", "fl2v", "ref2v"],
            "profiles": profiles,
            "default_profile": settings.DEFAULT_PROFILE,
            "resolutions": RESOLUTIONS,
            "fps": settings.FPS,
            "frame_grid": {"formula": "17n + 5", "min_n": 1},
            "native_segment_bounds": {
                "min_n": NATIVE_MIN_N,
                "max_n": NATIVE_MAX_N,
                "min_frames": NATIVE_MIN_FRAMES,
                "max_frames": NATIVE_MAX_FRAMES,
            },
        }

    @api.post("/v1/generations", response_model=JobStatus, status_code=202)
    async def create_generation(req: GenerationRequest):
        profile = req.profile or settings.DEFAULT_PROFILE
        if profile == "turbo":
            if not settings.TURBO_TRANSFORMER_REPO:
                raise ValueError("Turbo profile is not available without TURBO_TRANSFORMER_REPO")
            if req.mode == "ref2v":
                raise ValueError("Turbo profile is incompatible with ref2v mode")

        job_id = str(uuid4())
        request_json = req.model_dump_json()
        await job_store.create(job_id, request_json)
        try:
            call = await _call_maybe_async(
                generator_cls().generate.spawn, request_json, job_id
            )
            call_id = getattr(call, "object_id", str(call))
            await job_store.set_call(job_id, call_id)
        except Exception as exc:
            await job_store.mark_failed(job_id, str(exc))
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        record = await job_store.get(job_id)
        return _to_job_status(record)

    @api.post("/v1/generations/sync")
    async def create_generation_sync(req: GenerationRequest):
        request_json = req.model_dump_json()
        try:
            mp4_bytes, params = await _call_maybe_async(
                generator_cls().generate_bytes.remote, request_json
            )
        except ValueError:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if isinstance(params, ResolvedParams):
            params = params.model_dump()
        headers = {
            "X-Generation-Seed": str(params["seed"]),
            "X-Width": str(params["width"]),
            "X-Height": str(params["height"]),
            "X-Num-Frames": str(params["num_frames"]),
            "X-Steps": str(params["steps"]),
            "X-Profile": str(params["profile"]),
        }
        return Response(content=mp4_bytes, media_type="video/mp4", headers=headers)

    @api.get("/v1/generations/{id}", response_model=JobStatus)
    async def get_generation(id: str):
        record = await _sync_job(id)
        return _to_job_status(record)

    @api.get("/v1/generations/{id}/video")
    async def get_generation_video(id: str):
        record = await _sync_job(id)
        if record["status"] == "failed":
            raise HTTPException(
                status_code=500,
                detail=record.get("error") or "Generation failed",
            )
        if record["status"] != "succeeded":
            raise HTTPException(status_code=409, detail="Job is not yet complete")

        await _call_maybe_async(outputs_volume.reload)
        video_path = Path(record.get("video_path") or (Path(settings.OUTPUTS_DIR) / f"{id}.mp4"))
        if not video_path.is_file():
            raise HTTPException(status_code=404, detail="Video file not found")
        return FileResponse(
            path=str(video_path),
            media_type="video/mp4",
            filename=f"{id}.mp4",
        )

    @api.delete("/v1/generations/{id}", status_code=204)
    async def delete_generation(id: str):
        record = await job_store.get(id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Job {id} not found")
        await _call_maybe_async(outputs_volume.reload)
        video_path = Path(record.get("video_path") or (Path(settings.OUTPUTS_DIR) / f"{id}.mp4"))
        if video_path.exists():
            video_path.unlink(missing_ok=True)
        await job_store.delete(id)
        await _call_maybe_async(outputs_volume.commit)
        return Response(status_code=204)

    return api
