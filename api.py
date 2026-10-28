"""HTTP API for the video + audio generation service.

Design notes
------------
* Routes are registered in a fixed order, static paths first, so ``/v1/generations/sync``
  can never be captured by ``/v1/generations/{id}``.
* Only :class:`ApiError` produces a client visible message.  Every other exception is
  logged and rendered as an opaque ``500`` so internal details never leak.
* Validation failures are reported as ``400`` (FastAPI's default is ``422``) and the
  generated OpenAPI document is rewritten to match, so the published contract and the
  actual behaviour cannot drift apart.
* All blocking work (Modal RPCs, volume commits, file IO) is pushed off the event loop;
  nothing in the request path calls a synchronous network or disk API directly.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import tempfile
import threading
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import modal
from fastapi import FastAPI, Path as PathParameter, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import ValidationError
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException as StarletteHTTPException

from runtime.jobs import TERMINAL_STATES
from schemas import (
    ASPECT_RATIOS,
    JOB_STATES,
    MODES,
    PROFILES,
    RESOLUTION_NAMES,
    ErrorBody,
    ErrorDetail,
    GenerationRequest,
    JobStatus,
    ModelList,
    ResolvedParams,
)
from shapes import NATIVE_MAX_FRAMES, NATIVE_MAX_N, NATIVE_MIN_FRAMES, NATIVE_MIN_N, RESOLUTIONS

logger = logging.getLogger(__name__)

_STREAM_CHUNK = 1024 * 1024
#: Extra time granted past ``FUNCTION_TIMEOUT`` before a stuck job is failed.
_POLL_GRACE_SECONDS = 300.0

_STATUS_CODES: dict[int, str] = {
    400: "invalid_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    410: "gone",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "too_many_requests",
    500: "internal_error",
    502: "upstream_error",
    503: "service_unavailable",
    504: "upstream_timeout",
}


class ApiError(Exception):
    """An error that is safe to show to the caller."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.code = str(code)
        self.message = str(message)
        self.headers = headers


def _code_for_status(status_code: int) -> str:
    return _STATUS_CODES.get(int(status_code), "http_error")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _error_response(
    status_code: int,
    code: str | None,
    message: Any,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Render an error body, degrading gracefully if the payload cannot be modelled."""
    text = str(message) if message is not None else ""
    if not text.strip():
        text = _code_for_status(status_code).replace("_", " ")
    try:
        body = ErrorBody(
            error=ErrorDetail(code=str(code or _code_for_status(status_code)), message=text)
        ).model_dump(mode="json")
    except (ValidationError, TypeError, ValueError):  # pragma: no cover - defensive
        body = {
            "error": {
                "code": str(code or _code_for_status(status_code)),
                "message": text[:500] or "request failed",
            }
        }
    return JSONResponse(status_code=int(status_code), content=body, headers=headers)


async def _call_maybe_async(
    fn: Callable[..., Any],
    *args: Any,
    timeout: float | None = None,
    use_thread: bool = True,
    **kwargs: Any,
) -> Any:
    """Call ``fn`` whether it is sync, async, or exposes Modal's ``.aio`` variant.

    ``.aio`` is preferred when present because it keeps Modal RPCs on the event loop's
    own machinery instead of burning a worker thread per in-flight request.
    """

    async def _invoke() -> Any:
        aio = getattr(fn, "aio", None)
        if callable(aio):
            result = aio(*args, **kwargs)
            if not inspect.isawaitable(result):
                raise TypeError(
                    f"{getattr(fn, '__name__', fn)!r}.aio() did not return an awaitable"
                )
            return await result
        if inspect.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        result = await asyncio.to_thread(fn, *args, **kwargs) if use_thread else fn(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    if timeout is None:
        return await _invoke()
    return await asyncio.wait_for(_invoke(), timeout=float(timeout))


def _parse_timestamp(value: Any, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    else:
        raise ValueError(f"missing or malformed {field}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _unlink_if_present(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        logger.warning("failed to remove %s", path, exc_info=True)
        return False


def _close_quietly(handle) -> None:
    try:
        handle.close()
    except OSError:  # pragma: no cover - best effort
        pass


async def _iter_file(handle, chunk_size: int = _STREAM_CHUNK) -> AsyncIterator[bytes]:
    """Yield an already-open file handle in chunks without blocking the loop."""
    while True:
        chunk = await asyncio.to_thread(handle.read, chunk_size)
        if not chunk:
            break
        yield chunk


class _StreamTracker:
    """Counts in-flight downloads so the outputs volume is not reloaded under them."""

    def __init__(self) -> None:
        self._count = 0
        self._lock = threading.Lock()

    @property
    def active(self) -> int:
        with self._lock:
            return self._count

    def enter(self) -> None:
        with self._lock:
            self._count += 1

    def exit(self) -> None:
        with self._lock:
            self._count = max(0, self._count - 1)


# ---------------------------------------------------------------------------
# app factory
# ---------------------------------------------------------------------------
def create_api(generator_cls, job_store, settings, outputs_volume) -> FastAPI:
    """Build the FastAPI application.

    ``generator_cls`` is the Modal class (or any object) exposing ``generate`` and
    ``generate_bytes``; it is instantiated lazily once per container.
    """

    instances: dict[str, Any] = {}
    streams = _StreamTracker()
    poll_failures: dict[str, int] = {}

    def generator():
        instance = instances.get("generator")
        if instance is None:
            instance = generator_cls()
            instances["generator"] = instance
        return instance

    api = FastAPI(
        title="Video + Audio Generation API",
        version="2.0.0",
        responses={400: {"model": ErrorBody, "description": "Invalid request"}},
    )

    # -- shared helpers ---------------------------------------------------
    async def _reload_outputs() -> None:
        if outputs_volume is None:
            return
        reload = getattr(outputs_volume, "reload", None)
        if not callable(reload):
            return
        if streams.active:
            # Reloading while a response is streaming from the mount can truncate it.
            logger.debug("skipping outputs reload while %d stream(s) are active", streams.active)
            return
        try:
            await _call_maybe_async(reload)
        except Exception:  # pragma: no cover - reload is best effort
            logger.warning("failed to reload the outputs volume", exc_info=True)

    async def _commit_outputs() -> None:
        if outputs_volume is None:
            return
        commit = getattr(outputs_volume, "commit", None)
        if not callable(commit):
            return
        await _call_maybe_async(commit)

    def _video_path(record: dict, job_id: str) -> Path:
        stored = record.get("video_path")
        if isinstance(stored, str) and stored.strip():
            return Path(stored)
        return Path(settings.OUTPUTS_DIR) / f"{job_id}.mp4"

    def _deadline(record: dict) -> datetime:
        created = _parse_timestamp(record.get("created_at"), "created_at")
        return created + timedelta(
            seconds=float(settings.FUNCTION_TIMEOUT) + _POLL_GRACE_SECONDS
        )

    def _external_url(request: Request, path: str) -> str:
        base = (getattr(settings, "PUBLIC_BASE_URL", "") or "").rstrip("/")
        root = (request.scope.get("root_path") or "").rstrip("/")
        if base:
            return f"{base}{root}{path}"
        return f"{root}{path}"

    def _to_job_status(record: dict | None, request: Request) -> JobStatus:
        if not isinstance(record, dict) or not record.get("id"):
            raise ApiError(404, "not_found", "job was not found")
        status = record.get("status")
        if status not in JOB_STATES:
            raise ApiError(500, "corrupt_record", "job record has an invalid state")
        try:
            created_at = _parse_timestamp(record.get("created_at"), "created_at")
            updated_at = _parse_timestamp(record.get("updated_at"), "updated_at")
        except ValueError as exc:
            raise ApiError(500, "corrupt_record", "job record has malformed timestamps") from exc

        raw_result = record.get("result")
        result = None
        if raw_result is not None:
            try:
                result = ResolvedParams.model_validate(raw_result)
            except ValidationError as exc:
                raise ApiError(500, "corrupt_record", "job record has an invalid result") from exc

        job_id = str(record["id"])
        video_url = (
            _external_url(request, f"/v1/generations/{job_id}/video")
            if status == "succeeded"
            else None
        )
        error = record.get("error")
        try:
            return JobStatus(
                id=job_id,
                status=status,
                created_at=created_at,
                updated_at=updated_at,
                result=result,
                video_url=video_url,
                error=str(error) if error else None,
            )
        except ValidationError as exc:
            raise ApiError(500, "corrupt_record", "job record is inconsistent") from exc

    async def _cancel_call(call_id: Any) -> None:
        if not call_id:
            return
        try:
            call = await _call_maybe_async(modal.FunctionCall.from_id, str(call_id))
            cancel = getattr(call, "cancel", None)
            if callable(cancel):
                await _call_maybe_async(cancel)
        except Exception:
            logger.warning("failed to cancel Modal call %s", call_id, exc_info=True)

    def _describe(exc: BaseException) -> str:
        name = type(exc).__name__
        text = str(exc).strip().splitlines()
        detail = text[0][:400] if text else ""
        return f"{name}: {detail}" if detail else name

    async def _sync_job(job_id: str) -> dict:
        """Refresh a non-terminal job from its Modal call, then return the record."""
        record = await job_store.get(job_id)
        if record is None:
            raise ApiError(404, "not_found", f"job {job_id} was not found")
        status = record.get("status")
        if status not in JOB_STATES:
            raise ApiError(500, "corrupt_record", f"job {job_id} has an invalid state")
        if status in TERMINAL_STATES:
            return record

        # Enforced even when no call id was recorded: a job that never reached the
        # worker must still become terminal instead of staying queued forever.
        if datetime.now(timezone.utc) > _deadline(record):
            await job_store.mark_failed(job_id, "the generation timed out")
            poll_failures.pop(job_id, None)
            return await job_store.get(job_id) or record

        if not record.get("call_id"):
            return record

        call_id = record["call_id"]
        try:
            call = await _call_maybe_async(modal.FunctionCall.from_id, str(call_id))
            outcome = await _call_maybe_async(
                call.get, timeout=float(settings.POLL_TIMEOUT_SECONDS)
            )
        except (asyncio.TimeoutError, TimeoutError):
            # Modal's ``get`` raises TimeoutError while the call is still running.  A
            # worker that genuinely raises TimeoutError is retried too, but the counter
            # guarantees the job reaches a terminal state instead of hanging forever.
            failures = poll_failures.get(job_id, 0) + 1
            poll_failures[job_id] = failures
            if failures >= int(settings.POLL_MAX_TRANSIENT_FAILURES):
                logger.warning("job %s lost its worker call after %d polls", job_id, failures)
                await job_store.mark_failed(job_id, "the generation was lost")
                poll_failures.pop(job_id, None)
                return await job_store.get(job_id) or record
            return record
        except Exception as exc:
            logger.exception("job %s failed in the worker", job_id)
            await job_store.mark_failed(job_id, f"generation failed ({_describe(exc)})")
            poll_failures.pop(job_id, None)
            return await job_store.get(job_id) or record

        poll_failures.pop(job_id, None)
        payload = outcome if isinstance(outcome, dict) else None
        if payload is None:
            logger.error("job %s returned an unexpected payload: %r", job_id, type(outcome))
            await job_store.mark_failed(job_id, "the worker returned no result")
            return await job_store.get(job_id) or record

        try:
            params = ResolvedParams.model_validate(payload.get("params"))
        except ValidationError:
            logger.exception("job %s returned invalid params", job_id)
            await job_store.mark_failed(job_id, "the worker returned an invalid result")
            return await job_store.get(job_id) or record

        video_path = Path(payload.get("video_path") or _video_path(record, job_id))
        # The worker commits to the volume after the call returns; give the mount one
        # chance to catch up before declaring the artefact missing.
        if not await asyncio.to_thread(video_path.is_file):
            await _reload_outputs()
            if not await asyncio.to_thread(video_path.is_file):
                logger.warning("job %s finished but %s is not visible", job_id, video_path)
                return record

        updated = await job_store.mark_succeeded(
            job_id, params, str(video_path), expect=("queued", "running")
        )
        return updated or await job_store.get(job_id) or record

    def _prepare_request(req: GenerationRequest) -> GenerationRequest:
        """Resolve the profile (and its mode constraints) before touching the GPU."""
        profile = req.profile or settings.DEFAULT_PROFILE
        if profile not in PROFILES:
            raise ApiError(400, "invalid_request", f"unknown profile {profile!r}")
        if profile == "turbo":
            if not settings.TURBO_TRANSFORMER_REPO:
                raise ApiError(
                    400, "invalid_request", "the turbo profile is not configured on this server"
                )
            if req.effective_mode() == "ref2v":
                raise ApiError(
                    400, "invalid_request", "the turbo profile does not support ref2v"
                )
        try:
            return req.model_copy(update={"profile": profile})
        except (ValidationError, ValueError) as exc:  # pragma: no cover - defensive
            raise ApiError(400, "invalid_request", str(exc)) from exc

    # -- exception handlers ----------------------------------------------
    @api.exception_handler(ApiError)
    async def _handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _error_response(exc.status_code, exc.code, exc.message, exc.headers)

    @api.exception_handler(RequestValidationError)
    async def _handle_request_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        details: list[str] = []
        for error in exc.errors():
            location = ".".join(
                str(part)
                for part in error.get("loc", ())
                if part not in ("body", "query", "path", "header")
            )
            message = error.get("msg", "invalid value")
            details.append(f"{location}: {message}" if location else str(message))
        return _error_response(
            400, "invalid_request", "; ".join(details) or "invalid request"
        )

    @api.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        detail = exc.detail
        code = getattr(exc, "code", None) or _code_for_status(exc.status_code)
        if exc.status_code >= 500:
            # Never echo internal details of a server side failure.
            detail = _code_for_status(exc.status_code).replace("_", " ")
        headers = getattr(exc, "headers", None)
        return _error_response(exc.status_code, code, detail, headers)

    @api.exception_handler(ValidationError)
    async def _handle_validation_error(
        request: Request, exc: ValidationError
    ) -> JSONResponse:
        logger.exception("unhandled validation error", exc_info=exc)
        return _error_response(500, "internal_error", "internal error")

    @api.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return _error_response(500, "internal_error", "internal error")

    # -- routes -----------------------------------------------------------
    @api.get("/health", tags=["meta"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/v1/models", response_model=ModelList, tags=["meta"])
    async def list_models() -> ModelList:
        profiles = [
            profile
            for profile in PROFILES
            if profile != "turbo" or bool(settings.TURBO_TRANSFORMER_REPO)
        ]
        default_profile = (
            settings.DEFAULT_PROFILE
            if settings.DEFAULT_PROFILE in profiles
            else profiles[0]
        )
        return ModelList(
            model_repo_id=settings.MODEL_REPO_ID,
            modes=list(MODES),
            profiles=profiles,
            default_profile=default_profile,
            resolutions={
                name: {
                    aspect: RESOLUTIONS[name][aspect]
                    for aspect in ASPECT_RATIOS
                    if aspect in RESOLUTIONS[name]
                }
                for name in RESOLUTION_NAMES
            },
            fps=int(settings.FPS),
            frame_grid={
                "formula": "17n+5",
                "min_n": NATIVE_MIN_N,
                "max_n": NATIVE_MAX_N,
            },
            native_segment_bounds={
                "min_frames": NATIVE_MIN_FRAMES,
                "max_frames": NATIVE_MAX_FRAMES,
            },
        )

    # NOTE: declared before ``/v1/generations/{id}`` on purpose (see module docstring).
    @api.post(
        "/v1/generations/sync",
        response_class=FileResponse,
        tags=["generations"],
        responses={
            200: {"content": {"video/mp4": {}}, "description": "Rendered video"},
            413: {"model": ErrorBody},
            504: {"model": ErrorBody},
        },
    )
    async def generate_sync(req: GenerationRequest):
        prepared = _prepare_request(req)
        try:
            result = await _call_maybe_async(
                generator().generate_bytes.remote,
                prepared.model_dump_json(),
                timeout=float(settings.SYNC_REQUEST_TIMEOUT),
            )
        except asyncio.TimeoutError as exc:
            raise ApiError(504, "upstream_timeout", "the generation timed out") from exc
        except (ValidationError, ValueError):
            raise
        except Exception as exc:
            logger.exception("synchronous generation failed")
            raise ApiError(
                502, "upstream_error", f"the generation failed ({_describe(exc)})"
            ) from exc

        payload = _split_sync_result(result)
        mp4_bytes, _params = payload
        if len(mp4_bytes) > int(settings.MAX_SYNC_BYTES):
            raise ApiError(
                413,
                "payload_too_large",
                "the rendered video exceeds MAX_SYNC_BYTES; use the asynchronous API",
            )
        handle, tmp_name = tempfile.mkstemp(prefix="va-sync-", suffix=".mp4")
        try:
            with os.fdopen(handle, "wb") as out:
                await asyncio.to_thread(out.write, mp4_bytes)
        except BaseException:
            _unlink_if_present(Path(tmp_name))
            raise
        return FileResponse(
            tmp_name,
            media_type="video/mp4",
            filename="output.mp4",
            background=BackgroundTask(_unlink_if_present, Path(tmp_name)),
        )

    @api.post(
        "/v1/generations",
        response_model=JobStatus,
        status_code=202,
        tags=["generations"],
        responses={500: {"model": ErrorBody}},
    )
    async def create_generation(
        req: GenerationRequest, request: Request, response: Response
    ):
        prepared = _prepare_request(req)
        request_json = prepared.model_dump_json()
        job_id = str(uuid.uuid4())
        await job_store.create(job_id, request_json)

        job_header = {"X-Job-Id": job_id}
        # Echoed on both success and failure so a client can recover from a truncated
        # response without having to guess whether a job was created.
        response.headers["X-Job-Id"] = job_id
        try:
            call = await _call_maybe_async(
                generator().generate.spawn, request_json, job_id
            )
        except Exception as exc:
            logger.exception("failed to submit job %s", job_id)
            await job_store.mark_failed(job_id, "the job could not be submitted")
            raise ApiError(
                500,
                "submission_failed",
                f"the job could not be submitted ({_describe(exc)})",
                headers=job_header,
            ) from exc

        call_id = getattr(call, "object_id", None) or getattr(call, "call_id", None)
        if not call_id:
            await _cancel_call(getattr(call, "object_id", None))
            await job_store.mark_failed(job_id, "the job could not be submitted")
            raise ApiError(
                500, "submission_failed", "the job could not be submitted", headers=job_header
            )

        try:
            record = await job_store.set_call(job_id, str(call_id))
        except Exception as exc:
            logger.exception("failed to record the call id for job %s", job_id)
            await _cancel_call(call_id)
            await job_store.mark_failed(job_id, "the job could not be submitted")
            raise ApiError(
                500,
                "submission_failed",
                "the job could not be recorded",
                headers=job_header,
            ) from exc

        return _to_job_status(record or await job_store.get(job_id), request)

    @api.get(
        "/v1/generations/{id}",
        response_model=JobStatus,
        tags=["generations"],
        responses={404: {"model": ErrorBody}},
    )
    async def get_generation(
        request: Request, job_id: str = PathParameter(alias="id")
    ):
        record = await _sync_job(job_id)
        return _to_job_status(record, request)

    @api.get(
        "/v1/generations/{id}/video",
        tags=["generations"],
        response_class=StreamingResponse,
        responses={404: {"model": ErrorBody}, 409: {"model": ErrorBody}, 410: {"model": ErrorBody}},
    )
    async def get_generation_video(
        request: Request, job_id: str = PathParameter(alias="id")
    ):
        job_id = str(job_id)
        record = await _sync_job(job_id)
        status = record.get("status")
        if status == "failed":
            raise ApiError(410, "gone", record.get("error") or "the generation failed")
        if status != "succeeded":
            raise ApiError(409, "conflict", "the job has not finished yet")

        video_path = _video_path(record, job_id)
        try:
            handle, size = await asyncio.to_thread(_open_for_read, video_path)
        except FileNotFoundError:
            # The worker may have committed after our mount was last refreshed.
            await _reload_outputs()
            try:
                handle, size = await asyncio.to_thread(_open_for_read, video_path)
            except (FileNotFoundError, IsADirectoryError):
                raise ApiError(404, "not_found", "the video file is missing") from None

        streams.enter()

        async def _body() -> AsyncIterator[bytes]:
            try:
                async for chunk in _iter_file(handle):
                    yield chunk
            finally:
                try:
                    await asyncio.to_thread(_close_quietly, handle)
                finally:
                    streams.exit()

        headers = {
            "Content-Length": str(size),
            "Content-Disposition": f'attachment; filename="{job_id}.mp4"',
            "Cache-Control": "private, max-age=3600",
        }
        return StreamingResponse(_body(), media_type="video/mp4", headers=headers)

    @api.delete(
        "/v1/generations/{id}",
        status_code=204,
        tags=["generations"],
        response_class=Response,
        responses={404: {"model": ErrorBody}},
    )
    async def delete_generation(job_id: str = PathParameter(alias="id")):
        job_id = str(job_id)
        record = await job_store.get(job_id)
        if record is None:
            raise ApiError(404, "not_found", f"job {job_id} was not found")

        if record.get("status") not in TERMINAL_STATES:
            await _cancel_call(record.get("call_id"))

        video_path = _video_path(record, job_id)
        removed = await asyncio.to_thread(_unlink_if_present, video_path)
        if removed:
            try:
                await _commit_outputs()
            except Exception as exc:
                logger.exception("failed to commit the outputs volume after a delete")
                await job_store.mark_failed(job_id, "the video file could not be deleted")
                raise ApiError(
                    502, "commit_failed", "the video file could not be released"
                ) from exc
        await job_store.delete(job_id)
        poll_failures.pop(job_id, None)
        return Response(status_code=204)

    _install_openapi_rewrite(api)
    return api


def _open_for_read(path: Path) -> tuple[Any, int]:
    """Open a file and report its size from the same descriptor (no TOCTOU race)."""
    handle = path.open("rb")
    try:
        size = os.fstat(handle.fileno()).st_size
    except OSError:
        handle.close()
        raise
    return handle, size


def _split_sync_result(result: Any) -> tuple[bytes, dict[str, Any]]:
    """Unpack ``(mp4_bytes, params)`` from a worker reply, rejecting anything else."""
    mp4: Any = None
    params: Any = None
    if isinstance(result, (tuple, list)) and len(result) == 2:
        mp4, params = result
    elif isinstance(result, dict):
        mp4 = result.get("video") or result.get("mp4") or result.get("content")
        params = result.get("params") or result.get("result") or {}
    if not isinstance(mp4, (bytes, bytearray, memoryview)):
        raise ApiError(500, "internal_error", "the worker returned an invalid payload")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        try:
            params = params.model_dump()
        except AttributeError:
            raise ApiError(
                500, "internal_error", "the worker returned an invalid payload"
            ) from None
    return bytes(mp4), params


def _install_openapi_rewrite(api: FastAPI) -> None:
    """Drop FastAPI's ``422`` responses so the schema matches the 400 we actually send."""
    original = api.openapi

    def openapi() -> dict[str, Any]:
        if api.openapi_schema is not None:
            return api.openapi_schema
        schema = original()
        for operations in schema.get("paths", {}).values():
            if not isinstance(operations, dict):
                continue
            for operation in operations.values():
                if not isinstance(operation, dict):
                    continue
                responses = operation.get("responses")
                if isinstance(responses, dict):
                    responses.pop("422", None)
        components = schema.get("components", {}).get("schemas", {})
        if isinstance(components, dict):
            components.pop("HTTPValidationError", None)
            components.pop("ValidationError", None)
        api.openapi_schema = schema
        return schema

    api.openapi = openapi


__all__ = ["ApiError", "create_api"]
