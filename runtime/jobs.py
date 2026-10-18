import asyncio
import inspect
from datetime import datetime, timezone

import modal


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _dict_op(fn, *args, **kwargs):
    aio = getattr(fn, "aio", None)
    if callable(aio):
        return await aio(*args, **kwargs)
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    res = await asyncio.to_thread(fn, *args, **kwargs)
    if inspect.isawaitable(res):
        return await res
    return res


class JobStore:
    def __init__(self, settings) -> None:
        self.store = modal.Dict.from_name(
            settings.JOBS_DICT, create_if_missing=True
        )

    async def create(self, job_id: str, request_json: str) -> dict:
        now = _utc_iso()
        record = {
            "id": job_id,
            "request_json": request_json,
            "call_id": None,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "result": None,
            "video_path": None,
            "error": None,
        }
        await _dict_op(self.store.put, job_id, record)
        return record

    async def set_call(self, job_id: str, call_id: str) -> dict | None:
        record = await self.get(job_id)
        if record is None:
            return None
        record["call_id"] = call_id
        record["updated_at"] = _utc_iso()
        await _dict_op(self.store.put, job_id, record)
        return record

    async def mark_running(self, job_id: str) -> dict | None:
        record = await self.get(job_id)
        if record is None:
            return None
        record["status"] = "running"
        record["error"] = None
        record["updated_at"] = _utc_iso()
        await _dict_op(self.store.put, job_id, record)
        return record

    async def mark_succeeded(self, job_id: str, params, video_path: str) -> dict | None:
        record = await self.get(job_id)
        if record is None:
            return None
        if hasattr(params, "model_dump"):
            params = params.model_dump()
        record["status"] = "succeeded"
        record["result"] = params
        record["video_path"] = str(video_path)
        record["error"] = None
        record["updated_at"] = _utc_iso()
        await _dict_op(self.store.put, job_id, record)
        return record

    async def mark_failed(self, job_id: str, message: str) -> dict | None:
        record = await self.get(job_id)
        if record is None:
            return None
        record["status"] = "failed"
        record["error"] = str(message)
        record["updated_at"] = _utc_iso()
        await _dict_op(self.store.put, job_id, record)
        return record

    async def get(self, job_id: str) -> dict | None:
        return await _dict_op(self.store.get, job_id)

    async def delete(self, job_id: str) -> None:
        try:
            await _dict_op(self.store.pop, job_id)
        except KeyError:
            pass
