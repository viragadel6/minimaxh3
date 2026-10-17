import asyncio
from datetime import datetime, timezone

import modal


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class JobStore:
    def __init__(self, settings):
        self.dictionary = modal.Dict.from_name(
            settings.JOBS_DICT, create_if_missing=True
        )

    def _key(self, job_id, field):
        return f"{job_id}:{field}"

    async def create(self, job_id, request_json):
        now = utc_now()
        record = {
            "id": job_id,
            "request_json": request_json,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "result": None,
            "video_path": None,
            "error": None,
        }
        await self.dictionary.put.aio(self._key(job_id, "meta"), record)
        return await self.get(job_id)

    async def set_call(self, job_id, call_id):
        if await self.get(job_id) is None:
            return
        await self.dictionary.put.aio(
            self._key(job_id, "call"),
            {"call_id": call_id, "updated_at": utc_now()},
        )

    async def _state(self, job_id, status, **values):
        if await self.get(job_id) is None:
            return False
        await self.dictionary.put.aio(
            self._key(job_id, "state"),
            {"status": status, "updated_at": utc_now(), **values},
        )
        return True

    async def mark_running(self, job_id):
        return await self._state(job_id, "running", error=None)

    async def mark_succeeded(self, job_id, params, video_path):
        if hasattr(params, "model_dump"):
            params = params.model_dump()
        return await self._state(
            job_id,
            "succeeded",
            result=params,
            video_path=str(video_path),
            error=None,
        )

    async def mark_failed(self, job_id, message):
        return await self._state(
            job_id, "failed", error=str(message)[:8192], result=None
        )

    async def get(self, job_id):
        meta, state, call, deleted = await asyncio.gather(
            self.dictionary.get.aio(self._key(job_id, "meta")),
            self.dictionary.get.aio(self._key(job_id, "state")),
            self.dictionary.get.aio(self._key(job_id, "call")),
            self.dictionary.get.aio(self._key(job_id, "deleted")),
        )
        if meta is None or deleted:
            return None
        result = dict(meta)
        if state:
            result.update(state)
        if call:
            result["call_id"] = call["call_id"]
            result["updated_at"] = max(
                result["updated_at"], call["updated_at"]
            )
        return result

    async def delete(self, job_id):
        await self.dictionary.put.aio(
            self._key(job_id, "deleted"), {"deleted_at": utc_now()}
        )
        await asyncio.gather(
            self.dictionary.pop.aio(self._key(job_id, "meta"), None),
            self.dictionary.pop.aio(self._key(job_id, "state"), None),
            self.dictionary.pop.aio(self._key(job_id, "call"), None),
        )
