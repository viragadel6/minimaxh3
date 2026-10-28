"""Job persistence on a Modal ``Dict``.

All read-modify-write cycles go through :meth:`JobStore.update`, which serialises
mutations for one job id with an asyncio lock and verifies the expected status before
writing.  That prevents a slow poll from clobbering a terminal state (and prevents
duplicate success updates), and it collapses a poll into a single round trip.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from datetime import datetime, timezone
from typing import Callable, Iterable

import modal

TERMINAL_STATES = ("succeeded", "failed")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _dict_op(fn, *args, **kwargs):
    aio = getattr(fn, "aio", None)
    if callable(aio):
        result = aio(*args, **kwargs)
        if not inspect.isawaitable(result):
            return result
        return await result
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    result = await asyncio.to_thread(fn, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _new_record(job_id: str, request_json: str) -> dict:
    now = _utc_iso()
    return {
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


class JobStore:
    def __init__(self, settings) -> None:
        self.store = modal.Dict.from_name(settings.JOBS_DICT, create_if_missing=True)
        self._locks: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Lock]] = {}
        self._guard = threading.Lock()

    # -- locking ---------------------------------------------------------
    def _lock_for(self, job_id: str) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        with self._guard:
            entry = self._locks.get(job_id)
            if entry is None or entry[0] is not loop:
                entry = (loop, asyncio.Lock())
                self._locks[job_id] = entry
            return entry[1]

    def _drop_lock(self, job_id: str) -> None:
        with self._guard:
            self._locks.pop(job_id, None)

    # -- primitives ------------------------------------------------------
    async def create(self, job_id: str, request_json: str) -> dict:
        record = _new_record(job_id, request_json)
        await _dict_op(self.store.put, job_id, record)
        return record

    async def get(self, job_id: str) -> dict | None:
        record = await _dict_op(self.store.get, job_id)
        if record is None or not isinstance(record, dict):
            return None
        merged = _new_record(record.get("id", job_id), record.get("request_json") or "")
        merged.update(record)  # tolerate records written by older versions
        return merged

    async def update(
        self,
        job_id: str,
        mutator: Callable[[dict], dict | None],
        *,
        expect: Iterable[str] | None = None,
    ) -> dict | None:
        """Atomically mutate one record; returns the (possibly unchanged) record."""
        async with self._lock_for(job_id):
            record = await self.get(job_id)
            if record is None:
                return None
            if expect is not None and record.get("status") not in tuple(expect):
                return record
            updated = mutator(dict(record))
            if updated is None:
                return record
            updated["updated_at"] = _utc_iso()
            await _dict_op(self.store.put, job_id, updated)
            return updated

    # -- transitions -----------------------------------------------------
    async def set_call(self, job_id: str, call_id: str) -> dict | None:
        def mutate(record: dict) -> dict:
            record["call_id"] = str(call_id)
            return record

        return await self.update(job_id, mutate)

    async def mark_running(self, job_id: str) -> dict | None:
        def mutate(record: dict) -> dict | None:
            if record["status"] in TERMINAL_STATES:
                return None
            record["status"] = "running"
            record["error"] = None
            return record

        return await self.update(job_id, mutate, expect=("queued", "running"))

    async def mark_succeeded(
        self,
        job_id: str,
        params,
        video_path: str,
        *,
        expect: Iterable[str] = ("queued", "running"),
    ) -> dict | None:
        if hasattr(params, "model_dump"):
            params = params.model_dump()

        def mutate(record: dict) -> dict:
            record["status"] = "succeeded"
            record["result"] = params
            record["video_path"] = str(video_path)
            record["error"] = None
            return record

        return await self.update(job_id, mutate, expect=expect)

    async def mark_failed(self, job_id: str, message: str) -> dict | None:
        def mutate(record: dict) -> dict | None:
            if record["status"] == "succeeded":
                # A late failure must never overwrite a committed success.
                return None
            record["status"] = "failed"
            record["error"] = str(message) or "generation failed"
            return record

        return await self.update(job_id, mutate)

    async def mark_deleted(self, job_id: str, reason: str) -> dict | None:
        """Soft-delete bookkeeping for records whose artefact is being removed."""

        def mutate(record: dict) -> dict:
            record["status"] = "failed"
            record["error"] = str(reason)
            record["result"] = None
            record["video_path"] = None
            return record

        return await self.update(job_id, mutate)

    async def delete(self, job_id: str) -> None:
        async with self._lock_for(job_id):
            try:
                await _dict_op(self.store.pop, job_id)
            except KeyError:
                pass
        self._drop_lock(job_id)


__all__ = ["JobStore", "TERMINAL_STATES"]
