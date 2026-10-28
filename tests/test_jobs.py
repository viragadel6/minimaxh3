import asyncio

import pytest

from config import Settings
from runtime.jobs import JobStore, TERMINAL_STATES

PARAMS = {
    "width": 1344,
    "height": 768,
    "num_frames": 124,
    "fps": 24,
    "duration_seconds": 124 / 24,
    "seed": 1,
    "profile": "fast",
    "steps": 20,
    "mode": "t2v",
    "segments": [124],
}


@pytest.fixture
def store():
    return JobStore(Settings())


def test_create_writes_a_complete_record(store):
    record = asyncio.run(store.create("job-1", "{}"))
    assert record["id"] == "job-1"
    assert record["status"] == "queued"
    assert record["created_at"] == record["updated_at"]
    for key in ("call_id", "result", "video_path", "error"):
        assert record[key] is None


def test_get_missing_returns_none(store):
    assert asyncio.run(store.get("nope")) is None


def test_get_fills_in_fields_from_legacy_records(store):
    store.store.put("legacy", {"id": "legacy", "status": "queued"})
    record = asyncio.run(store.get("legacy"))
    assert record["request_json"] == ""
    assert record["result"] is None


def test_transitions(store):
    asyncio.run(store.create("job-1", "{}"))
    record = asyncio.run(store.set_call("job-1", "call-1"))
    assert record["call_id"] == "call-1"
    assert asyncio.run(store.mark_running("job-1"))["status"] == "running"
    record = asyncio.run(store.mark_succeeded("job-1", PARAMS, "/outputs/job-1.mp4"))
    assert record["status"] == "succeeded"
    assert record["result"] == PARAMS
    assert record["video_path"] == "/outputs/job-1.mp4"
    assert record["updated_at"] >= record["created_at"]


def test_terminal_states_are_final(store):
    asyncio.run(store.create("job-1", "{}"))
    asyncio.run(store.mark_succeeded("job-1", PARAMS, "/x.mp4"))
    assert asyncio.run(store.mark_failed("job-1", "late"))["status"] == "succeeded"
    assert asyncio.run(store.mark_running("job-1"))["status"] == "succeeded"
    assert asyncio.run(
        store.mark_succeeded("job-1", PARAMS, "/y.mp4", expect=("queued", "running"))
    )["video_path"] == "/x.mp4"


def test_failed_job_can_be_retried_as_running(store):
    asyncio.run(store.create("job-1", "{}"))
    asyncio.run(store.mark_failed("job-1", "boom"))
    assert asyncio.run(store.mark_running("job-1"))["status"] == "failed"


def test_updated_at_changes_on_every_write(store):
    asyncio.run(store.create("job-1", "{}"))
    first = asyncio.run(store.mark_running("job-1"))["updated_at"]
    second = asyncio.run(store.mark_failed("job-1", "boom"))["updated_at"]
    assert second >= first


def test_expect_guard_skips_the_mutation(store):
    asyncio.run(store.create("job-1", "{}"))
    record = asyncio.run(
        store.update("job-1", lambda r: {**r, "status": "failed"}, expect=("succeeded",))
    )
    assert record["status"] == "queued"


def test_mutator_returning_none_keeps_the_record(store):
    asyncio.run(store.create("job-1", "{}"))
    record = asyncio.run(store.update("job-1", lambda r: None))
    assert record["status"] == "queued"


def test_update_of_missing_job_returns_none(store):
    assert asyncio.run(store.update("nope", lambda r: r)) is None


def test_delete_removes_the_record_and_lock(store):
    asyncio.run(store.create("job-1", "{}"))

    async def _delete():
        await store.delete("job-1")
        assert "job-1" not in store._locks

    asyncio.run(_delete())
    assert asyncio.run(store.get("job-1")) is None


def test_delete_is_idempotent(store):
    asyncio.run(store.delete("never-existed"))


def test_concurrent_updates_do_not_lose_writes(store):
    asyncio.run(store.create("job-1", "{}"))

    async def _bump():
        for _ in range(20):
            await store.update(
                "job-1", lambda r: {**r, "video_path": (r["video_path"] or "") + "x"}
            )

    async def _run():
        await asyncio.gather(_bump(), _bump())
        return await store.get("job-1")

    record = asyncio.run(_run())
    assert len(record["video_path"]) == 40


def test_mark_deleted_clears_artefacts(store):
    asyncio.run(store.create("job-1", "{}"))
    asyncio.run(store.mark_succeeded("job-1", PARAMS, "/x.mp4"))
    record = asyncio.run(store.mark_deleted("job-1", "removed by the user"))
    assert record["status"] == "failed"
    assert record["video_path"] is None
    assert record["result"] is None


def test_terminal_states_constant():
    assert set(TERMINAL_STATES) == {"succeeded", "failed"}
