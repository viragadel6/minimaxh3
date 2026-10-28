import asyncio
import dataclasses
import json

import pytest

import api as api_module
import modal
from runtime.jobs import JobStore
from schemas import GenerationRequest
from api import ApiError, _call_maybe_async, _split_sync_result
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# doubles
# ---------------------------------------------------------------------------
class _FakeCall:
    """Stand-in for ``modal.FunctionCall`` (job polling and sync renders)."""

    def __init__(self, payload=None, error=None, pending=0):
        self.payload = payload
        self.error = error
        self.pending = pending
        self.cancelled = False

    def get(self, timeout=None):
        if self.pending > 0:
            self.pending -= 1
            raise TimeoutError("The function call has not finished")
        if self.error is not None:
            raise self.error
        return self.payload

    def cancel(self):
        self.cancelled = True


class _FakeHandle:
    def __init__(self, call_id):
        self.object_id = call_id


class _FakeRemote:
    """Stand-in for a Modal method handle: ``spawn`` returns a call object."""

    def __init__(self, calls, mode, control=None):
        self._calls = calls
        self._mode = mode
        self._control = control if control is not None else {}

    def spawn(self, request_json, job_id=None):
        if self._mode == "job":
            return _FakeHandle("call-1")
        # A synchronous render is spawned too, so a timeout can cancel it.
        if self._control.get("sync_error") is not None:
            call = _FakeCall(error=self._control["sync_error"])
        else:
            payload = self._calls.pop(0) if self._calls else None
            call = _FakeCall(
                payload=payload, pending=int(self._control.get("sync_pending", 0))
            )
        self._control["last_sync_call"] = call
        return call


class _FakeGeneratorHandle:
    def __init__(self, sync_results, control):
        self.generate = _FakeRemote([], "job", control)
        self.generate_bytes = _FakeRemote(sync_results, "bytes", control)


class _FakeVolume:
    def __init__(self):
        self.commits = 0
        self.reloads = 0

    def commit(self):
        self.commits += 1

    def reload(self):
        self.reloads += 1


def _settings(tmp_path, **overrides):
    from config import Settings

    base = Settings(OUTPUTS_DIR=str(tmp_path), OUTPUT_STAGING_DIR=str(tmp_path / "staging"))
    return dataclasses.replace(base, **overrides)


def _params_dict():
    return {
        "width": 1344,
        "height": 768,
        "num_frames": 124,
        "fps": 24,
        "duration_seconds": 124 / 24,
        "seed": 7,
        "profile": "fast",
        "steps": 20,
        "mode": "t2v",
        "segments": [124],
    }


def _build_env(root, monkeypatch, **overrides):
    settings = _settings(root, **overrides)
    store = JobStore(settings)
    volume = _FakeVolume()

    class Facade:
        def __init__(self):
            self.control = {}
            self.call = _FakeCall(payload={"params": _params_dict(), "video_path": None})
            self.sync_results = []

        def __call__(self):
            return _FakeGeneratorHandle(self.sync_results, self.control)

    facade = Facade()

    class FakeFunctionCall:
        @classmethod
        def from_id(cls, call_id):
            return facade.call

    monkeypatch.setattr(modal, "FunctionCall", FakeFunctionCall, raising=False)
    app = api_module.create_api(facade, store, settings, volume)
    return {
        "settings": settings,
        "store": store,
        "volume": volume,
        "facade": facade,
        "client": TestClient(app, raise_server_exceptions=False),
        "app": app,
        "root": root,
    }


@pytest.fixture
def env_factory(tmp_path, monkeypatch):
    counter = {"n": 0}

    def _make(**overrides):
        counter["n"] += 1
        root = tmp_path / f"env{counter['n']}"
        root.mkdir(parents=True, exist_ok=True)
        return _build_env(root, monkeypatch, **overrides)

    return _make


@pytest.fixture
def api_env(env_factory):
    return env_factory()


# ---------------------------------------------------------------------------
# metadata endpoints
# ---------------------------------------------------------------------------
def test_health_and_models(api_env):
    client = api_env["client"]
    assert client.get("/health").json() == {"status": "ok"}

    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["frame_grid"]["formula"] == "17n+5"
    assert body["default_profile"] in body["profiles"]
    assert "turbo" not in body["profiles"]
    assert set(body["resolutions"]) == {"480p", "768p"}
    assert all(len(v) >= 5 for v in body["resolutions"].values())


def test_openapi_documents_400_and_not_422(api_env):
    schema = api_env["app"].openapi()
    for operations in schema["paths"].values():
        for operation in operations.values():
            if not isinstance(operation, dict):
                continue
            responses = operation.get("responses") or {}
            assert "422" not in responses
    components = schema["components"]["schemas"]
    assert "ErrorBody" in components
    assert "HTTPValidationError" not in components
    assert "ErrorDetail" in components


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def test_invalid_request_is_a_400_error_body(api_env):
    client = api_env["client"]
    response = client.post("/v1/generations", json={"prompt": ""})
    assert response.status_code == 400
    body = response.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == "invalid_request"
    assert body["error"]["message"]


def test_unknown_field_is_rejected(api_env):
    response = api_env["client"].post(
        "/v1/generations", json={"prompt": "hi", "num_frame": 124}
    )
    assert response.status_code == 400
    assert "num_frame" in response.text


def test_unknown_route_uses_error_body(api_env):
    response = api_env["client"].get("/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_turbo_rejected_when_not_configured(api_env):
    response = api_env["client"].post(
        "/v1/generations", json={"prompt": "hi", "profile": "turbo"}
    )
    assert response.status_code == 400
    assert "turbo" in response.json()["error"]["message"]


# ---------------------------------------------------------------------------
# async job lifecycle
# ---------------------------------------------------------------------------
def _create_job(client, **payload):
    payload = {"prompt": "hi", **payload}
    response = client.post("/v1/generations", json=payload)
    assert response.status_code == 202, response.text
    return response


def test_create_returns_queued_job_and_header(api_env):
    response = _create_job(api_env["client"])
    body = response.json()
    assert body["status"] == "queued"
    assert body["result"] is None
    assert body["video_url"] is None
    assert response.headers["X-Job-Id"] == body["id"]
    stored = asyncio.run(api_env["store"].get(body["id"]))
    assert json.loads(stored["request_json"])["profile"] == api_env["settings"].DEFAULT_PROFILE


def test_pending_job_stays_queued(api_env):
    job = _create_job(api_env["client"]).json()
    api_env["facade"].call.pending = 1
    response = api_env["client"].get(f"/v1/generations/{job['id']}")
    assert response.status_code == 200
    assert response.json()["status"] == "queued"


def test_pending_job_fails_after_repeated_polls(api_env):
    job = _create_job(api_env["client"]).json()
    api_env["facade"].call.pending = 10
    for _ in range(api_env["settings"].POLL_MAX_TRANSIENT_FAILURES):
        response = api_env["client"].get(f"/v1/generations/{job['id']}")
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["error"]


def test_expired_job_without_call_id_is_failed(api_env):
    client = api_env["client"]
    job = _create_job(client).json()
    stale = asyncio.run(api_env["store"].get(job["id"]))
    stale["call_id"] = None
    stale["created_at"] = "2020-01-01T00:00:00+00:00"
    asyncio.run(api_env["store"].update(job["id"], lambda _record: stale))
    response = client.get(f"/v1/generations/{job['id']}")
    assert response.json()["status"] == "failed"
    assert "timed out" in response.json()["error"]


def test_control_plane_error_is_transient_and_does_not_fail_the_job(api_env, monkeypatch):
    client = api_env["client"]
    job = _create_job(client).json()

    class BrokenLookup:
        @classmethod
        def from_id(cls, call_id):
            raise ConnectionError("modal control plane unavailable")

    monkeypatch.setattr(modal, "FunctionCall", BrokenLookup, raising=False)
    response = client.get(f"/v1/generations/{job['id']}")
    body = response.json()
    # A lookup failure says nothing about the render: the job stays non-terminal.
    assert body["status"] in ("queued", "running")
    assert body["error"] is None

    # ... but the retry budget still bounds it, so it cannot hang forever.
    for _ in range(api_env["settings"].POLL_MAX_TRANSIENT_FAILURES):
        response = client.get(f"/v1/generations/{job['id']}")
    assert response.json()["status"] == "failed"
    assert "lost" in response.json()["error"]


def test_worker_exception_fails_the_job(api_env):
    job = _create_job(api_env["client"]).json()
    api_env["facade"].call.error = RuntimeError("boom\nsecret traceback")
    response = api_env["client"].get(f"/v1/generations/{job['id']}")
    body = response.json()
    assert body["status"] == "failed"
    assert "RuntimeError" in body["error"]
    assert "secret traceback" not in body["error"]


def test_success_then_video_download(api_env):
    client = api_env["client"]
    job = _create_job(client).json()
    video = api_env["root"] / f"{job['id']}.mp4"
    video.write_bytes(b"0123456789" * 1000)
    api_env["facade"].call.payload = {
        "params": _params_dict(),
        "video_path": str(video),
    }

    response = client.get(f"/v1/generations/{job['id']}")
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["result"]["num_frames"] == 124
    assert body["video_url"].endswith(f"/v1/generations/{job['id']}/video")

    download = client.get(f"/v1/generations/{job['id']}/video")
    assert download.status_code == 200
    assert download.content == video.read_bytes()
    assert download.headers["content-type"] == "video/mp4"
    assert download.headers["content-length"] == str(video.stat().st_size)


def test_video_download_rejects_unfinished_job(api_env):
    job = _create_job(api_env["client"]).json()
    response = api_env["client"].get(f"/v1/generations/{job['id']}/video")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


def test_video_download_410_for_failed_job(api_env):
    job = _create_job(api_env["client"]).json()
    api_env["facade"].call.error = ValueError("nope")
    api_env["client"].get(f"/v1/generations/{job['id']}")
    response = api_env["client"].get(f"/v1/generations/{job['id']}/video")
    assert response.status_code == 410


def test_success_with_missing_file_is_not_reported(api_env):
    client = api_env["client"]
    job = _create_job(client).json()
    api_env["facade"].call.payload = {
        "params": _params_dict(),
        "video_path": str(api_env["root"] / "gone.mp4"),
    }
    response = client.get(f"/v1/generations/{job['id']}")
    assert response.json()["status"] == "queued"
    assert api_env["volume"].reloads >= 1


def test_invalid_worker_params_marks_failure(api_env):
    client = api_env["client"]
    job = _create_job(client).json()
    api_env["facade"].call.payload = {"params": {"nope": 1}, "video_path": "/tmp/x.mp4"}
    response = client.get(f"/v1/generations/{job['id']}")
    assert response.json()["status"] == "failed"


def test_delete_removes_file_and_record(api_env):
    client = api_env["client"]
    job = _create_job(client).json()
    video = api_env["root"] / f"{job['id']}.mp4"
    video.write_bytes(b"x")
    api_env["facade"].call.payload = {"params": _params_dict(), "video_path": str(video)}
    client.get(f"/v1/generations/{job['id']}")

    response = client.delete(f"/v1/generations/{job['id']}")
    assert response.status_code == 204
    assert not video.exists()
    assert api_env["volume"].commits == 1
    assert client.get(f"/v1/generations/{job['id']}").status_code == 404


def test_delete_cancels_running_call(api_env):
    client = api_env["client"]
    job = _create_job(client).json()
    response = client.delete(f"/v1/generations/{job['id']}")
    assert response.status_code == 204
    assert api_env["facade"].call.cancelled is True


def test_delete_unknown_job_is_404(api_env):
    response = api_env["client"].delete("/v1/generations/does-not-exist")
    assert response.status_code == 404


def test_duplicate_success_cannot_overwrite(api_env):
    client = api_env["client"]
    job = _create_job(client).json()
    video = api_env["root"] / f"{job['id']}.mp4"
    video.write_bytes(b"x")
    api_env["facade"].call.payload = {"params": _params_dict(), "video_path": str(video)}
    assert client.get(f"/v1/generations/{job['id']}").json()["status"] == "succeeded"
    # A late failure must not clobber the success.
    api_env["facade"].call.error = RuntimeError("late")
    assert client.get(f"/v1/generations/{job['id']}").json()["status"] == "succeeded"


# ---------------------------------------------------------------------------
# synchronous endpoint
# ---------------------------------------------------------------------------
def test_sync_returns_video(api_env):
    client = api_env["client"]
    facade = api_env["facade"]
    facade.sync_results.append((b"mp4-bytes", _params_dict()))
    response = client.post("/v1/generations/sync", json={"prompt": "hi"})
    assert response.status_code == 200, response.text
    assert response.content == b"mp4-bytes"
    assert response.headers["content-type"] == "video/mp4"


def test_sync_timeout_cancels_the_remote_call(env_factory):
    env = env_factory(SYNC_REQUEST_TIMEOUT=0.01)
    env["facade"].control["sync_pending"] = 10**6  # never becomes ready
    response = env["client"].post("/v1/generations/sync", json={"prompt": "hi"})
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "upstream_timeout"
    # The abandoned render must not keep the GPU busy for a caller that gave up.
    assert env["facade"].control["last_sync_call"].cancelled is True


def test_sync_rejects_oversized_payload(env_factory):
    env = env_factory(MAX_SYNC_BYTES=5)
    env["facade"].sync_results.append((b"x" * 10, _params_dict()))
    response = env["client"].post("/v1/generations/sync", json={"prompt": "hi"})
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_sync_worker_error_is_a_502(env_factory):
    env = env_factory()
    env["facade"].control["sync_error"] = RuntimeError("cuda oom")
    response = env["client"].post("/v1/generations/sync", json={"prompt": "hi"})
    assert response.status_code == 502
    assert "cuda oom" in response.json()["error"]["message"]


def test_sync_invalid_worker_payload(api_env):
    client = api_env["client"]
    api_env["facade"].sync_results.append(("not-a-video", {}))
    response = client.post("/v1/generations/sync", json={"prompt": "hi"})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"


def test_sync_validation_is_400(api_env):
    response = api_env["client"].post("/v1/generations/sync", json={"prompt": ""})
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def test_split_sync_result_forms():
    assert _split_sync_result((b"a", {"x": 1})) == (b"a", {"x": 1})
    assert _split_sync_result({"video": b"b", "params": {"y": 2}}) == (b"b", {"y": 2})
    with pytest.raises(ApiError):
        _split_sync_result(("a", "b"))
    with pytest.raises(ApiError):
        _split_sync_result((b"a", 5))


def test_error_response_coerces_non_strings():
    from api import _error_response

    response = _error_response(400, 123, 456)
    body = json.loads(response.body)
    assert body["error"]["code"] == "123"
    assert body["error"]["message"] == "456"


def test_call_maybe_async_variants():
    assert asyncio.run(_call_maybe_async(lambda: 3)) == 3

    async def coro():
        return 4

    assert asyncio.run(_call_maybe_async(coro)) == 4

    class WithAio:
        def aio(self):
            async def inner():
                return 5

            return inner()

    assert asyncio.run(_call_maybe_async(WithAio().aio)) == 5

    class BadAio:
        def __call__(self):
            return 6

        def aio(self):
            return 6

    with pytest.raises(TypeError):
        asyncio.run(_call_maybe_async(BadAio()))


def test_generation_request_profile_writeback_is_serialised():
    req = GenerationRequest(prompt="hi")
    updated = req.model_copy(update={"profile": "fast"})
    assert json.loads(updated.model_dump_json())["profile"] == "fast"


# ---------------------------------------------------------------------------
# streaming lease
# ---------------------------------------------------------------------------
def test_stream_lease_releases_exactly_once():
    from api import _StreamLease, _StreamTracker

    class Handle:
        def __init__(self):
            self.closes = 0

        def close(self):
            self.closes += 1

    tracker = _StreamTracker()
    handle = Handle()
    lease = _StreamLease(tracker, handle)
    tracker.release = tracker.release  # sanity: bound method
    tracker._count = 1
    lease.release()
    lease.release()
    assert tracker.active == 0
    assert handle.closes == 1


def test_stream_tracker_never_goes_negative():
    from api import _StreamTracker

    tracker = _StreamTracker()
    tracker.release()
    tracker.release()
    assert tracker.active == 0


def test_error_message_flattens_structured_details():
    from api import _error_response

    structured = json.loads(_error_response(400, "invalid_request", {"detail": "bad seed"}).body)
    assert structured["error"]["message"] == "bad seed"

    nested = json.loads(_error_response(500, None, "").body)
    assert nested["error"]["message"] == "internal error"
