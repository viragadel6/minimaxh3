"""End-to-end contract test: API <-> job store <-> Modal worker <-> generator.

The unit tests pin each layer separately; this one wires them together the way
``app.py`` does and drives a full submit -> poll -> download -> delete cycle. It is
the test that catches drift in the worker's reply shape (``{params, video_path}``),
in the job record lifecycle, and in where the rendered file actually lands.
"""

import asyncio
import dataclasses
import threading
import time

import pytest

import stubs

stubs.install()

import numpy as np  # noqa: E402

import app as app_module  # noqa: E402
from runtime import generator as generator_module  # noqa: E402
from runtime.generator import Generator  # noqa: E402
from runtime.jobs import JobStore  # noqa: E402
from test_generator import _FakeManager  # noqa: E402

from api import create_api  # noqa: E402
from config import Settings  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"payload" * 64


# ---------------------------------------------------------------------------
# a local stand-in for Modal's remote call machinery
# ---------------------------------------------------------------------------
class _LocalCall:
    """Runs a coroutine in a background thread, like ``method.spawn`` + ``get``."""

    def __init__(self, coro_factory):
        self._done = threading.Event()
        self._payload = None
        self._error = None
        self.cancelled = False
        self.object_id = f"call-{id(self)}"
        self._thread = threading.Thread(target=self._run, args=(coro_factory,), daemon=True)
        self._thread.start()

    def _run(self, coro_factory):
        try:
            self._payload = asyncio.run(coro_factory())
        except BaseException as exc:  # noqa: BLE001 - re-raised through ``get``
            self._error = exc
        finally:
            self._done.set()

    def get(self, timeout=None):
        if not self._done.wait(timeout):
            raise TimeoutError("The function call has not finished")
        if self._error is not None:
            raise self._error
        return self._payload

    def cancel(self):
        self.cancelled = True


class _LocalMethod:
    def __init__(self, worker_getter, name):
        self._worker_getter = worker_getter
        self._name = name

    def spawn(self, *args, **kwargs):
        method = getattr(self._worker_getter(), self._name)
        return _LocalCall(lambda: method(*args, **kwargs))


class _LocalClass:
    """``generator_cls`` stand-in exposing ``.generate`` / ``.generate_bytes``.

    ``create_api`` accepts a class (Modal's ``@app.cls``); calling it must return an
    object with the two remote method handles.
    """

    generate = None
    generate_bytes = None

    def __init__(self, worker_getter):
        self.generate = _LocalMethod(worker_getter, "generate")
        self.generate_bytes = _LocalMethod(worker_getter, "generate_bytes")


class _Volume:
    def __init__(self):
        self.commits = 0
        self.reloads = 0

    def commit(self):
        self.commits += 1

    def reload(self):
        self.reloads += 1


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def stack(tmp_path, monkeypatch):
    settings = dataclasses.replace(
        Settings(),
        MODELS_DIR=str(tmp_path / "models"),
        OUTPUTS_DIR=str(tmp_path / "outputs"),
        OUTPUT_STAGING_DIR=str(tmp_path / "staging"),
        SYNC_REQUEST_TIMEOUT=30.0,
        POLL_TIMEOUT_SECONDS=5.0,
    )
    (tmp_path / "outputs").mkdir()
    monkeypatch.setattr(app_module, "settings", settings)

    manager = _FakeManager()
    generator = Generator(manager, settings)

    def _segment(self, adapter, transformer, req, params, length, index, first, last, refs, seed):
        return (
            np.zeros((length, 6, 8, 3), dtype=np.uint8),
            np.zeros((length * 10, 2), dtype=np.float32),
        )

    def _encode_mp4(frames, audio, sample_rate, fps, out_path, *args, **kwargs):
        from pathlib import Path

        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(FAKE_MP4)
        return path

    monkeypatch.setattr(Generator, "_run_one_segment", _segment)
    monkeypatch.setattr(generator_module, "encode_mp4", _encode_mp4)

    worker = app_module.VideoAudioGenerator()
    worker.generator = generator
    worker.job_store = JobStore(settings)
    worker._publish_lock = asyncio.Lock()
    worker._run_lock = asyncio.Lock()

    # In production the worker and the API share one Modal volume handle; mirror that
    # by pointing the module global at the same object the API receives.
    volume = _Volume()
    monkeypatch.setattr(app_module, "outputs_volume", volume)
    # ``create_api`` instantiates its argument once per container, exactly like
    # ``create_api(VideoAudioGenerator, ...)`` does with the Modal class.
    api = create_api(
        lambda: _LocalClass(lambda: worker), worker.job_store, settings, volume
    )
    client = TestClient(api, raise_server_exceptions=True)

    try:
        yield {
            "client": client,
            "settings": settings,
            "store": worker.job_store,
            "volume": volume,
            "worker": worker,
            "generator": generator,
            "tmp": tmp_path,
        }
    finally:
        generator.shutdown_nowait()


def _wait_for_terminal(client, job_id, deadline=30.0):
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        response = client.get(f"/v1/generations/{job_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish in time")


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_full_job_lifecycle(stack):
    client = stack["client"]
    created = client.post("/v1/generations", json={"prompt": "a cat", "num_frames": 124})
    assert created.status_code == 202, created.text
    job_id = created.json()["id"]
    assert created.headers["X-Job-Id"] == job_id

    body = _wait_for_terminal(client, job_id)
    assert body["status"] == "succeeded", body
    assert body["result"]["num_frames"] == 124
    assert body["video_url"].endswith(f"/v1/generations/{job_id}/video")

    # The worker published the real file under the outputs directory.
    published = stack["tmp"] / "outputs" / f"{job_id}.mp4"
    assert published.is_file()
    assert published.read_bytes() == FAKE_MP4

    # No staging leftovers: the directory is removed after a successful publish.
    staging_root = stack["tmp"] / "staging" / "va-staging"
    assert not staging_root.exists() or not any(staging_root.iterdir())

    download = client.get(f"/v1/generations/{job_id}/video")
    assert download.status_code == 200
    assert download.content == FAKE_MP4
    assert download.headers["content-length"] == str(len(FAKE_MP4))
    # The stream lease must be balanced after the response completes.
    assert stack["volume"].commits >= 1

    deleted = client.delete(f"/v1/generations/{job_id}")
    assert deleted.status_code == 204
    assert not published.exists()
    assert client.get(f"/v1/generations/{job_id}").status_code == 404


def test_sync_endpoint_returns_the_rendered_bytes(stack):
    response = stack["client"].post(
        "/v1/generations/sync", json={"prompt": "a dog", "num_frames": 124}
    )
    assert response.status_code == 200, response.text
    assert response.content == FAKE_MP4
    assert response.headers["content-type"] == "video/mp4"


def test_worker_failure_surfaces_as_a_failed_job(stack, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated CUDA OOM")

    monkeypatch.setattr(Generator, "_run_one_segment", _boom)
    client = stack["client"]
    job_id = client.post(
        "/v1/generations", json={"prompt": "x", "num_frames": 124}
    ).json()["id"]

    body = _wait_for_terminal(client, job_id)
    assert body["status"] == "failed"
    assert "CUDA OOM" in body["error"]
    assert body["video_url"] is None
    # The failed render must leave nothing behind on the volume.
    outputs = stack["tmp"] / "outputs"
    assert not list(outputs.glob("*.mp4"))
    assert client.get(f"/v1/generations/{job_id}/video").status_code == 410


def test_deleting_a_running_job_removes_it_completely(stack):
    client = stack["client"]
    created = client.post("/v1/generations", json={"prompt": "slow", "num_frames": 124})
    job_id = created.json()["id"]

    # Delete while the render is still in flight.
    assert client.delete(f"/v1/generations/{job_id}").status_code == 204
    assert asyncio.run(stack["store"].get(job_id)) is None
    assert client.get(f"/v1/generations/{job_id}").status_code == 404
    assert client.get(f"/v1/generations/{job_id}/video").status_code == 404
    # NOTE: stopping the in-flight render is Modal's cancellation, not ours; the local
    # stand-in cannot interrupt a running thread, so only the record lifecycle is
    # asserted here.
