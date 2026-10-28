"""End-to-end orchestration tests for :class:`runtime.generator.Generator`.

The GPU worker is replaced with a stub that returns synthetic frames, so the parts
being tested are exactly the ones the audit flagged: segment planning, seed handling,
frame chaining between segments, audio plumbing and the encode step.
"""

import asyncio
import threading

import numpy as np
import pytest

import stubs

stubs.install()

import torch  # noqa: E402  (stub)

from config import Settings  # noqa: E402
from runtime import generator as generator_module  # noqa: E402
from runtime.generator import Generator  # noqa: E402
from schemas import GenerationRequest  # noqa: E402
from shapes import plan_segments  # noqa: E402


class _FakeAdapter:
    def __init__(self):
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        frames = np.zeros((kwargs["num_frames"], 6, 8, 3), dtype=np.uint8)
        audio = np.zeros((kwargs["num_frames"] * 100, 2), dtype=np.float32)
        return frames, audio

    def sample_rate(self):
        return 44100


class _FakeCache:
    def __init__(self):
        self.begins = []
        self.ends = 0

    def begin(self, **kwargs):
        self.begins.append(kwargs)

    def end(self):
        self.ends += 1


class _FakeLoraManager:
    def __init__(self):
        self.prepared = []
        self.activated = []
        self.deactivated = 0
        self.threads = []

    def prepare(self, transformer, specs):
        # LoRA preparation can move weights to the GPU, so it must happen on the
        # single GPU worker thread.
        self.threads.append(threading.current_thread().name)
        self.prepared.append(list(specs))
        return []

    def activate(self, transformer, prepared):
        self.threads.append(threading.current_thread().name)
        self.activated.append(prepared)

    def deactivate(self, transformer):
        self.threads.append(threading.current_thread().name)
        self.deactivated += 1


class _FakeManager:
    def __init__(self):
        self.adapter = _FakeAdapter()
        self.pipelines = {"fl": self.adapter}
        self.transformers = {"fl": torch.nn.Module()}
        self.caches = {"fl": _FakeCache()}
        self.lora_manager = _FakeLoraManager()
        self.resident = []

    def pipeline_for(self, profile, mode):
        return "fl"

    def ensure_resident(self, key):
        self.resident.append(key)
        return self.pipelines[key]


@pytest.fixture
def generator(monkeypatch):
    manager = _FakeManager()
    gen = Generator(manager, Settings())
    recorded = {"segments": [], "encoded": None}

    def _fake_segment(self, adapter, transformer, req, params, segment_length, segment_index,
                      first_frame, last_frame, references, seed):
        recorded["segments"].append(
            {
                "index": segment_index,
                "length": segment_length,
                "seed": seed,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "include_audio": req.include_audio,
            }
        )
        return (
            np.full((segment_length, 6, 8, 3), segment_index, dtype=np.uint8),
            np.zeros((segment_length * 10, 2), dtype=np.float32),
        )

    async def _fake_encode(self, fn, *args):
        if fn is generator_module.encode_mp4:
            recorded["encoded"] = args
            return None
        return fn(*args)

    monkeypatch.setattr(Generator, "_run_one_segment", _fake_segment)
    monkeypatch.setattr(Generator, "_run_encode", _fake_encode)
    try:
        yield gen, manager, recorded
    finally:
        asyncio.run(gen.shutdown())


def _run(coro):
    return asyncio.run(coro)


def test_all_model_mutation_happens_on_the_gpu_thread(generator):
    gen, manager, recorded = generator
    _run(gen.run(GenerationRequest(prompt="hi", num_frames=124, seed=1)))
    assert manager.lora_manager.threads, "LoRA lifecycle should have run"
    assert all(name.startswith("va-gpu") for name in manager.lora_manager.threads), (
        manager.lora_manager.threads
    )


def test_single_segment_run(generator):
    gen, manager, recorded = generator
    req = GenerationRequest(prompt="hi", num_frames=124, seed=5)
    frames, audio, sample_rate, params = _run(gen.run(req))

    assert params.mode == "t2v"
    assert params.num_frames == 124
    assert params.seed == 5
    assert frames.shape[0] == 124
    assert audio is not None and audio.shape[1] == 2
    assert sample_rate == 44100
    assert len(recorded["segments"]) == 1
    assert recorded["segments"][0]["seed"] == 5
    assert recorded["segments"][0]["include_audio"] is True
    # LoRA/cache lifecycle is balanced.
    assert manager.caches["fl"].ends == 1
    assert manager.lora_manager.deactivated == 1


def test_multi_segment_run_chains_frames_and_seeds(generator):
    gen, manager, recorded = generator
    req = GenerationRequest(prompt="hi", num_frames=685, seed=9)
    frames, _audio, _sample_rate, params = _run(gen.run(req))

    segments, final = plan_segments(685)
    assert len(recorded["segments"]) == len(segments)
    assert [s["length"] for s in recorded["segments"]] == segments
    assert [s["seed"] for s in recorded["segments"]] == [9 + i for i in range(len(segments))]
    assert recorded["segments"][0]["first_frame"] is None
    # Every later segment starts from the previous segment's last frame.
    for current in recorded["segments"][1:]:
        assert current["first_frame"] is not None
    assert frames.shape[0] == final


def test_seed_is_bounded(generator):
    gen, _manager, recorded = generator
    req = GenerationRequest(prompt="hi", num_frames=124, seed=2**63 - 1)
    params = _run(gen.run(req))[3]
    assert 0 <= params.seed <= 2**63 - 1
    for entry in recorded["segments"]:
        assert 0 <= entry["seed"] <= 2**63 - 1


def test_random_seed_is_recorded(generator):
    gen, _manager, _recorded = generator
    params = _run(gen.run(GenerationRequest(prompt="hi", num_frames=124)))[3]
    assert 0 <= params.seed <= 2**63 - 1


def test_audio_is_skipped_when_disabled(generator):
    gen, manager, recorded = generator
    req = GenerationRequest(prompt="hi", num_frames=124, include_audio=False)
    frames, audio, _sample_rate, _params = _run(gen.run(req))
    assert frames.shape[0] == 124
    assert audio is None
    assert recorded["segments"][0]["include_audio"] is False
    assert _sample_rate > 0


def test_resolve_leaves_the_request_untouched(generator):
    gen, _manager, _recorded = generator
    req = GenerationRequest(prompt="hi", duration_seconds=5.0)
    resolved = _run(gen.resolve(req))
    params, pipe_key = resolved.params, resolved.pipe_key
    assert pipe_key == "fl"
    assert req.duration_seconds == 5.0
    assert req.profile is None
    assert params.profile == Settings().DEFAULT_PROFILE
    assert abs(params.duration_seconds - params.num_frames / params.fps) < 1e-6


def test_render_mp4_rejects_empty_output(generator, tmp_path):
    gen, _manager, _recorded = generator
    req = GenerationRequest(prompt="hi", num_frames=124)

    async def _empty_run(_req):
        return np.zeros((0, 6, 8, 3), dtype=np.uint8), None, 44100, None

    gen.run = _empty_run  # type: ignore[assignment]
    with pytest.raises(ValueError):
        _run(gen.render_mp4(req, tmp_path / "out.mp4"))


def test_shutdown_is_idempotent_and_blocks_new_work(generator):
    gen, _manager, _recorded = generator
    _run(gen.shutdown())
    _run(gen.shutdown())
    with pytest.raises(RuntimeError):
        _run(gen.resolve(GenerationRequest(prompt="hi")))
