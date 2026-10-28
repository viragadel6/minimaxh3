from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from schemas import (
    ErrorBody,
    ErrorDetail,
    GenerationRequest,
    JobStatus,
    LoraSpec,
    ReferenceInput,
    ResolvedParams,
)

UTC = timezone.utc


def _params(**overrides):
    base = dict(
        width=1344,
        height=768,
        num_frames=124,
        fps=24,
        duration_seconds=124 / 24,
        seed=1,
        profile="fast",
        steps=20,
        mode="t2v",
        segments=[124],
    )
    base.update(overrides)
    return ResolvedParams(**base)


# ---------------------------------------------------------------------------
# input strictness
# ---------------------------------------------------------------------------
def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="hi", num_frame=124)  # typo
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="hi", lora=[])  # typo


def test_assignment_is_validated():
    req = GenerationRequest(prompt="hi")
    with pytest.raises(ValidationError):
        req.duration_seconds = -1.0
    with pytest.raises(ValidationError):
        req.num_inference_steps = 0


def test_blank_strings_are_rejected():
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="   ")
    with pytest.raises(ValidationError):
        ReferenceInput(type="image", source="")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 0.0])
def test_duration_must_be_positive_and_finite(value):
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="hi", duration_seconds=value)


def test_duration_and_frames_are_mutually_exclusive():
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="hi", duration_seconds=1.0, num_frames=124)


def test_num_frames_must_follow_grid():
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="hi", num_frames=100)


def test_steps_and_seed_bounds():
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="hi", num_inference_steps=0)
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="hi", seed=-1)
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="hi", seed=2**63)


def test_reference_lists_are_bounded():
    refs = [{"type": "image", "source": f"https://x/{i}.png"} for i in range(9)]
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="hi", references=refs)


# ---------------------------------------------------------------------------
# references / loras
# ---------------------------------------------------------------------------
def test_reference_time_window_validation():
    with pytest.raises(ValidationError):
        ReferenceInput(type="audio", source="a.wav", start_seconds=-1)
    with pytest.raises(ValidationError):
        ReferenceInput(type="audio", source="a.wav", start_seconds=2, end_seconds=1)
    with pytest.raises(ValidationError):
        ReferenceInput(type="image", source="a.png", start_seconds=0, end_seconds=1)
    with pytest.raises(ValidationError):
        ReferenceInput(type="audio", source="a.wav", strength=1.5)
    with pytest.raises(ValidationError):
        ReferenceInput(type="audio", source="a.wav", strength=float("nan"))


def test_lora_spec_validation():
    with pytest.raises(ValidationError):
        LoraSpec(source="x", scale=float("nan"))
    with pytest.raises(ValidationError):
        LoraSpec(source="x", scale=100)
    assert LoraSpec(source="x", weight_name="  ").weight_name is None


# ---------------------------------------------------------------------------
# mode inference
# ---------------------------------------------------------------------------
def test_mode_inference_basics():
    assert GenerationRequest(prompt="hi").effective_mode() == "t2v"
    assert (
        GenerationRequest(
            prompt="hi", first_frame={"source": "https://x/a.png"}
        ).effective_mode()
        == "i2v"
    )
    assert (
        GenerationRequest(
            prompt="hi",
            first_frame={"source": "https://x/a.png"},
            last_frame={"source": "https://x/b.png"},
        ).effective_mode()
        == "fl2v"
    )
    assert (
        GenerationRequest(
            prompt="hi", references=[{"type": "image", "source": "https://x/a.png"}]
        ).effective_mode()
        == "ref2v"
    )


def test_last_frame_without_first_frame_has_a_clear_error():
    with pytest.raises(ValidationError) as excinfo:
        GenerationRequest(prompt="hi", last_frame={"source": "https://x/b.png"})
    assert "last_frame" in str(excinfo.value)


def test_references_with_last_frame_has_a_clear_error():
    with pytest.raises(ValidationError) as excinfo:
        GenerationRequest(
            prompt="hi",
            last_frame={"source": "https://x/b.png"},
            references=[{"type": "image", "source": "https://x/a.png"}],
        )
    message = str(excinfo.value)
    assert "last_frame" in message or "mode" in message


def test_audio_references_are_allowed_for_frame_modes():
    req = GenerationRequest(
        prompt="hi",
        mode="i2v",
        first_frame={"source": "https://x/a.png"},
        references=[{"type": "audio", "source": "https://x/a.wav"}],
    )
    assert req.effective_mode() == "i2v"
    t2v = GenerationRequest(
        prompt="hi",
        references=[{"type": "audio", "source": "https://x/a.wav"}],
    )
    assert t2v.effective_mode() == "t2v"


def test_ref2v_allows_first_frame_and_rejects_last_frame():
    req = GenerationRequest(
        prompt="hi",
        mode="ref2v",
        first_frame={"source": "https://x/a.png"},
        references=[{"type": "image", "source": "https://x/b.png"}],
    )
    assert req.effective_mode() == "ref2v"
    with pytest.raises(ValidationError):
        GenerationRequest(
            prompt="hi",
            mode="ref2v",
            first_frame={"source": "https://x/a.png"},
            last_frame={"source": "https://x/c.png"},
            references=[{"type": "image", "source": "https://x/b.png"}],
        )


def test_aspect_auto_accepts_video_references():
    req = GenerationRequest(
        prompt="hi",
        aspect_ratio="auto",
        references=[{"type": "video", "source": "https://x/a.mp4"}],
    )
    assert req.aspect_source() == "https://x/a.mp4"


def test_aspect_source_priority_is_deterministic():
    req = GenerationRequest(
        prompt="hi",
        aspect_ratio="auto",
        first_frame={"source": "https://x/first.png"},
        references=[{"type": "image", "source": "https://x/ref.png"}],
    )
    assert req.aspect_source() == "https://x/first.png"


def test_aspect_auto_requires_a_source():
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="hi", aspect_ratio="auto")


# ---------------------------------------------------------------------------
# derived values do not pollute model_fields_set
# ---------------------------------------------------------------------------
def test_inference_does_not_mutate_fields():
    req = GenerationRequest(prompt="hi")
    assert req.duration_seconds is None
    assert req.mode is None
    assert "duration_seconds" not in req.model_fields_set
    assert "mode" not in req.model_fields_set
    assert req.effective_mode() == "t2v"
    assert req.effective_num_frames(24) == 124


def test_effective_frames_normalises_duration():
    req = GenerationRequest(prompt="hi", duration_seconds=5.0)
    frames = req.effective_num_frames(24)
    assert (frames - 5) % 17 == 0
    assert frames >= 124


# ---------------------------------------------------------------------------
# output models
# ---------------------------------------------------------------------------
def test_resolved_params_enforce_invariants():
    with pytest.raises(ValidationError):
        _params(width=0)
    with pytest.raises(ValidationError):
        _params(steps=0)
    with pytest.raises(ValidationError):
        _params(profile="ultra")
    with pytest.raises(ValidationError):
        _params(mode="nope")
    with pytest.raises(ValidationError):
        _params(duration_seconds=1.0)  # inconsistent with num_frames / fps
    with pytest.raises(ValidationError):
        _params(segments=[100])  # not 17n + 5
    with pytest.raises(ValidationError):
        _params(segments=[20])  # below the native minimum


def test_job_status_invariants():
    now = datetime.now(UTC)
    base = dict(id="abc", created_at=now, updated_at=now)
    with pytest.raises(ValidationError):
        JobStatus(**base, status="succeeded")  # no result
    with pytest.raises(ValidationError):
        JobStatus(**base, status="failed")  # no error
    with pytest.raises(ValidationError):
        JobStatus(**base, status="queued", error="boom")
    with pytest.raises(ValidationError):
        JobStatus(
            id="abc",
            status="queued",
            created_at=now,
            updated_at=now - timedelta(seconds=1),
        )
    ok = JobStatus(**base, status="running")
    assert ok.status == "running"


def test_error_body_requires_code_and_message():
    with pytest.raises(ValidationError):
        ErrorDetail(code="", message="x")
    with pytest.raises(ValidationError):
        ErrorDetail(code="x", message="")
    with pytest.raises(ValidationError):
        ErrorDetail(code="x", message="y", extra="nope")
    body = ErrorBody(error=ErrorDetail(code="x", message="y"))
    assert body.model_dump() == {"error": {"code": "x", "message": "y"}}
