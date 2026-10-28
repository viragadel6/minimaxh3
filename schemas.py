"""Pydantic schemas for the public HTTP API.

Design notes
------------
* Input models are *strict*: ``extra="forbid"`` so typos are rejected instead of
  silently ignored, ``validate_assignment=True`` so post-construction mutation is
  validated, and every string field has ``min_length=1``.
* ``GenerationRequest`` never mutates itself inside a validator (that would pollute
  ``model_fields_set``).  Derived values are exposed through ``effective_*``
  accessors instead, so callers can distinguish "user supplied" from "inferred".
* Output models (``ResolvedParams``, ``JobStatus``) enforce their cross-field
  invariants so an inconsistent record can never be serialised as a successful
  HTTP response.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Literal, get_args

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from shapes import (
    MAX_SEGMENT_FRAMES,
    NATIVE_MIN_N,
    frames_for_duration,
    normalize_num_frames,
    validate_num_frames,
)

Mode = Literal["t2v", "i2v", "fl2v", "ref2v"]
Profile = Literal["quality", "fast", "turbo"]
ResolutionName = Literal["480p", "768p"]
AspectRatio = Literal["16:9", "9:16", "1:1", "4:3", "3:4", "auto"]
JobState = Literal["queued", "running", "succeeded", "failed"]
ReferenceType = Literal["image", "video", "audio"]

MODES: tuple[str, ...] = get_args(Mode)
PROFILES: tuple[str, ...] = get_args(Profile)
RESOLUTION_NAMES: tuple[str, ...] = get_args(ResolutionName)
ASPECT_RATIOS: tuple[str, ...] = get_args(AspectRatio)
JOB_STATES: tuple[str, ...] = get_args(JobState)
REFERENCE_TYPES: tuple[str, ...] = get_args(ReferenceType)

MAX_SEED = 2**63 - 1
DEFAULT_DURATION_SECONDS = 5.0
#: Only audio can accompany a mode that is defined by frames.
AUDIO_ONLY = "audio"


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )


def _finite(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


class MediaInput(_StrictModel):
    source: str = Field(min_length=1)


class ReferenceInput(_StrictModel):
    type: ReferenceType
    source: str = Field(min_length=1)
    strength: float | None = Field(default=None, ge=0.0, le=1.0)
    start_seconds: float | None = Field(default=None, ge=0.0)
    end_seconds: float | None = Field(default=None, ge=0.0)

    @field_validator("strength", "start_seconds", "end_seconds")
    @classmethod
    def _validate_finite(cls, value: float | None) -> float | None:
        return _finite(value, "value")

    @model_validator(mode="after")
    def _validate_reference(self) -> "ReferenceInput":
        start, end = self.start_seconds, self.end_seconds
        if end is not None:
            if start is None:
                raise ValueError("end_seconds requires start_seconds")
            if end <= start:
                raise ValueError("end_seconds must be greater than start_seconds")
        if self.type == "image" and (start is not None or end is not None):
            raise ValueError("start_seconds/end_seconds are only valid for video or audio references")
        return self


class LoraSpec(_StrictModel):
    source: str = Field(min_length=1)
    weight_name: str | None = None
    scale: float = Field(default=1.0, ge=0.0, le=10.0)

    @field_validator("weight_name")
    @classmethod
    def _normalize_weight_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("scale")
    @classmethod
    def _validate_scale(cls, value: float) -> float:
        if not math.isfinite(float(value)):
            raise ValueError("scale must be finite")
        return float(value)


class GenerationRequest(_StrictModel):
    prompt: str = Field(min_length=1)
    mode: Mode | None = None
    resolution: ResolutionName = "768p"
    aspect_ratio: AspectRatio = "16:9"
    duration_seconds: float | None = Field(default=None, gt=0.0)
    num_frames: int | None = Field(default=None, gt=0)
    profile: Profile | None = None
    num_inference_steps: int | None = Field(default=None, gt=0, le=1000)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    first_frame: MediaInput | None = None
    last_frame: MediaInput | None = None
    references: list[ReferenceInput] = Field(
        default_factory=list, max_length=8
    )
    loras: list[LoraSpec] = Field(default_factory=list, max_length=8)
    include_audio: bool = True

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    @field_validator("duration_seconds")
    @classmethod
    def _validate_duration(cls, value: float | None) -> float | None:
        return _finite(value, "duration_seconds")

    @model_validator(mode="after")
    def _validate_request(self) -> "GenerationRequest":
        if self.duration_seconds is not None and self.num_frames is not None:
            raise ValueError("duration_seconds and num_frames cannot both be set")
        if self.duration_seconds is not None:
            if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
                raise ValueError("duration_seconds must be a positive finite number")
        if self.num_frames is not None:
            validate_num_frames(self.num_frames)
        if self.aspect_ratio == "auto" and self.aspect_source() is None:
            raise ValueError(
                "aspect_ratio='auto' requires first_frame, last_frame or an "
                "image/video reference to read the size from"
            )
        # Runs the (pure) inference + constraint checks eagerly so an invalid
        # request fails at construction time, not halfway through a generation.
        self._infer_mode()
        return self

    # ------------------------------------------------------------------
    # effective values (never mutates the model)
    # ------------------------------------------------------------------
    def _visual_references(self) -> list[ReferenceInput]:
        return [r for r in self.references if r.type in ("image", "video")]

    def _audio_references(self) -> list[ReferenceInput]:
        return [r for r in self.references if r.type == AUDIO_ONLY]

    def _infer_mode(self) -> str:
        visual = self._visual_references()
        if self.mode is not None:
            self._check_mode(self.mode, visual)
            return self.mode
        if self.first_frame is None and self.last_frame is not None:
            raise ValueError(
                "mode cannot be inferred: last_frame requires first_frame; "
                "set mode='fl2v' and provide both frames"
            )
        if visual:
            if self.last_frame is not None:
                raise ValueError(
                    "mode cannot be inferred: last_frame cannot be combined with "
                    "reference images/videos; set mode explicitly"
                )
            self._check_mode("ref2v", visual)
            return "ref2v"
        if self.first_frame is not None and self.last_frame is not None:
            self._check_mode("fl2v", visual)
            return "fl2v"
        if self.first_frame is not None:
            self._check_mode("i2v", visual)
            return "i2v"
        self._check_mode("t2v", visual)
        return "t2v"

    def _check_mode(self, mode: str, visual: list[ReferenceInput]) -> None:
        # Audio references are additive conditioning for every mode.
        if mode == "i2v":
            if self.first_frame is None:
                raise ValueError("mode 'i2v' requires first_frame")
            if self.last_frame is not None:
                raise ValueError("mode 'i2v' does not allow last_frame")
            if visual:
                raise ValueError(
                    "mode 'i2v' does not allow image or video references "
                    "(only audio references)"
                )
        elif mode == "fl2v":
            if self.first_frame is None or self.last_frame is None:
                raise ValueError("mode 'fl2v' requires first_frame and last_frame")
            if visual:
                raise ValueError(
                    "mode 'fl2v' does not allow image or video references "
                    "(only audio references)"
                )
        elif mode == "ref2v":
            if not visual:
                raise ValueError("mode 'ref2v' requires at least one image or video reference")
            if self.last_frame is not None:
                raise ValueError("mode 'ref2v' requires last_frame to be unassigned")
            # first_frame is optional in ref2v (an additional conditioning frame);
            # the contract is symmetric with the last_frame rule above.
        elif mode == "t2v":
            if self.first_frame is not None or self.last_frame is not None or visual:
                raise ValueError(
                    "mode 't2v' requires first_frame, last_frame and image/video "
                    "references to be unassigned"
                )
        else:  # pragma: no cover - guarded by Literal typing
            raise ValueError(f"Unknown mode: {mode}")

    def effective_mode(self) -> str:
        return self._infer_mode()

    def effective_duration_seconds(self, fps: int = 24) -> float:
        if self.duration_seconds is not None:
            return float(self.duration_seconds)
        return self.effective_num_frames(fps) / float(fps)

    def effective_num_frames(self, fps: int = 24, max_frames: int | None = None) -> int:
        """Frame count honouring ``num_frames`` or ``duration_seconds``."""
        if self.num_frames is not None:
            frames = validate_num_frames(self.num_frames)
        else:
            seconds = (
                self.duration_seconds
                if self.duration_seconds is not None
                else DEFAULT_DURATION_SECONDS
            )
            frames = normalize_num_frames(frames_for_duration(seconds, fps))
        if max_frames is not None:
            frames = min(frames, int(max_frames))
            frames = normalize_num_frames(frames)
        return frames

    def aspect_source(self) -> str | None:
        """Deterministic source used to resolve ``aspect_ratio="auto"``.

        Priority: ``first_frame`` > ``last_frame`` > first image/video reference.
        """
        if self.first_frame is not None:
            return self.first_frame.source
        if self.last_frame is not None:
            return self.last_frame.source
        for ref in self.references:
            if ref.type in ("image", "video"):
                return ref.source
        return None


class ResolvedParams(_StrictModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    num_frames: int = Field(gt=0)
    fps: int = Field(gt=0)
    duration_seconds: float = Field(gt=0.0)
    seed: int = Field(ge=0, le=MAX_SEED)
    profile: Profile
    steps: int = Field(gt=0)
    mode: Mode
    segments: list[int] = Field(min_length=1)

    @field_validator("segments")
    @classmethod
    def _validate_segments(cls, segments: list[int]) -> list[int]:
        for seg in segments:
            if seg < 17 * NATIVE_MIN_N + 5 or seg > MAX_SEGMENT_FRAMES:
                raise ValueError(
                    f"segment length {seg} outside native bounds "
                    f"[{17 * NATIVE_MIN_N + 5}, {MAX_SEGMENT_FRAMES}]"
                )
            if (seg - 5) % 17 != 0:
                raise ValueError("segment lengths must follow 17n + 5")
        return segments

    @model_validator(mode="after")
    def _validate_consistency(self) -> "ResolvedParams":
        if not math.isfinite(self.duration_seconds):
            raise ValueError("duration_seconds must be finite")
        expected = self.num_frames / self.fps
        if abs(self.duration_seconds - expected) > max(1e-3, 1e-6 * expected):
            raise ValueError(
                "duration_seconds must equal num_frames / fps "
                f"({self.num_frames}/{self.fps}={expected}, got {self.duration_seconds})"
            )
        coverage = 1 + sum(seg - 1 for seg in self.segments)
        if coverage < self.num_frames:
            raise ValueError(
                f"segments cover {coverage} frames but num_frames is {self.num_frames}"
            )
        return self


class JobStatus(_StrictModel):
    id: str = Field(min_length=1)
    status: JobState
    created_at: AwareDatetime
    updated_at: AwareDatetime
    result: ResolvedParams | None = None
    video_url: str | None = Field(default=None, min_length=1)
    error: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_state(self) -> "JobStatus":
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not be earlier than created_at")
        if self.status == "succeeded":
            if self.result is None:
                raise ValueError("a succeeded job must carry a result")
            if self.error is not None:
                raise ValueError("a succeeded job must not carry an error")
        elif self.status == "failed":
            if self.error is None:
                raise ValueError("a failed job must carry an error")
        elif self.status == "queued":
            if self.result is not None or self.error is not None:
                raise ValueError("a queued job must not carry a result or an error")
        elif self.status == "running":
            if self.error is not None:
                raise ValueError("a running job must not carry an error")
        return self


class ErrorDetail(_StrictModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ErrorBody(_StrictModel):
    error: ErrorDetail


class ModelList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_repo_id: str
    modes: list[str]
    profiles: list[str]
    default_profile: str
    resolutions: dict[str, dict[str, tuple[int, int]]]
    fps: int
    frame_grid: dict[str, int | str]
    native_segment_bounds: dict[str, int]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
