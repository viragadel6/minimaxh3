from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from shapes import validate_num_frames


class MediaInput(BaseModel):
    source: str


class ReferenceInput(BaseModel):
    type: Literal["image", "video", "audio"]
    source: str
    strength: float | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None


class LoraSpec(BaseModel):
    source: str
    weight_name: str | None = None
    scale: float = 1.0


class GenerationRequest(BaseModel):
    prompt: str
    mode: Literal["t2v", "i2v", "fl2v", "ref2v"] | None = None
    resolution: Literal["480p", "768p"] = "768p"
    aspect_ratio: Literal["16:9", "9:16", "1:1", "4:3", "3:4", "auto"] = "16:9"
    duration_seconds: float | None = None
    num_frames: int | None = None
    profile: Literal["quality", "fast", "turbo"] | None = None
    num_inference_steps: int | None = None
    seed: int | None = None
    first_frame: MediaInput | None = None
    last_frame: MediaInput | None = None
    references: list[ReferenceInput] = Field(default_factory=list)
    loras: list[LoraSpec] = Field(default_factory=list)
    include_audio: bool = True

    @model_validator(mode="after")
    def _validate_and_resolve(self) -> "GenerationRequest":
        if self.duration_seconds is not None and self.num_frames is not None:
            raise ValueError("duration_seconds and num_frames cannot both be set")
        if self.duration_seconds is None and self.num_frames is None:
            self.duration_seconds = 5.0
        if self.duration_seconds is not None and self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.num_frames is not None:
            validate_num_frames(self.num_frames)

        if self.mode is None:
            if len(self.references) > 0:
                self.mode = "ref2v"
            elif self.first_frame is not None and self.last_frame is not None:
                self.mode = "fl2v"
            elif self.first_frame is not None:
                self.mode = "i2v"
            else:
                self.mode = "t2v"

        if self.mode == "i2v":
            if self.first_frame is None:
                raise ValueError("mode 'i2v' requires first_frame")
            if self.last_frame is not None or len(self.references) > 0:
                raise ValueError("mode 'i2v' does not allow last_frame or references")
        elif self.mode == "fl2v":
            if self.first_frame is None or self.last_frame is None:
                raise ValueError("mode 'fl2v' requires first_frame and last_frame")
            if len(self.references) > 0:
                raise ValueError("mode 'fl2v' does not allow references")
        elif self.mode == "ref2v":
            if len(self.references) == 0:
                raise ValueError("mode 'ref2v' requires at least one reference")
            if self.last_frame is not None:
                raise ValueError("mode 'ref2v' requires last_frame to be unassigned")
        elif self.mode == "t2v":
            if (
                self.first_frame is not None
                or self.last_frame is not None
                or len(self.references) > 0
            ):
                raise ValueError(
                    "mode 't2v' requires first_frame, last_frame, and references to be unassigned"
                )

        if self.aspect_ratio == "auto":
            has_image_ref = any(r.type == "image" for r in self.references)
            if self.first_frame is None and not has_image_ref:
                raise ValueError(
                    "aspect_ratio='auto' requires first_frame or an image reference"
                )

        return self


class ResolvedParams(BaseModel):
    width: int
    height: int
    num_frames: int
    fps: int
    duration_seconds: float
    seed: int
    profile: str
    steps: int
    mode: str
    segments: list[int]


class JobStatus(BaseModel):
    id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: str
    updated_at: str
    result: ResolvedParams | None = None
    video_url: str | None = None
    error: str | None = None


class ErrorBody(BaseModel):
    error: dict[str, str]

    @field_validator("error")
    @classmethod
    def _validate_error_dict(cls, v: dict[str, str]) -> dict[str, str]:
        if "code" not in v or "message" not in v:
            raise ValueError("error must contain 'code' and 'message'")
        return v
