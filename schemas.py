from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from shapes import validate_num_frames


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class MediaInput(StrictModel):
    source: str = Field(min_length=1)


class ReferenceInput(StrictModel):
    type: Literal["image", "video", "audio"]
    source: str = Field(min_length=1)
    strength: float | None = None
    start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_interval(self):
        if self.end_seconds is not None:
            if self.end_seconds <= (self.start_seconds or 0.0):
                raise ValueError("end_seconds must exceed start_seconds")
        return self


class LoraSpec(StrictModel):
    source: str = Field(min_length=1)
    weight_name: str | None = None
    scale: float = 1.0


class GenerationRequest(StrictModel):
    prompt: str = Field(min_length=1)
    mode: Literal["t2v", "i2v", "fl2v", "ref2v"] | None = None
    resolution: Literal["480p", "768p"] = "768p"
    aspect_ratio: Literal[
        "16:9", "9:16", "1:1", "4:3", "3:4", "auto"
    ] = "16:9"
    duration_seconds: float | None = Field(default=None, gt=0)
    num_frames: int | None = None
    profile: Literal["quality", "fast", "turbo"] | None = None
    num_inference_steps: int | None = Field(default=None, ge=1)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    first_frame: MediaInput | None = None
    last_frame: MediaInput | None = None
    references: list[ReferenceInput] = Field(default_factory=list)
    loras: list[LoraSpec] = Field(default_factory=list)
    include_audio: bool = True

    @field_validator("prompt")
    @classmethod
    def nonempty_prompt(cls, value):
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value

    @field_validator("num_frames")
    @classmethod
    def check_frame_grid(cls, value):
        return None if value is None else validate_num_frames(value)

    @model_validator(mode="after")
    def validate_request(self):
        if self.duration_seconds is None and self.num_frames is None:
            self.duration_seconds = 5.0
        elif self.duration_seconds is not None and self.num_frames is not None:
            raise ValueError("Specify duration_seconds or num_frames, not both")

        if self.mode is None:
            if self.references:
                self.mode = "ref2v"
            elif self.first_frame is not None and self.last_frame is not None:
                self.mode = "fl2v"
            elif self.first_frame is not None:
                self.mode = "i2v"
            else:
                self.mode = "t2v"

        if self.mode == "t2v":
            if self.first_frame or self.last_frame or self.references:
                raise ValueError("t2v does not accept conditioning media")
        elif self.mode == "i2v":
            if self.first_frame is None:
                raise ValueError("i2v requires first_frame")
            if self.last_frame is not None or self.references:
                raise ValueError("Use fl2v or ref2v for these additional conditions")
        elif self.mode == "fl2v":
            if self.first_frame is None or self.last_frame is None:
                raise ValueError("fl2v requires first_frame and last_frame")
            if self.references:
                raise ValueError("fl2v does not accept references")
        elif self.mode == "ref2v":
            if not self.references:
                raise ValueError("ref2v requires at least one reference")
            if self.last_frame is not None:
                raise ValueError("ref2v does not accept last_frame")

        if self.aspect_ratio == "auto":
            has_image = any(item.type == "image" for item in self.references)
            if self.first_frame is None and not has_image:
                raise ValueError(
                    "Automatic aspect ratio requires first_frame or an image reference"
                )
        return self


class ResolvedParams(StrictModel):
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


class JobStatus(StrictModel):
    id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: str
    updated_at: str
    result: ResolvedParams | None = None
    video_url: str | None = None
    error: str | None = None


class ErrorBody(StrictModel):
    error: dict[str, str]

    @field_validator("error")
    @classmethod
    def require_error_fields(cls, value):
        if not {"code", "message"}.issubset(value):
            raise ValueError("error must contain code and message")
        return value
