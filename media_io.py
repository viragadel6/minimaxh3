"""Media IO helpers: fetch, decode, normalise, stitch, trim and encode.

Every public helper validates its arguments and raises :class:`ValueError` (or
:class:`RuntimeError` for subprocess failures) with an actionable message.  Pure
numeric helpers (``to_uint8_frames``, ``to_stereo``, ``trim``, ``stitch_segments``)
accept both :mod:`numpy` arrays and :mod:`torch` tensors; torch is imported lazily
so the API container does not need it.
"""

from __future__ import annotations

import asyncio
import base64
import io
import math
import os
import subprocess
import tempfile
import threading
import urllib.parse
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, Sequence

import httpx
import numpy as np
from PIL import Image, ImageOps

from shapes import FPS

try:  # torch is optional: the API container never imports it.
    import torch
except ImportError:  # pragma: no cover - exercised in the API image
    torch = None  # type: ignore[assignment]

ValueRange = Literal["auto", "0-1", "-1-1", "0-255"]
Layout = Literal["auto", "channel_last", "channel_first", "channel_first_stacked"]

_MAX_LOCAL_SOURCE_LEN = 4096
_HTTP_TIMEOUT = httpx.Timeout(120.0, connect=15.0)
_HTTP_LIMITS = httpx.Limits(max_connections=32, max_keepalive_connections=16)
_CHUNK_BYTES = 8 * 1024 * 1024

_http_client: httpx.AsyncClient | None = None
_http_lock: asyncio.Lock | None = None


# ---------------------------------------------------------------------------
# HTTP client / fetching
# ---------------------------------------------------------------------------
def _get_http_lock() -> asyncio.Lock:
    global _http_lock
    if _http_lock is None:
        _http_lock = asyncio.Lock()
    return _http_lock


async def _get_http_client() -> httpx.AsyncClient:
    """Return a lazily created, process wide client so connections are pooled."""
    global _http_client
    client = _http_client
    if client is not None and not client.is_closed:
        return client
    async with _get_http_lock():
        if _http_client is None or _http_client.is_closed:
            _http_client = httpx.AsyncClient(
                follow_redirects=True,
                timeout=_HTTP_TIMEOUT,
                limits=_HTTP_LIMITS,
            )
        return _http_client


async def aclose_http_client() -> None:
    global _http_client
    client = _http_client
    _http_client = None
    if client is not None and not client.is_closed:
        await client.aclose()


def _split_scheme(value: str) -> str:
    head, sep, _ = value.partition(":")
    return head.strip().lower() if sep else ""


def _decode_data_uri(uri: str) -> bytes:
    header, sep, payload = uri.partition(",")
    if not sep:
        raise ValueError("malformed data: URI (missing ',')")
    meta = header[len("data:") :] if header.lower().startswith("data:") else header
    params = [p.strip().lower() for p in meta.split(";") if p.strip()]
    is_base64 = "base64" in params
    if is_base64:
        cleaned = "".join(urllib.parse.unquote(payload).split())
        try:
            return base64.b64decode(cleaned, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise ValueError("malformed base64 payload in data: URI") from exc
    return urllib.parse.unquote_to_bytes(payload)


def _read_local_bytes(source: str) -> bytes:
    if len(source) > _MAX_LOCAL_SOURCE_LEN:
        raise ValueError(f"media source is too long to be a path: {source[:64]}...")
    try:
        path = Path(source).expanduser()
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid media source {source!r}: {exc}") from exc
    try:
        if not path.is_file():
            raise ValueError(f"unsupported media source: {source}")
        return path.read_bytes()
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(f"cannot read media source {source!r}: {exc}") from exc


async def fetch_bytes(source: str) -> bytes:
    """Resolve ``data:``/``http(s):``/local-path media sources to raw bytes."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("media source must be a non-empty string")
    scheme = _split_scheme(source)
    if scheme == "data":
        return _decode_data_uri(source)
    if scheme in ("http", "https"):
        client = await _get_http_client()
        last_exc: Exception | None = None
        for _ in range(2):
            try:
                response = await client.get(source)
                response.raise_for_status()
                return response.content
            except httpx.HTTPStatusError as exc:
                raise ValueError(
                    f"HTTP {exc.response.status_code} while fetching {source}"
                ) from exc
            except httpx.TransportError as exc:  # retry once on transient errors
                last_exc = exc
        raise ValueError(f"failed to fetch {source}: {last_exc}")
    # Local files: never block the event loop on disk IO.
    return await asyncio.to_thread(_read_local_bytes, source)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def _validate_size(width: int, height: int) -> tuple[int, int]:
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    return width, height


def _to_rgb(img: Image.Image) -> Image.Image:
    """Convert to RGB, compositing any alpha channel over white instead of dropping it."""
    if img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info
    ):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img.convert("RGB")


def load_image(data: bytes, width: int, height: int) -> Image.Image:
    """Decode ``data``, cover-crop it to ``width`` x ``height``."""
    width, height = _validate_size(width, height)
    with Image.open(io.BytesIO(data)) as img:
        transposed = _to_rgb(ImageOps.exif_transpose(img))
        src_w, src_h = transposed.size
        if src_w <= 0 or src_h <= 0:
            raise ValueError(f"decoded image has invalid size {src_w}x{src_h}")
        scale = max(width / src_w, height / src_h)
        new_w = max(width, int(round(src_w * scale)))
        new_h = max(height, int(round(src_h * scale)))
        resized = transposed.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (new_w - width) // 2
        top = (new_h - height) // 2
        return resized.crop((left, top, left + width, top + height))


def image_size(data: bytes) -> tuple[int, int]:
    """Return the display size without decoding pixel data (EXIF aware)."""
    with Image.open(io.BytesIO(data)) as img:
        width, height = img.size
        try:
            orientation = img.getexif().get(0x0112, 1)
        except Exception:  # pragma: no cover - malformed EXIF
            orientation = 1
        if orientation in (5, 6, 7, 8):
            width, height = height, width
        if width <= 0 or height <= 0:
            raise ValueError(f"decoded image has invalid size {width}x{height}")
        return width, height


# ---------------------------------------------------------------------------
# FFmpeg decoding
# ---------------------------------------------------------------------------
def _finite_or_none(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _format_seconds(value: float) -> str:
    """Format a timestamp for FFmpeg (never scientific notation)."""
    text = f"{value:.6f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _time_args(start: float | None, end: float | None) -> list[str]:
    """Return ``-ss``/``-t`` output options for a (start, end) window."""
    start_v = _finite_or_none(start, "start_seconds")
    end_v = _finite_or_none(end, "end_seconds")
    args: list[str] = []
    if start_v is not None and start_v < 0:
        raise ValueError("start_seconds must be greater than or equal to 0")
    if end_v is not None:
        effective_start = start_v if start_v is not None else 0.0
        duration = end_v - effective_start
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("end_seconds must be greater than start_seconds")
    if start_v is not None and start_v > 0:
        args.extend(["-ss", _format_seconds(start_v)])
    if end_v is not None:
        args.extend(["-t", _format_seconds(end_v - (start_v or 0.0))])
    return args


def _seek_args(start: float | None, end: float | None) -> tuple[list[str], list[str]]:
    """Split the trim window into (input args, output args).

    A fast input seek is only usable for seekable files; on ``pipe:0`` FFmpeg
    must seek with the decoder, so the same options are applied to the output.
    """
    start_v = _finite_or_none(start, "start_seconds")
    end_v = _finite_or_none(end, "end_seconds")
    if start_v is not None and start_v < 0:
        raise ValueError("start_seconds must be greater than or equal to 0")
    if end_v is not None:
        effective_start = start_v if start_v is not None else 0.0
        if end_v - effective_start <= 0:
            raise ValueError("end_seconds must be greater than start_seconds")
    input_args: list[str] = []
    output_args: list[str] = _time_args(start_v, end_v)
    if start_v is not None and start_v > 0:
        input_args = ["-ss", _format_seconds(start_v)]
    return input_args, output_args


def _ffmpeg_base() -> list[str]:
    return ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"]


def _run_ffmpeg(
    cmd: list[str],
    *,
    input_bytes: bytes | None = None,
    timeout: float | None = None,
) -> bytes:
    try:
        proc = subprocess.run(
            cmd,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"FFmpeg timed out after {timeout}s") from exc
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg failed (exit {proc.returncode}): {err}")
    return proc.stdout


def _looks_seekable_container(data: bytes) -> bool:
    """MP4/MOV place ``moov`` at the end, which cannot be streamed from a pipe."""
    if len(data) < 12:
        return False
    return data[4:8] == b"ftyp"


def _run_ffmpeg_decode(
    data: bytes,
    post_args: Sequence[str],
    *,
    start: float | None = None,
    end: float | None = None,
    timeout: float | None = 600.0,
) -> bytes:
    """Decode ``data`` with FFmpeg, preferring a temp file for seekable containers."""
    input_args, output_args = _seek_args(start, end)
    if _looks_seekable_container(data):
        return _decode_via_file(data, input_args, output_args, post_args, timeout)

    pipe_cmd = [
        *_ffmpeg_base(),
        "-i",
        "pipe:0",
        *output_args,
        *post_args,
        "pipe:1",
    ]
    try:
        out = _run_ffmpeg(pipe_cmd, input_bytes=data, timeout=timeout)
    except RuntimeError:
        out = b""
    if out:
        return out
    # Piped decoding can legitimately produce nothing for non-streamable input.
    return _decode_via_file(data, input_args, output_args, post_args, timeout)


def _decode_via_file(
    data: bytes,
    input_args: Sequence[str],
    output_args: Sequence[str],
    post_args: Sequence[str],
    timeout: float | None,
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="va-media-") as tmpdir:
        in_path = Path(tmpdir) / "input.bin"
        in_path.write_bytes(data)
        cmd = [
            *_ffmpeg_base(),
            *input_args,
            "-i",
            str(in_path),
            *output_args,
            *post_args,
            "pipe:1",
        ]
        return _run_ffmpeg(cmd, timeout=timeout)


# ---------------------------------------------------------------------------
# Video / audio decoding
# ---------------------------------------------------------------------------
def load_video_frames(
    data: bytes,
    width: int,
    height: int,
    start: float | None = None,
    end: float | None = None,
    fps: int = FPS,
) -> np.ndarray:
    """Decode a video to ``(frames, height, width, 3)`` uint8 frames."""
    width, height = _validate_size(width, height)
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")
    vf = (
        f"fps={fps},"
        f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height},setsar=1,format=rgb24"
    )
    raw = _run_ffmpeg_decode(
        data,
        ["-an", "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo"],
        start=start,
        end=end,
    )
    frame_size = width * height * 3
    num_frames = len(raw) // frame_size
    if num_frames == 0:
        raise ValueError("decoded video contains no frames")
    # Read exactly num_frames worth of bytes; avoids copying the whole payload.
    frames = np.frombuffer(raw, dtype=np.uint8, count=num_frames * frame_size)
    return np.ascontiguousarray(frames.reshape(num_frames, height, width, 3))


def load_audio(
    data: bytes,
    sample_rate: int,
    start: float | None = None,
    end: float | None = None,
) -> np.ndarray:
    """Decode audio to ``(samples, 2)`` float32 in ``[-1, 1]``."""
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise ValueError("sample_rate must be an integer")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    raw = _run_ffmpeg_decode(
        data,
        [
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(sample_rate),
            "-acodec",
            "pcm_f32le",
            "-f",
            "f32le",
        ],
        start=start,
        end=end,
    )
    num_samples = len(raw) // 8
    if num_samples == 0:
        raise ValueError("decoded audio contains no samples")
    # Little endian is explicit so the samples decode identically on big endian hosts.
    samples = np.frombuffer(raw, dtype="<f4", count=num_samples * 2)
    return np.ascontiguousarray(samples.reshape(num_samples, 2), dtype=np.float32)


# ---------------------------------------------------------------------------
# Layout / dtype normalisation
# ---------------------------------------------------------------------------
def _permute_frames(x, order: tuple[int, ...]):
    if hasattr(x, "permute"):
        return x.permute(*order)
    return np.transpose(x, order)


def _resolve_layout(shape: tuple[int, ...], layout: str) -> str:
    """Return one of ``channel_last``, ``channel_first``, ``channel_first_stacked``."""
    if layout != "auto":
        if layout not in ("channel_last", "channel_first", "channel_first_stacked"):
            raise ValueError(f"unknown layout {layout!r}")
        return layout
    channels_last_ok = shape[-1] in (1, 3, 4)
    channels_second_ok = shape[1] in (1, 3, 4)
    channels_first_ok = shape[0] in (1, 3, 4)
    if channels_last_ok and not channels_second_ok and not channels_first_ok:
        return "channel_last"
    if channels_second_ok and not channels_last_ok and not channels_first_ok:
        return "channel_first"
    if channels_first_ok and not channels_last_ok and not channels_second_ok:
        return "channel_first_stacked"
    if channels_last_ok:
        return "channel_last"
    raise ValueError(
        f"ambiguous video shape {shape}: pass layout="
        "'channel_last' (F,H,W,C), 'channel_first' (F,C,H,W) or "
        "'channel_first_stacked' (C,F,H,W)"
    )


def _normalize_4d_layout(x, layout: str = "auto"):
    """Normalise a 4D/5D video to ``(frames, height, width, channels)``."""
    ndim = int(x.ndim)
    if ndim == 5:
        if int(x.shape[0]) != 1:
            raise ValueError(
                f"batched video input with batch={int(x.shape[0])} is not supported; "
                "pass a single video"
            )
        x = x[0]
        ndim = 4
    if ndim != 4:
        raise ValueError(f"expected a 4D or 5D video, got {ndim}D")
    shape = tuple(int(s) for s in x.shape)
    if min(shape) <= 0:
        raise ValueError(f"video has an empty dimension: {shape}")
    resolved = _resolve_layout(shape, layout)
    if resolved == "channel_last":
        return x
    if resolved == "channel_first":
        return _permute_frames(x, (0, 2, 3, 1))
    return _permute_frames(x, (1, 2, 3, 0))


def _expand_channels(x, is_torch: bool):
    channels = int(x.shape[-1])
    if channels == 1:
        if is_torch:
            return x.expand(*x.shape[:-1], 3)
        return np.repeat(x, 3, axis=-1)
    if channels > 3:
        return x[..., :3]
    return x


def _range_params(
    value_range: ValueRange, vmin: float, vmax: float
) -> tuple[float, float]:
    """Return ``(scale, bias)`` mapping the source range onto ``[0, 255]``."""
    if value_range == "0-1":
        return 255.0, 0.0
    if value_range == "-1-1":
        return 127.5, 127.5
    if value_range == "0-255":
        return 1.0, 0.0
    if value_range != "auto":
        raise ValueError(f"unknown value_range {value_range!r}")
    tol = 1e-3
    if vmax > 1.0 + tol:
        # Includes float videos in 0..255 that carry tiny negative ringing.
        return 1.0, 0.0
    if vmin < -tol:
        return 127.5, 127.5
    return 255.0, 0.0


def _pil_frames_to_array(frames: Sequence[Image.Image]) -> np.ndarray:
    return np.stack(
        [
            np.asarray(_to_rgb(ImageOps.exif_transpose(img)), dtype=np.uint8)
            for img in frames
        ],
        axis=0,
    )


def to_uint8_frames(
    video: Any, value_range: ValueRange = "auto", layout: str = "auto"
) -> np.ndarray:
    """Coerce video-like input into ``(frames, H, W, 3)`` uint8 numpy frames."""
    if isinstance(video, (list, tuple)):
        if len(video) == 0:
            raise ValueError("empty video frame list")
        if isinstance(video[0], Image.Image):
            arr = _pil_frames_to_array(video)
            return np.ascontiguousarray(arr)
        items = list(video)
        if len(items) == 1 and not isinstance(items[0], (list, tuple)):
            single = items[0]
            ndim = int(single.ndim) if hasattr(single, "ndim") else np.ndim(single)
            if ndim == 3:
                # A single frame: add the frame axis instead of unwrapping to 3D.
                items = [single[None]]
            else:
                items = [single]
        converted: list[np.ndarray] = []
        for item in items:
            converted.append(to_uint8_frames(item, value_range, layout))
        if len(converted) == 1:
            return converted[0]
        # Concatenate along the frame axis (stacking would add a bogus dimension).
        return np.ascontiguousarray(np.concatenate(converted, axis=0))

    if torch is not None and isinstance(video, torch.Tensor):
        t = _normalize_4d_layout(video.detach(), layout)
        t = _expand_channels(t, True)
        if t.shape[0] == 0:
            raise ValueError("video contains no frames")
        if t.dtype == torch.uint8:
            out = t
        elif t.dtype.is_floating_point:
            tf = t.float()
            finite = torch.isfinite(tf)
            if not bool(finite.any()):
                tf = torch.zeros_like(tf)
                vmin = vmax = 0.0
            else:
                vmin = float(tf[finite].min())
                vmax = float(tf[finite].max())
            scale, bias = _range_params(value_range, vmin, vmax)
            # One fused affine transform: no nested `where` evaluating both branches.
            tf = tf * scale + bias
            tf = torch.nan_to_num(tf, nan=0.0, posinf=255.0, neginf=0.0)
            out = tf.round().clamp_(0.0, 255.0).to(torch.uint8)
        else:
            info = torch.iinfo(t.dtype)
            # Integer buffers (uint16 etc.) are scaled instead of clipped.
            tf = t.float() * (255.0 / float(info.max))
            out = tf.round().clamp_(0.0, 255.0).to(torch.uint8)
        out = out.contiguous()
        return out.cpu().numpy()

    arr = np.asarray(video)
    if arr.ndim == 3:
        arr = arr[None]
    arr = _normalize_4d_layout(arr, layout)
    arr = _expand_channels(arr, False)
    if arr.shape[0] == 0:
        raise ValueError("video contains no frames")
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        max_value = float(np.iinfo(arr.dtype).max)
        arr_f = arr.astype(np.float32) * (255.0 / max_value)
    else:
        arr_f = arr.astype(np.float32)
        finite = np.isfinite(arr_f)
        if not finite.any():
            vmin = vmax = 0.0
        else:
            vmin = float(arr_f[finite].min())
            vmax = float(arr_f[finite].max())
        scale, bias = _range_params(value_range, vmin, vmax)
        arr_f = arr_f * scale + bias
        arr_f = np.nan_to_num(arr_f, nan=0.0, posinf=255.0, neginf=0.0)
    arr_f = np.clip(np.rint(arr_f), 0.0, 255.0)
    return np.ascontiguousarray(arr_f.astype(np.uint8))


# ---------------------------------------------------------------------------
# Audio normalisation
# ---------------------------------------------------------------------------
def _audio_to_numpy(audio: Any) -> np.ndarray:
    if torch is not None and isinstance(audio, torch.Tensor):
        if audio.is_complex():
            raise ValueError("complex audio tensors are not supported")
        if audio.dtype.is_floating_point:
            return audio.detach().to(torch.float32).cpu().numpy()
        info = torch.iinfo(audio.dtype)
        scale = 1.0 / max(abs(float(info.min)), float(info.max))
        return audio.detach().to(torch.float32).cpu().numpy() * scale
    arr = np.asarray(audio)
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32, copy=False)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        scale = 1.0 / max(abs(float(info.min)), float(info.max))
        return arr.astype(np.float32) * scale
    return arr.astype(np.float32)


def to_stereo(audio: Any) -> np.ndarray:
    """Coerce audio input into a ``(samples, 2)`` float32 array in ``[-1, 1]``."""
    if audio is None:
        return np.zeros((0, 2), dtype=np.float32)
    if isinstance(audio, (list, tuple)):
        if len(audio) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if len(audio) == 1 and not isinstance(audio[0], (list, tuple)):
            audio = audio[0]
        else:
            parts = [_audio_to_numpy(item) for item in audio]
            parts = [p.reshape(-1) if p.ndim == 0 else p for p in parts]
            if all(p.ndim == 1 for p in parts):
                arr = np.concatenate(parts, axis=0)
            else:
                arr = np.concatenate([p if p.ndim == 2 else p[:, None] for p in parts], axis=0)
            return _finalize_stereo(arr)

    arr = _audio_to_numpy(audio)
    return _finalize_stereo(arr)


def _finalize_stereo(arr: np.ndarray) -> np.ndarray:
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"expected 1D or 2D audio, got {arr.ndim}D")
    if arr.size == 0 or arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    # Channel-first only when the second axis cannot be a channel count.
    if arr.shape[1] > 2 and arr.shape[0] in (1, 2):
        arr = arr.T
    if arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if arr.shape[1] == 1:
        arr = np.repeat(arr, 2, axis=1)
    elif arr.shape[1] > 2:
        # Downmix instead of silently dropping centre/surround channels.
        mono = arr.mean(axis=1, keepdims=True)
        arr = np.repeat(mono, 2, axis=1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _fit_audio_length(stereo: np.ndarray, num_samples: int) -> np.ndarray:
    """Pad with silence or truncate ``stereo`` to exactly ``num_samples`` rows."""
    if num_samples < 0:
        raise ValueError("num_samples must be >= 0")
    if len(stereo) >= num_samples:
        return np.ascontiguousarray(stereo[:num_samples], dtype=np.float32)
    pad = num_samples - len(stereo)
    return np.ascontiguousarray(
        np.pad(stereo, ((0, pad), (0, 0)), mode="constant"), dtype=np.float32
    )


def _samples_for_frames(frames: int, sample_rate: int, fps: int) -> int:
    return int(round(frames * sample_rate / fps))


def _crossfade(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Equal-gain crossfade (fades sum to one, so correlated audio keeps its level)."""
    overlap = len(left)
    if overlap == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if overlap == 1:
        return np.ascontiguousarray((left + right) * 0.5, dtype=np.float32)
    theta = np.linspace(0.0, math.pi / 2.0, overlap, dtype=np.float32)[:, None]
    fade_out = np.cos(theta)
    fade_in = np.sin(theta)
    total = fade_out + fade_in
    blended = left * (fade_out / total) + right * (fade_in / total)
    return np.ascontiguousarray(blended, dtype=np.float32)


# ---------------------------------------------------------------------------
# Stitching / trimming
# ---------------------------------------------------------------------------
def stitch_segments(
    video_list: Sequence[Any],
    audio_list: Sequence[Any] | None,
    sample_rate: int,
    fps: int,
    overlap_frames: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Concatenate per-segment outputs, removing the shared boundary frame."""
    if not video_list:
        raise ValueError("video_list cannot be empty")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if isinstance(overlap_frames, bool) or not isinstance(overlap_frames, int):
        raise ValueError("overlap_frames must be an integer")
    if overlap_frames < 0 or overlap_frames >= fps:
        raise ValueError("overlap_frames must be in [0, fps)")

    videos = [to_uint8_frames(seg) for seg in video_list]
    if overlap_frames:
        for idx, seg in enumerate(videos):
            if idx > 0 and len(seg) <= overlap_frames:
                raise ValueError(
                    f"segment {idx} has {len(seg)} frames and cannot supply the "
                    f"{overlap_frames} frame(s) shared with the previous segment"
                )
    pieces = [videos[0]] + [seg[overlap_frames:] for seg in videos[1:]]
    stitched_video = np.ascontiguousarray(np.concatenate(pieces, axis=0))

    if audio_list is None:
        return stitched_video, None
    if len(audio_list) != len(videos):
        raise ValueError(
            f"audio_list has {len(audio_list)} entries but video_list has {len(videos)}"
        )
    if all(a is None for a in audio_list):
        return stitched_video, None

    # Each segment's own audio duration, so a silent segment keeps sync.
    segment_audio: list[np.ndarray] = []
    for seg, aud in zip(videos, audio_list, strict=True):
        expected = _samples_for_frames(len(seg), sample_rate, fps)
        if aud is None:
            segment_audio.append(np.zeros((expected, 2), dtype=np.float32))
        else:
            segment_audio.append(_fit_audio_length(to_stereo(aud), expected))

    overlap_samples = (
        _samples_for_frames(overlap_frames, sample_rate, fps) if overlap_frames else 0
    )
    chunks: list[np.ndarray] = []
    pending = segment_audio[0]
    for nxt in segment_audio[1:]:
        overlap = min(overlap_samples, len(pending), len(nxt))
        if overlap <= 0:
            chunks.append(pending)
            pending = nxt
            continue
        chunks.append(pending[:-overlap])
        chunks.append(_crossfade(pending[-overlap:], nxt[:overlap]))
        pending = nxt[overlap:]
    chunks.append(pending)
    stitched_audio = np.ascontiguousarray(
        np.concatenate([c for c in chunks if len(c)], axis=0), dtype=np.float32
    )
    return stitched_video, stitched_audio


def trim(
    frames: Any,
    audio: Any | None,
    F: int,
    sample_rate: int,
    fps: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Trim/pad ``frames`` and ``audio`` to exactly ``F`` frames."""
    if isinstance(F, bool) or not isinstance(F, int):
        raise ValueError("F must be an integer")
    if F < 0:
        raise ValueError("F must be non-negative")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")

    frames = to_uint8_frames(frames)
    if len(frames) >= F:
        frames = frames[:F]
    else:
        pad_f = F - len(frames)
        frames = np.pad(
            frames, ((0, pad_f), (0, 0), (0, 0), (0, 0)), mode="constant"
        )
    frames = np.ascontiguousarray(frames, dtype=np.uint8)

    if audio is None:
        return frames, None
    stereo = to_stereo(audio)
    target_samples = _samples_for_frames(F, sample_rate, fps)
    return frames, _fit_audio_length(stereo, target_samples)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def _pad_to_even(frames: np.ndarray) -> np.ndarray:
    """yuv420p requires even width/height; replicate the last row/column."""
    height, width = frames.shape[1], frames.shape[2]
    pad_h = height % 2
    pad_w = width % 2
    if not pad_h and not pad_w:
        return frames
    return np.pad(
        frames,
        ((0, 0), (0, pad_h), (0, pad_w), (0, 0)),
        mode="edge",
    )


def _drain_stream(stream, sink: list[bytes]) -> None:
    try:
        for line in iter(stream.readline, b""):
            sink.append(line)
    except Exception:  # pragma: no cover - best effort log capture
        pass
    finally:
        with suppress(Exception):
            stream.close()


def encode_mp4(
    frames: Any,
    audio_or_None: Any | None,
    sample_rate: int,
    fps: int,
    out_path: str | Path,
    preset: str = "veryfast",
    crf: int = 17,
    audio_bitrate: str = "192k",
    timeout: float | None = 600.0,
) -> Path:
    """Encode ``frames`` (and optional audio) into ``out_path`` atomically."""
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    frames = to_uint8_frames(frames)
    if frames.shape[0] == 0:
        raise ValueError("cannot encode an empty frame sequence")
    num_frames, height, width, _ = frames.shape
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid frame size {width}x{height}")
    frames = _pad_to_even(frames)
    height, width = frames.shape[1], frames.shape[2]

    out_path = Path(out_path)
    if out_path.is_dir():
        raise ValueError(f"output path is a directory: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="va-encode-", dir=str(out_path.parent)) as tmpdir:
        tmp_dir = Path(tmpdir)
        tmp_out = tmp_dir / "out.mp4"
        cmd = [
            *_ffmpeg_base(),
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
        ]
        if audio_or_None is not None:
            stereo = to_stereo(audio_or_None)
            target_samples = _samples_for_frames(num_frames, sample_rate, fps)
            stereo = _fit_audio_length(stereo, target_samples)
            stereo = np.clip(stereo, -1.0, 1.0)  # clipping is required by PCM
            audio_file = tmp_dir / "audio.f32"
            stereo.astype("<f4", copy=False).tofile(str(audio_file))
            cmd.extend(
                [
                    "-f",
                    "f32le",
                    "-ar",
                    str(sample_rate),
                    "-ac",
                    "2",
                    "-i",
                    str(audio_file),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:a",
                    "aac",
                    "-b:a",
                    str(audio_bitrate),
                ]
            )
        else:
            cmd.extend(["-map", "0:v:0", "-an"])

        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                str(preset),
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(tmp_out),
            ]
        )

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stderr_chunks: list[bytes] = []
        # Drain stderr concurrently: otherwise FFmpeg blocks once its pipe fills.
        drainer = threading.Thread(
            target=_drain_stream,
            args=(proc.stderr, stderr_chunks),
            daemon=True,
            name="va-ffmpeg-stderr",
        )
        drainer.start()
        broken_pipe = False
        try:
            assert proc.stdin is not None
            buffer = bytearray()
            for frame in frames:
                buffer += np.ascontiguousarray(frame).tobytes()
                if len(buffer) >= _CHUNK_BYTES:
                    proc.stdin.write(buffer)
                    buffer.clear()
            if buffer:
                proc.stdin.write(buffer)
            proc.stdin.flush()
            proc.stdin.close()
            returncode = proc.wait(timeout=timeout)
        except BrokenPipeError:
            broken_pipe = True
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.wait()
            raise RuntimeError(f"FFmpeg encode timed out after {timeout}s") from exc
        finally:
            for stream in (proc.stdin, proc.stdout):
                if stream is not None:
                    with suppress(Exception):
                        stream.close()
            drainer.join(timeout=10.0)

        stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace").strip()
        if broken_pipe:
            raise RuntimeError(f"FFmpeg rejected the input stream: {stderr_text}")
        if returncode != 0:
            raise RuntimeError(f"FFmpeg encode_mp4 failed (exit {returncode}): {stderr_text}")
        if not tmp_out.is_file():
            raise RuntimeError("FFmpeg reported success but produced no output file")
        os.replace(tmp_out, out_path)
    return out_path
