import asyncio
import base64
import binascii
import io
import ipaddress
import math
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageOps

from shapes import FPS


MAX_MEDIA_BYTES = 512 * 1024 * 1024
MAX_DECODE_BYTES = 2 * 1024 * 1024 * 1024
MAX_IMAGE_PIXELS = 64 * 1024 * 1024
FFMPEG_TIMEOUT = 600


async def _check_public_request(request):
    if request.url.scheme not in {"http", "https"}:
        raise ValueError("Only HTTP(S) media URLs are supported")
    if request.url.username or request.url.password:
        raise ValueError("Credentials in media URLs are not supported")
    host = request.url.host
    port = request.url.port or (443 if request.url.scheme == "https" else 80)
    addresses = await asyncio.get_running_loop().getaddrinfo(
        host, port, type=socket.SOCK_STREAM
    )
    if not addresses:
        raise ValueError("The media hostname did not resolve")
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("Media URLs must resolve to public addresses")


async def fetch_bytes(source: str) -> bytes:
    if source.startswith("data:"):
        try:
            header, payload = source.split(",", 1)
            if ";base64" not in header.lower():
                raise ValueError("Only base64 data URIs are supported")
            if len(payload) > 4 * ((MAX_MEDIA_BYTES + 2) // 3):
                raise ValueError("Media exceeds the download size limit")
            data = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"Invalid data URI: {exc}") from exc
        if not data or len(data) > MAX_MEDIA_BYTES:
            raise ValueError("Media is empty or exceeds the download size limit")
        return data

    if not source.startswith(("http://", "https://")):
        raise ValueError("Media source must be an HTTP(S) URL or a base64 data URI")

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=120,
            max_redirects=5,
            trust_env=False,
            event_hooks={"request": [_check_public_request]},
        ) as client:
            async with client.stream("GET", source) as response:
                response.raise_for_status()
                length = response.headers.get("content-length")
                if length and int(length) > MAX_MEDIA_BYTES:
                    raise ValueError("Media exceeds the download size limit")
                chunks = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > MAX_MEDIA_BYTES:
                        raise ValueError("Media exceeds the download size limit")
                    chunks.append(chunk)
        data = b"".join(chunks)
        if not data:
            raise ValueError("Downloaded media is empty")
        return data
    except httpx.HTTPError as exc:
        raise ValueError(f"Media download failed: {type(exc).__name__}") from exc


def _open_image(data):
    image = Image.open(io.BytesIO(data))
    if image.width * image.height > MAX_IMAGE_PIXELS:
        image.close()
        raise ValueError("Image exceeds the pixel limit")
    return image


def load_image(data: bytes, width: int, height: int) -> Image.Image:
    with _open_image(data) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        return ImageOps.fit(
            image,
            (width, height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )


def image_size(data: bytes) -> tuple[int, int]:
    with _open_image(data) as original:
        image = ImageOps.exif_transpose(original)
        return image.size


def _interval_args(start, end):
    start = 0.0 if start is None else float(start)
    if start < 0 or not math.isfinite(start):
        raise ValueError("Invalid media start time")
    result = ["-ss", str(start)] if start else []
    if end is not None:
        end = float(end)
        if not math.isfinite(end) or end <= start:
            raise ValueError("Invalid media end time")
        result += ["-t", str(end - start)]
    return result


def _decode(data: bytes, output_args: list[str]) -> bytes:
    if not data or len(data) > MAX_MEDIA_BYTES:
        raise ValueError("Invalid input media size")
    with tempfile.TemporaryDirectory(prefix="va-decode-") as directory:
        source = Path(directory) / "input.media"
        source.write_bytes(data)
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            command = [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                "-threads", "2",
                "-protocol_whitelist", "file,pipe",
                "-format_whitelist",
                "mov,matroska,webm,avi,mp3,wav,flac,ogg,aac,mpeg,mpegts,image2",
                "-i", str(source),
                *output_args,
                "pipe:1",
            ]
            process = subprocess.Popen(
                command, stdout=output, stderr=errors, stdin=subprocess.DEVNULL
            )
            deadline = time.monotonic() + FFMPEG_TIMEOUT
            try:
                while process.poll() is None:
                    if time.monotonic() > deadline:
                        raise ValueError("Media decoding timed out")
                    if os.fstat(output.fileno()).st_size > MAX_DECODE_BYTES:
                        raise ValueError("Decoded media exceeds the size limit")
                    time.sleep(0.02)
                if process.returncode:
                    errors.seek(0)
                    message = errors.read(8192).decode("utf-8", errors="replace")
                    raise ValueError(f"FFmpeg could not decode the media: {message}")
                size = os.fstat(output.fileno()).st_size
                if size > MAX_DECODE_BYTES:
                    raise ValueError("Decoded media exceeds the size limit")
                output.seek(0)
                return output.read()
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()


def load_video_frames(data, width, height, start=None, end=None):
    video_filter = (
        f"fps={FPS},"
        f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height},setsar=1"
    )
    raw = _decode(
        data,
        [
            *_interval_args(start, end),
            "-map", "0:v:0", "-an",
            "-vf", video_filter,
            "-pix_fmt", "rgb24", "-f", "rawvideo",
        ],
    )
    frame_bytes = width * height * 3
    if not raw or len(raw) % frame_bytes:
        raise ValueError("The input contains no complete video frames")
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)


def load_audio(data, sample_rate, start=None, end=None):
    raw = _decode(
        data,
        [
            *_interval_args(start, end),
            "-map", "0:a:0", "-vn",
            "-ac", "2", "-ar", str(sample_rate),
            "-acodec", "pcm_f32le", "-f", "f32le",
        ],
    )
    if not raw or len(raw) % 8:
        raise ValueError("The input contains no complete stereo audio samples")
    return np.frombuffer(raw, dtype="<f4").reshape(-1, 2).copy()


def _video_layout(value):
    if value.ndim == 5:
        if value.shape[0] != 1:
            raise ValueError("Only a single generated video is supported")
        value = value[0]
    if value.ndim != 4:
        raise ValueError("Video must have four or five dimensions")
    if value.shape[-1] in (1, 3, 4):
        order = None
    elif value.shape[1] in (1, 3, 4):
        order = (0, 2, 3, 1)
    elif value.shape[0] in (1, 3, 4):
        order = (1, 2, 3, 0)
    else:
        raise ValueError("Cannot identify the video channel dimension")
    if order is not None:
        if hasattr(value, "permute"):
            value = value.permute(*order)
        else:
            value = value.transpose(order)
    return value


def to_uint8_frames(video):
    import torch

    if isinstance(video, (list, tuple)):
        if not video:
            raise ValueError("Generated video is empty")
        if all(isinstance(item, Image.Image) for item in video):
            return np.stack([np.asarray(item.convert("RGB")) for item in video])
        if len(video) == 1:
            video = video[0]
        elif all(isinstance(item, torch.Tensor) for item in video):
            video = torch.stack(video)
        else:
            video = np.asarray(video)

    if isinstance(video, torch.Tensor):
        value = _video_layout(video.detach())
        if value.shape[-1] == 1:
            value = value.expand(*value.shape[:-1], 3)
        else:
            value = value[..., :3]
        if value.dtype != torch.uint8:
            value = value.float()
            low, high = value.amin(), value.amax()
            value = torch.where(
                low < 0,
                (value + 1.0) * 127.5,
                torch.where(high <= 1.0, value * 255.0, value),
            )
            value = torch.nan_to_num(
                value, nan=0.0, posinf=255.0, neginf=0.0
            ).round().clamp_(0, 255).to(torch.uint8)
        value = value.contiguous()
        if value.is_cuda:
            host = torch.empty(
                value.shape, dtype=torch.uint8, device="cpu", pin_memory=True
            )
            host.copy_(value, non_blocking=True)
            torch.cuda.current_stream(value.device).synchronize()
            return host.numpy()
        return value.cpu().numpy()

    value = _video_layout(np.asarray(video))
    if value.shape[-1] == 1:
        value = np.repeat(value, 3, axis=-1)
    else:
        value = value[..., :3]
    if value.dtype != np.uint8:
        value = value.astype(np.float32)
        low = float(np.nanmin(value))
        high = float(np.nanmax(value))
        if low < 0:
            value = (value + 1.0) * 127.5
        elif high <= 1.0:
            value = value * 255.0
        value = np.nan_to_num(value, nan=0.0, posinf=255.0, neginf=0.0)
        value = np.rint(value).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(value)


def to_stereo(audio):
    import torch

    if audio is None:
        return np.zeros((0, 2), dtype=np.float32)
    if isinstance(audio, (list, tuple)) and len(audio) == 1:
        if isinstance(audio[0], (torch.Tensor, np.ndarray)):
            audio = audio[0]
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().float().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    while audio.ndim > 2 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2:
        raise ValueError("Audio must contain a single waveform")
    if audio.shape[-1] > 8 and audio.shape[0] <= 32:
        audio = audio.T
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] >= 2:
        audio = audio[:, :2]
    else:
        raise ValueError("Audio has no channels")
    return np.ascontiguousarray(np.nan_to_num(audio), dtype=np.float32)


def _fit_audio(audio, count):
    value = to_stereo(audio)
    if len(value) >= count:
        return value[:count]
    return np.pad(value, ((0, count - len(value)), (0, 0)))


def stitch_segments(video_list, audio_list, sample_rate, fps):
    if not video_list:
        raise ValueError("No segments were generated")
    frames = np.concatenate(
        [video_list[0]] + [value[1:] for value in video_list[1:]],
        axis=0,
    )
    if not audio_list or all(value is None for value in audio_list):
        return frames, None
    if len(audio_list) != len(video_list):
        raise ValueError("Video and audio segment counts differ")

    values = [
        _fit_audio(audio, int(len(video) * sample_rate / fps))
        for video, audio in zip(video_list, audio_list)
    ]
    pieces = [values[0]]
    overlap = sample_rate // fps
    for value in values[1:]:
        previous = pieces.pop()
        count = min(overlap, len(previous), len(value))
        if count == 0:
            pieces.extend((previous, value))
            continue
        if count == 1:
            angles = np.array([math.pi / 4], dtype=np.float32)
        else:
            angles = np.linspace(0, math.pi / 2, count, dtype=np.float32)
        crossfade = (
            previous[-count:] * np.cos(angles)[:, None]
            + value[:count] * np.sin(angles)[:, None]
        )
        pieces.extend((previous[:-count], crossfade, value[count:]))
    return frames, np.concatenate(pieces, axis=0).astype(np.float32, copy=False)


def trim(frames, audio, F, sample_rate, fps):
    if len(frames) < F:
        raise ValueError("The model returned fewer frames than requested")
    frames = frames[:F]
    if audio is not None:
        audio = _fit_audio(audio, int(F * sample_rate / fps))
    return frames, audio


def encode_mp4(
    frames,
    audio_or_None,
    sample_rate,
    fps,
    out_path,
    preset,
    crf,
    audio_bitrate,
):
    frames = to_uint8_frames(frames)
    if len(frames) == 0:
        raise ValueError("Cannot encode an empty video")
    height, width = frames.shape[1:3]
    if width % 2 or height % 2:
        raise ValueError("YUV420 video dimensions must be even")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".va-encode-", dir=out_path.parent
    ) as directory:
        directory = Path(directory)
        temporary_video = directory / "video.mp4"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s:v", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:0",
        ]
        if audio_or_None is not None:
            audio_path = directory / "audio.f32"
            audio = _fit_audio(
                audio_or_None, int(len(frames) * sample_rate / fps)
            )
            audio.astype("<f4", copy=False).tofile(audio_path)
            command += [
                "-f", "f32le", "-ar", str(sample_rate), "-ac", "2",
                "-i", str(audio_path),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:a", "aac", "-b:a", str(audio_bitrate),
            ]
        else:
            command += ["-map", "0:v:0", "-an"]
        command += [
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-threads", "2",
            "-frames:v", str(len(frames)),
            "-t", f"{len(frames) / fps:.9f}",
            "-movflags", "+faststart", str(temporary_video),
        ]

        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=errors,
            )
            failure = None
            try:
                for frame in frames:
                    process.stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))
                process.stdin.close()
                process.wait(timeout=FFMPEG_TIMEOUT)
            except BaseException as exc:
                failure = exc
                if process.poll() is None:
                    process.kill()
                process.wait()
            finally:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
            if failure is not None or process.returncode:
                errors.seek(0)
                message = errors.read(8192).decode("utf-8", errors="replace")
                raise RuntimeError(f"MP4 encoding failed: {message}") from failure
        os.replace(temporary_video, out_path)
