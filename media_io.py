import base64
import io
import math
import os
import subprocess
import tempfile
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageOps

from shapes import FPS


async def fetch_bytes(source: str) -> bytes:
    if source.startswith("data:"):
        header, encoded = source.split(",", 1)
        if ";base64" in header:
            return base64.b64decode(encoded)
        return encoded.encode("utf-8")
    if source.startswith(("http://", "https://")):
        async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
            response = await client.get(source)
            response.raise_for_status()
            return response.content
    path = Path(source)
    if path.is_file():
        return path.read_bytes()
    raise ValueError(f"Unsupported media source: {source}")


def load_image(data: bytes, width: int, height: int) -> Image.Image:
    with Image.open(io.BytesIO(data)) as img:
        transposed = ImageOps.exif_transpose(img).convert("RGB")
        src_w, src_h = transposed.size
        scale = max(width / src_w, height / src_h)
        new_w = max(width, round(src_w * scale))
        new_h = max(height, round(src_h * scale))
        resized = transposed.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (new_w - width) // 2
        top = (new_h - height) // 2
        return resized.crop((left, top, left + width, top + height))


def image_size(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as img:
        transposed = ImageOps.exif_transpose(img)
        return transposed.size


def _time_args(start: float | None, end: float | None) -> list[str]:
    args: list[str] = []
    if start is not None and start > 0:
        args.extend(["-ss", str(start)])
    if end is not None:
        duration = end - (start or 0.0)
        if duration <= 0:
            raise ValueError("end_seconds must be greater than start_seconds")
        args.extend(["-t", str(duration)])
    return args


def _run_ffmpeg_decode(data: bytes, pre_args: list[str], post_args: list[str]) -> bytes:
    pipe_cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        *pre_args,
        "-i",
        "pipe:0",
        *post_args,
        "pipe:1",
    ]
    proc = subprocess.run(
        pipe_cmd,
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode == 0 and len(proc.stdout) > 0:
        return proc.stdout

    with tempfile.TemporaryDirectory(prefix="va-media-") as tmpdir:
        in_path = Path(tmpdir) / "input.bin"
        in_path.write_bytes(data)
        file_cmd = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            *pre_args,
            "-i",
            str(in_path),
            *post_args,
            "pipe:1",
        ]
        proc2 = subprocess.run(
            file_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc2.returncode != 0:
            err = proc2.stderr.decode("utf-8", errors="replace")
            raise ValueError(f"FFmpeg decode failed: {err}")
        return proc2.stdout


def load_video_frames(
    data: bytes,
    width: int,
    height: int,
    start: float | None = None,
    end: float | None = None,
) -> np.ndarray:
    vf = (
        f"fps={FPS},"
        f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height}"
    )
    raw = _run_ffmpeg_decode(
        data,
        _time_args(start, end),
        ["-an", "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo"],
    )
    frame_size = width * height * 3
    num_frames = len(raw) // frame_size
    if num_frames == 0:
        raise ValueError("Decoded video contains no frames")
    trimmed = raw[: num_frames * frame_size]
    return np.frombuffer(trimmed, dtype=np.uint8).reshape(num_frames, height, width, 3).copy()


def load_audio(
    data: bytes,
    sample_rate: int,
    start: float | None = None,
    end: float | None = None,
) -> np.ndarray:
    raw = _run_ffmpeg_decode(
        data,
        _time_args(start, end),
        ["-vn", "-ac", "2", "-ar", str(sample_rate), "-acodec", "pcm_f32le", "-f", "f32le"],
    )
    num_samples = len(raw) // 8
    if num_samples == 0:
        raise ValueError("Decoded audio contains no samples")
    trimmed = raw[: num_samples * 8]
    return np.frombuffer(trimmed, dtype=np.float32).reshape(num_samples, 2).copy()


def _normalize_4d_layout(x):
    if x.ndim == 5:
        x = x[0]
    if x.ndim != 4:
        raise ValueError(f"Expected 4D or 5D video input, got {x.ndim}D")
    if x.shape[-1] in (1, 3, 4):
        pass
    elif x.shape[1] in (1, 3, 4):
        x = x.permute(0, 2, 3, 1) if hasattr(x, "permute") else np.transpose(x, (0, 2, 3, 1))
    elif x.shape[0] in (1, 3, 4):
        x = x.permute(1, 2, 3, 0) if hasattr(x, "permute") else np.transpose(x, (1, 2, 3, 0))
    else:
        raise ValueError(f"Unrecognized video shape: {tuple(x.shape)}")
    return x


def to_uint8_frames(video) -> np.ndarray:
    import torch

    if isinstance(video, (list, tuple)):
        if len(video) == 0:
            raise ValueError("Empty video frame list")
        if isinstance(video[0], Image.Image):
            return np.stack([np.asarray(img.convert("RGB"), dtype=np.uint8) for img in video], axis=0)
        if len(video) == 1 and isinstance(video[0], (torch.Tensor, np.ndarray)):
            video = video[0]
        elif isinstance(video[0], torch.Tensor):
            video = torch.stack(list(video), dim=0)
        else:
            video = np.asarray(video)

    if isinstance(video, torch.Tensor):
        t = _normalize_4d_layout(video.detach())
        if t.shape[-1] == 1:
            t = t.expand(*t.shape[:-1], 3)
        elif t.shape[-1] > 3:
            t = t[..., :3]
        if t.dtype != torch.uint8:
            tf = t.float()
            vmin = tf.amin()
            vmax = tf.amax()
            tf = torch.where(
                vmin < 0.0,
                (tf + 1.0) * 127.5,
                torch.where(vmax <= 1.0, tf * 255.0, tf),
            )
            t = torch.nan_to_num(tf, nan=0.0, posinf=255.0, neginf=0.0).round().clamp(0.0, 255.0).to(torch.uint8)
        t = t.contiguous()
        if t.is_cuda:
            cpu_pinned = torch.empty(t.shape, dtype=torch.uint8, device="cpu", pin_memory=True)
            cpu_pinned.copy_(t, non_blocking=True)
            torch.cuda.current_stream(t.device).synchronize()
            return cpu_pinned.numpy()
        return t.cpu().numpy()

    arr = _normalize_4d_layout(np.asarray(video))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float32)
        vmin = float(np.nanmin(arr_f))
        vmax = float(np.nanmax(arr_f))
        if vmin < 0.0:
            arr_f = (arr_f + 1.0) * 127.5
        elif vmax <= 1.0:
            arr_f = arr_f * 255.0
        arr_f = np.nan_to_num(arr_f, nan=0.0, posinf=255.0, neginf=0.0)
        arr = np.clip(np.rint(arr_f), 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(arr)


def to_stereo(audio) -> np.ndarray:
    import torch

    if audio is None:
        return np.zeros((0, 2), dtype=np.float32)
    if isinstance(audio, (list, tuple)) and len(audio) == 1:
        audio = audio[0]
    if isinstance(audio, torch.Tensor):
        arr = audio.detach().float().cpu().numpy()
    else:
        arr = np.asarray(audio, dtype=np.float32)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D audio, got {arr.ndim}D")
    if arr.shape[0] <= 8 and arr.shape[1] > 8:
        arr = arr.T
    if arr.shape[1] == 1:
        arr = np.repeat(arr, 2, axis=1)
    elif arr.shape[1] > 2:
        arr = arr[:, :2]
    return np.ascontiguousarray(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0), dtype=np.float32)


def stitch_segments(
    video_list: list[np.ndarray],
    audio_list: list[np.ndarray | None] | None,
    sample_rate: int,
    fps: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if not video_list:
        raise ValueError("video_list cannot be empty")
    stitched_video = np.concatenate(
        [video_list[0]] + [seg[1:] for seg in video_list[1:]],
        axis=0,
    )
    if not audio_list or all(a is None for a in audio_list):
        return stitched_video, None

    overlap = sample_rate // fps
    stereo_segments = [to_stereo(a) for a in audio_list]
    current = stereo_segments[0]
    for nxt in stereo_segments[1:]:
        ov = min(overlap, len(current), len(nxt))
        if ov <= 0:
            current = np.concatenate([current, nxt], axis=0)
            continue
        theta = np.linspace(0.0, math.pi / 2.0, ov, dtype=np.float32)[:, None]
        fade_out = np.cos(theta)
        fade_in = np.sin(theta)
        blended = current[-ov:] * fade_out + nxt[:ov] * fade_in
        current = np.concatenate([current[:-ov], blended, nxt[ov:]], axis=0)
    return stitched_video, np.ascontiguousarray(current, dtype=np.float32)


def trim(
    frames: np.ndarray,
    audio: np.ndarray | None,
    F: int,
    sample_rate: int,
    fps: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    frames = to_uint8_frames(frames)
    if len(frames) >= F:
        frames = frames[:F]
    else:
        pad_f = F - len(frames)
        frames = np.pad(frames, ((0, pad_f), (0, 0), (0, 0), (0, 0)), mode="constant")

    if audio is None:
        return frames, None

    stereo = to_stereo(audio)
    target_samples = int(F * sample_rate / fps)
    if len(stereo) >= target_samples:
        stereo = stereo[:target_samples]
    else:
        pad_s = target_samples - len(stereo)
        stereo = np.pad(stereo, ((0, pad_s), (0, 0)), mode="constant")
    return frames, np.ascontiguousarray(stereo, dtype=np.float32)


def encode_mp4(
    frames: np.ndarray,
    audio_or_None: np.ndarray | None,
    sample_rate: int,
    fps: int,
    out_path: str | Path,
    preset: str = "veryfast",
    crf: int = 17,
    audio_bitrate: str = "192k",
) -> Path:
    frames = to_uint8_frames(frames)
    num_frames, height, width, _ = frames.shape
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="va-encode-", dir=str(out_path.parent)) as tmpdir:
        tmp_dir = Path(tmpdir)
        tmp_out = tmp_dir / "out.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
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
            target_samples = int(num_frames * sample_rate / fps)
            if len(stereo) >= target_samples:
                stereo = stereo[:target_samples]
            else:
                stereo = np.pad(stereo, ((0, target_samples - len(stereo)), (0, 0)), mode="constant")
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
        assert proc.stdin is not None
        try:
            for frame in frames:
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            proc.stdin.close()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        _, stderr_bytes = proc.communicate()
        if proc.returncode != 0:
            err_msg = stderr_bytes.decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg encode_mp4 failed: {err_msg}")
        os.replace(tmp_out, out_path)
    return out_path
