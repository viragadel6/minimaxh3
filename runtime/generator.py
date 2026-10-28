"""Request orchestration: resolution, media preparation, GPU segments and encoding.

Concurrency model
-----------------
* Exactly one worker thread (``gpu_executor``) touches the GPU, so no asyncio lock is
  needed to serialise GPU work and a cancelled request cannot release a lock while its
  worker thread is still running.
* Model mutation (LoRA preparation/activation, residency changes) is additionally
  guarded by a plain :class:`threading.Lock`, which is loop agnostic.
* CPU heavy work (stitching, trimming, encoding) runs on ``encode_executor`` and is
  bounded by a semaphore so pending jobs cannot hoard uncompressed frames.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

from config import profile_repair_tau, profile_steps
from media_io import (
    encode_mp4,
    fetch_bytes,
    image_size,
    load_audio,
    load_image,
    load_video_frames,
    stitch_segments,
    to_stereo,
    to_uint8_frames,
    trim,
)
from schemas import GenerationRequest, ResolvedParams
from shapes import RESOLUTIONS, closest_aspect, plan_segments
from va_kernels import attention_router
from va_kernels.attention_router import denoiser_scope

logger = logging.getLogger(__name__)

_FRAME_OVERLAP = 1
_MAX_SEED = 2**63 - 1


@dataclass(frozen=True)
class ResolvedRequest:
    params: ResolvedParams
    pipe_key: str


def _last_frame_pil(frames):
    return Image.fromarray(frames[-1])


def _references_for_segment(
    references: Sequence[dict],
    segment_index: int,
    frame_offset: int,
    segment_frames: int,
    fps: int,
    total_segments: int,
) -> list[dict]:
    """Shift audio reference windows onto this segment's timeline."""
    if total_segments <= 1 or not references:
        return list(references)
    segment_start = frame_offset / fps
    segment_end = (frame_offset + segment_frames) / fps
    adjusted: list[dict] = []
    for reference in references:
        if reference.get("type") != "audio":
            adjusted.append(reference)
            continue
        base_start = float(reference.get("start_seconds") or 0.0)
        base_end = reference.get("end_seconds")
        new_start = base_start + segment_start
        new_end = base_start + segment_end if base_end is None else min(
            float(base_end), base_start + segment_end
        )
        if new_end <= new_start:
            continue
        updated = dict(reference)
        updated["start_seconds"] = new_start
        updated["end_seconds"] = new_end
        adjusted.append(updated)
    return adjusted


class Generator:
    def __init__(self, manager, settings) -> None:
        self.manager = manager
        self.settings = settings
        self.gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="va-gpu")
        self.encode_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="va-encode"
        )
        self.media_executor = ThreadPoolExecutor(
            max_workers=max(2, settings.MEDIA_CONCURRENCY),
            thread_name_prefix="va-media",
        )
        self._model_lock = threading.Lock()
        self._encode_semaphore: asyncio.Semaphore | None = None
        self._media_semaphore: asyncio.Semaphore | None = None
        self._closed = False

    # -- lifecycle -------------------------------------------------------
    async def shutdown(self) -> None:
        """Stop accepting work and join the worker threads.

        ``ThreadPoolExecutor.shutdown(wait=True)`` blocks, so it runs off the event loop
        (a GPU segment can take minutes to drain).
        """
        if self._closed:
            return
        self._closed = True
        for executor in (self.gpu_executor, self.encode_executor, self.media_executor):
            await asyncio.to_thread(executor.shutdown, True)

    def shutdown_nowait(self) -> None:
        """Non-blocking variant for synchronous teardown hooks (``@modal.exit``)."""
        self._closed = True
        for executor in (self.gpu_executor, self.encode_executor, self.media_executor):
            executor.shutdown(wait=False)

    async def __aenter__(self) -> "Generator":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.shutdown()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Generator has been shut down")

    def _encode_slots(self) -> asyncio.Semaphore:
        if self._encode_semaphore is None:
            self._encode_semaphore = asyncio.Semaphore(
                max(1, self.settings.ENCODE_QUEUE_LIMIT)
            )
        return self._encode_semaphore

    def _media_slots(self) -> asyncio.Semaphore:
        if self._media_semaphore is None:
            self._media_semaphore = asyncio.Semaphore(
                max(1, self.settings.MEDIA_CONCURRENCY)
            )
        return self._media_semaphore

    async def _run_encode(self, fn, *args):
        loop = asyncio.get_running_loop()
        async with self._encode_slots():
            return await loop.run_in_executor(self.encode_executor, partial(fn, *args))

    # -- resolution ------------------------------------------------------
    def _profile(self, req: GenerationRequest) -> str:
        profile = req.profile if req.profile is not None else self.settings.DEFAULT_PROFILE
        if not isinstance(profile, str) or not profile:
            raise ValueError(f"invalid profile {profile!r}")
        # Raises a descriptive ValueError for unknown profiles.
        profile_steps(self.settings, profile)
        return profile

    async def resolve(
        self,
        req: GenerationRequest,
        prefetched: dict[str, asyncio.Task] | None = None,
    ) -> ResolvedRequest:
        self._check_open()
        profile = self._profile(req)
        mode = req.effective_mode()
        pipe_key = self.manager.pipeline_for(profile, mode)
        steps = (
            req.num_inference_steps
            if req.num_inference_steps is not None
            else profile_steps(self.settings, profile)
        )

        aspect = req.aspect_ratio
        if aspect == "auto":
            src = req.aspect_source()
            if src is None:
                raise ValueError(
                    "aspect_ratio='auto' requires first_frame, last_frame or an "
                    "image/video reference"
                )
            raw_bytes = await self._fetch(src, prefetched)
            iw, ih = await asyncio.to_thread(image_size, raw_bytes)
            aspect = closest_aspect(iw, ih)

        width, height = RESOLUTIONS[req.resolution][aspect]
        total_frames = req.effective_num_frames(
            self.settings.FPS, max_frames=self.settings.max_frames()
        )
        segments, final_frames = plan_segments(total_frames)
        seed = req.seed if req.seed is not None else secrets.randbits(63)
        seed %= _MAX_SEED + 1  # normalise once so ResolvedParams.seed is authoritative

        params = ResolvedParams(
            width=width,
            height=height,
            num_frames=final_frames,
            fps=self.settings.FPS,
            duration_seconds=final_frames / self.settings.FPS,
            seed=seed,
            profile=profile,
            steps=steps,
            mode=mode,
            segments=segments,
        )
        return ResolvedRequest(params=params, pipe_key=pipe_key)

    async def _fetch(
        self, source: str, prefetched: dict[str, asyncio.Task] | None
    ) -> bytes:
        if prefetched is None:
            return await fetch_bytes(source)
        task = prefetched.get(source)
        if task is None:
            task = asyncio.create_task(fetch_bytes(source))
            prefetched[source] = task
        return await task

    # -- media -----------------------------------------------------------
    def _sample_rate(self, pipe_key: str, needed: bool) -> int:
        if not needed:
            return self.settings.FPS * 1000
        self.manager.ensure_resident(pipe_key)
        adapter = self.manager.pipelines[pipe_key]
        return int(adapter.sample_rate())

    async def prepare_media(
        self,
        req: GenerationRequest,
        params: ResolvedParams,
        pipe_key: str,
        prefetched: dict[str, asyncio.Task] | None = None,
    ) -> dict:
        # ``include_audio`` only controls whether audio is *returned*; an audio
        # reference is still conditioning input, so it has to be decoded even when the
        # caller asked for a silent result.
        needs_audio = req.include_audio or any(r.type == "audio" for r in req.references)
        loop = asyncio.get_running_loop()
        sample_rate = await loop.run_in_executor(
            self.gpu_executor, partial(self._sample_rate, pipe_key, needs_audio)
        )
        tasks_map = prefetched if prefetched is not None else {}
        decode_cache: dict[tuple, Any] = {}
        semaphore = self._media_slots()

        async def _bytes_for(source: str) -> bytes:
            task = tasks_map.get(source)
            if task is None:
                task = asyncio.create_task(fetch_bytes(source))
                tasks_map[source] = task
            return await task

        async def _decode(cache_key: tuple, factory, *args):
            if cache_key in decode_cache:
                return decode_cache[cache_key]
            async with semaphore:
                value = await asyncio.to_thread(factory, *args)
            decode_cache[cache_key] = value
            return value

        async def _decode_frame(media_in):
            if media_in is None:
                return None
            data = await _bytes_for(media_in.source)
            return await _decode(
                ("image", media_in.source, params.width, params.height),
                load_image,
                data,
                params.width,
                params.height,
            )

        async def _decode_ref(ref_in):
            if ref_in.type not in ("image", "video", "audio"):
                raise ValueError(f"unsupported reference type {ref_in.type!r}")
            data = await _bytes_for(ref_in.source)
            if ref_in.type == "image":
                decoded = await _decode(
                    ("image", ref_in.source, params.width, params.height),
                    load_image,
                    data,
                    params.width,
                    params.height,
                )
            elif ref_in.type == "video":
                decoded = await _decode(
                    (
                        "video",
                        ref_in.source,
                        params.width,
                        params.height,
                        ref_in.start_seconds,
                        ref_in.end_seconds,
                    ),
                    load_video_frames,
                    data,
                    params.width,
                    params.height,
                    ref_in.start_seconds,
                    ref_in.end_seconds,
                    self.settings.FPS,
                )
            else:
                decoded = await _decode(
                    ("audio", ref_in.source, sample_rate, ref_in.start_seconds, ref_in.end_seconds),
                    load_audio,
                    data,
                    sample_rate,
                    ref_in.start_seconds,
                    ref_in.end_seconds,
                )
            return {
                "type": ref_in.type,
                "data": decoded,
                "strength": ref_in.strength,
                "start_seconds": ref_in.start_seconds,
                "end_seconds": ref_in.end_seconds,
            }

        media_task = asyncio.ensure_future(_decode_frame(req.first_frame))
        last_task = asyncio.ensure_future(_decode_frame(req.last_frame))
        ref_tasks = [asyncio.ensure_future(_decode_ref(ref)) for ref in req.references]
        pending = [media_task, last_task, *ref_tasks]
        try:
            results = await asyncio.gather(*pending)
        except BaseException:
            # Cancel in-flight decodes/fetches so no orphaned task logs later.
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in list(tasks_map.values()):
                if not task.done():
                    task.cancel()
            raise
        return {
            "first_frame": results[0],
            "last_frame": results[1],
            "references": list(results[2:]),
            "sample_rate": sample_rate,
        }

    # -- GPU work --------------------------------------------------------
    def _prepare_loras(self, transformer, specs):
        with self._model_lock:
            return self.manager.lora_manager.prepare(transformer, specs)

    def _activate_loras(self, transformer, prepared):
        with self._model_lock:
            self.manager.lora_manager.activate(transformer, prepared)

    def _deactivate_loras(self, transformer):
        with self._model_lock:
            self.manager.lora_manager.deactivate(transformer)

    def _gen_device(self, transformer) -> torch.device:
        for param in transformer.parameters():
            if param.is_cuda:
                return param.device
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        raise RuntimeError("the transformer is not on a CUDA device")

    def _run_one_segment(
        self,
        adapter,
        transformer,
        req: GenerationRequest,
        params: ResolvedParams,
        segment_length: int,
        segment_index: int,
        first_frame,
        last_frame,
        references,
        seed: int,
    ):
        """Run a single denoising segment (GPU thread) and copy results to host."""
        device = self._gen_device(transformer)
        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.inference_mode(), denoiser_scope(transformer):
            video, audio = adapter.run(
                prompt=req.prompt,
                height=params.height,
                width=params.width,
                num_frames=segment_length,
                steps=params.steps,
                generator=generator,
                first_frame=first_frame,
                last_frame=last_frame,
                references=references,
                include_audio=req.include_audio,
            )
            frames = to_uint8_frames(video)
            del video
            stereo = to_stereo(audio) if (req.include_audio and audio is not None) else None
            del audio
            torch.cuda.empty_cache()
        return frames, stereo

    async def _run_segments(
        self,
        req: GenerationRequest,
        params: ResolvedParams,
        media: dict,
        prepared_loras,
        pipe_key: str,
    ) -> tuple[list, list]:
        loop = asyncio.get_running_loop()
        adapter = await loop.run_in_executor(
            self.gpu_executor, self.manager.ensure_resident, pipe_key
        )
        transformer = self.manager.transformers[pipe_key]
        cache = self.manager.caches[pipe_key]
        repair_tau = profile_repair_tau(self.settings, params.profile)
        cache_threshold = (
            self.settings.FAST_CACHE_THRESHOLD if params.profile == "fast" else 0.0
        )
        previous_config = attention_router.get_config()
        activated = False
        video_segments: list = []
        audio_segments: list = []
        try:
            attention_router.configure(
                enabled=self.settings.ATTENTION_BACKEND == "fp8",
                repair_tau=repair_tau,
                min_seq=self.settings.ATTN_FP8_MIN_SEQ,
            )
            await loop.run_in_executor(
                self.gpu_executor,
                partial(self._activate_loras, transformer, prepared_loras),
            )
            activated = True

            prev_last_pil = None
            frame_offset = 0
            num_segments = len(params.segments)
            for seg_idx, seg_len in enumerate(params.segments):
                seg_first = media["first_frame"] if seg_idx == 0 else prev_last_pil
                seg_last = media["last_frame"] if seg_idx == num_segments - 1 else None
                references = _references_for_segment(
                    media["references"],
                    seg_idx,
                    frame_offset,
                    seg_len,
                    params.fps,
                    num_segments,
                )
                cache.begin(
                    num_steps=params.steps,
                    threshold=cache_threshold,
                    warmup=self.settings.CACHE_WARMUP_STEPS,
                    tail=self.settings.CACHE_TAIL_STEPS,
                    min_skippable=self.settings.CACHE_MIN_SKIPPABLE_STEPS,
                )
                try:
                    frames, stereo = await loop.run_in_executor(
                        self.gpu_executor,
                        partial(
                            self._run_one_segment,
                            adapter,
                            transformer,
                            req,
                            params,
                            seg_len,
                            seg_idx,
                            seg_first,
                            seg_last,
                            references,
                            (params.seed + seg_idx) % (_MAX_SEED + 1),
                        ),
                    )
                finally:
                    cache.end()
                video_segments.append(frames)
                audio_segments.append(stereo)
                if seg_idx < num_segments - 1 and len(frames):
                    prev_last_pil = await self._run_encode(_last_frame_pil, frames)
                frame_offset += seg_len - _FRAME_OVERLAP
            return video_segments, audio_segments
        finally:
            if activated:
                try:
                    await loop.run_in_executor(
                        self.gpu_executor,
                        partial(self._deactivate_loras, transformer),
                    )
                except Exception:  # pragma: no cover - best effort cleanup
                    logger.exception("failed to deactivate LoRA adapters")
            attention_router.configure(
                enabled=previous_config.enabled,
                repair_tau=previous_config.repair_tau,
                min_seq=previous_config.min_seq,
            )

    # -- orchestration ---------------------------------------------------
    async def run(self, req: GenerationRequest):
        self._check_open()
        prefetched: dict[str, asyncio.Task] = {}
        resolved = await self.resolve(req, prefetched)
        params = resolved.params
        pipe_key = resolved.pipe_key
        loop = asyncio.get_running_loop()

        await loop.run_in_executor(
            self.gpu_executor, self.manager.ensure_resident, pipe_key
        )
        transformer = self.manager.transformers[pipe_key]

        media_task = asyncio.ensure_future(
            self.prepare_media(req, params, pipe_key, prefetched)
        )
        # LoRA preparation can move tensors onto the GPU, so it must run on the GPU
        # thread: ``asyncio.to_thread`` would put CUDA work on a second thread and
        # break the single-GPU-thread invariant the executors are built around.
        lora_task = asyncio.ensure_future(
            loop.run_in_executor(
                self.gpu_executor, partial(self._prepare_loras, transformer, req.loras)
            )
        )
        try:
            media, prepared_loras = await asyncio.gather(media_task, lora_task)
        except BaseException:
            for task in (media_task, lora_task):
                task.cancel()
            await asyncio.gather(media_task, lora_task, return_exceptions=True)
            raise
        finally:
            # Release decoded byte buffers: `media` already holds decoded arrays.
            for task in list(prefetched.values()):
                if not task.done():
                    task.cancel()
            prefetched.clear()

        video_segments, audio_segments = await self._run_segments(
            req, params, media, prepared_loras, pipe_key
        )
        sample_rate = media["sample_rate"]

        stitched_frames, stitched_audio = await self._run_encode(
            stitch_segments,
            video_segments,
            audio_segments if req.include_audio else None,
            sample_rate,
            params.fps,
            _FRAME_OVERLAP,
        )
        final_frames, final_audio = await self._run_encode(
            trim,
            stitched_frames,
            stitched_audio,
            params.num_frames,
            sample_rate,
            params.fps,
        )
        if final_audio is not None:
            final_audio = await self._run_encode(to_stereo, final_audio)
        return final_frames, final_audio, sample_rate, params

    async def render_mp4(
        self, req: GenerationRequest, out_path: str | Path
    ) -> ResolvedParams:
        frames, audio, sample_rate, params = await self.run(req)
        if frames is None or len(frames) == 0:
            raise ValueError("generation produced no frames; refusing to encode an empty video")
        await self._run_encode(
            encode_mp4,
            frames,
            audio,
            sample_rate,
            params.fps,
            out_path,
            self.settings.X264_PRESET,
            self.settings.X264_CRF,
            self.settings.AUDIO_BITRATE,
            self.settings.FFMPEG_TIMEOUT,
        )
        return params


__all__ = ["Generator", "ResolvedRequest"]
