import asyncio
import secrets
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import torch
from PIL import Image

from kernels import attention_router
from kernels.attention_router import denoiser_scope
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
from shapes import RESOLUTIONS, closest_aspect, frames_for_duration, plan_segments


class Generator:
    def __init__(self, manager, settings) -> None:
        self.manager = manager
        self.settings = settings
        self.gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="va-gpu")
        self.encode_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="va-encode")
        self.gpu_lock = asyncio.Lock()

    async def resolve(
        self,
        req: GenerationRequest,
        prefetched: dict[str, asyncio.Task] | None = None,
    ) -> ResolvedParams:
        profile = req.profile or self.settings.DEFAULT_PROFILE
        self.manager.pipeline_for(profile, req.mode)

        step_defaults = {
            "quality": self.settings.QUALITY_STEPS,
            "fast": self.settings.FAST_STEPS,
            "turbo": self.settings.TURBO_STEPS,
        }
        steps = (
            req.num_inference_steps
            if req.num_inference_steps is not None
            else step_defaults[profile]
        )

        aspect = req.aspect_ratio
        if aspect == "auto":
            src = (
                req.first_frame.source
                if req.first_frame is not None
                else next(r.source for r in req.references if r.type == "image")
            )
            if prefetched is not None and src in prefetched:
                raw_bytes = await prefetched[src]
            elif prefetched is not None:
                prefetched[src] = asyncio.create_task(fetch_bytes(src))
                raw_bytes = await prefetched[src]
            else:
                raw_bytes = await fetch_bytes(src)
            iw, ih = await asyncio.to_thread(image_size, raw_bytes)
            aspect = closest_aspect(iw, ih)

        width, height = RESOLUTIONS[req.resolution][aspect]
        total_frames = (
            req.num_frames
            if req.num_frames is not None
            else frames_for_duration(req.duration_seconds)
        )
        segments, final_f = plan_segments(total_frames)
        seed = req.seed if req.seed is not None else secrets.randbits(63)

        return ResolvedParams(
            width=width,
            height=height,
            num_frames=final_f,
            fps=self.settings.FPS,
            duration_seconds=final_f / self.settings.FPS,
            seed=seed,
            profile=profile,
            steps=steps,
            mode=req.mode,
            segments=segments,
        )

    async def prepare_media(
        self,
        req: GenerationRequest,
        params: ResolvedParams,
        prefetched: dict[str, asyncio.Task] | None = None,
    ) -> dict:
        pipe_key = self.manager.pipeline_for(params.profile, params.mode)
        sr = self.manager.pipelines[pipe_key].sample_rate()
        tasks_map = prefetched if prefetched is not None else {}

        def _get_bytes_task(source: str) -> asyncio.Task:
            if source not in tasks_map:
                tasks_map[source] = asyncio.create_task(fetch_bytes(source))
            return tasks_map[source]

        async def _decode_frame(media_in):
            if media_in is None:
                return None
            data = await _get_bytes_task(media_in.source)
            return await asyncio.to_thread(load_image, data, params.width, params.height)

        async def _decode_ref(ref_in):
            data = await _get_bytes_task(ref_in.source)
            if ref_in.type == "image":
                decoded = await asyncio.to_thread(
                    load_image, data, params.width, params.height
                )
            elif ref_in.type == "video":
                decoded = await asyncio.to_thread(
                    load_video_frames,
                    data,
                    params.width,
                    params.height,
                    ref_in.start_seconds,
                    ref_in.end_seconds,
                )
            else:
                decoded = await asyncio.to_thread(
                    load_audio,
                    data,
                    sr,
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

        results = await asyncio.gather(
            _decode_frame(req.first_frame),
            _decode_frame(req.last_frame),
            *(_decode_ref(r) for r in req.references),
        )
        return {
            "first_frame": results[0],
            "last_frame": results[1],
            "references": list(results[2:]),
            "sample_rate": sr,
        }

    def _execute_gpu_segments(
        self,
        req: GenerationRequest,
        params: ResolvedParams,
        media: dict,
        prepared_loras: list,
    ) -> tuple[list, list]:
        pipe_key = self.manager.pipeline_for(params.profile, params.mode)
        adapter = self.manager.ensure_resident(pipe_key)
        transformer = self.manager.transformers[pipe_key]
        cache = self.manager.caches[pipe_key]

        tau_map = {
            "quality": self.settings.QUALITY_REPAIR_TAU,
            "fast": self.settings.FAST_REPAIR_TAU,
            "turbo": self.settings.TURBO_REPAIR_TAU,
        }
        repair_tau = tau_map[params.profile]
        cache_threshold = (
            self.settings.FAST_CACHE_THRESHOLD if params.profile == "fast" else 0.0
        )

        attention_router.configure(
            enabled=self.settings.ATTENTION_BACKEND == "fp8",
            repair_tau=repair_tau,
            min_seq=self.settings.ATTN_FP8_MIN_SEQ,
        )

        video_segments = []
        audio_segments = []
        prev_last_pil = None

        try:
            self.manager.lora_manager.activate(transformer, prepared_loras)
            with torch.inference_mode(), denoiser_scope(transformer):
                num_segs = len(params.segments)
                for seg_idx, seg_len in enumerate(params.segments):
                    seg_first = media["first_frame"] if seg_idx == 0 else prev_last_pil
                    seg_last = (
                        media["last_frame"] if seg_idx == num_segs - 1 else None
                    )
                    cache.begin(
                        num_steps=params.steps,
                        threshold=cache_threshold,
                        warmup=self.settings.CACHE_WARMUP_STEPS,
                        tail=self.settings.CACHE_TAIL_STEPS,
                    )
                    try:
                        gen = torch.Generator(device="cuda").manual_seed(
                            (params.seed + seg_idx) % (2**63)
                        )
                        v_out, a_out = adapter.run(
                            prompt=req.prompt,
                            height=params.height,
                            width=params.width,
                            num_frames=seg_len,
                            steps=params.steps,
                            generator=gen,
                            first_frame=seg_first,
                            last_frame=seg_last,
                            references=media["references"],
                        )
                        v_u8 = to_uint8_frames(v_out)
                        a_stereo = (
                            to_stereo(a_out)
                            if (req.include_audio and a_out is not None)
                            else None
                        )
                        video_segments.append(v_u8)
                        audio_segments.append(a_stereo)
                        prev_last_pil = Image.fromarray(v_u8[-1])
                        del v_out, a_out
                    finally:
                        cache.end()
            return video_segments, audio_segments
        finally:
            cache.end()
            self.manager.lora_manager.deactivate(transformer)

    async def run(self, req: GenerationRequest):
        prefetched: dict[str, asyncio.Task] = {}
        params = await self.resolve(req, prefetched)
        pipe_key = self.manager.pipeline_for(params.profile, params.mode)
        transformer = self.manager.transformers[pipe_key]

        media, prepared_loras = await asyncio.gather(
            self.prepare_media(req, params, prefetched),
            asyncio.to_thread(self.manager.lora_manager.prepare, transformer, req.loras),
        )

        loop = asyncio.get_running_loop()
        async with self.gpu_lock:
            video_segments, audio_segments = await loop.run_in_executor(
                self.gpu_executor,
                partial(
                    self._execute_gpu_segments,
                    req,
                    params,
                    media,
                    prepared_loras,
                ),
            )

        sample_rate = media["sample_rate"]
        stitched_frames, stitched_audio = stitch_segments(
            video_segments,
            audio_segments if req.include_audio else None,
            sample_rate,
            params.fps,
        )
        final_frames, final_audio = trim(
            stitched_frames,
            stitched_audio,
            params.num_frames,
            sample_rate,
            params.fps,
        )
        if final_audio is not None:
            final_audio = to_stereo(final_audio)
        return final_frames, final_audio, sample_rate, params

    async def render_mp4(self, req: GenerationRequest, out_path: str | Path) -> ResolvedParams:
        frames, audio, sample_rate, params = await self.run(req)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self.encode_executor,
            partial(
                encode_mp4,
                frames,
                audio,
                sample_rate,
                params.fps,
                out_path,
                self.settings.X264_PRESET,
                self.settings.X264_CRF,
                self.settings.AUDIO_BITRATE,
            ),
        )
        return params
