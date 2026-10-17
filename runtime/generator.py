import asyncio
import secrets
from concurrent.futures import ThreadPoolExecutor
from functools import partial

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
from schemas import ResolvedParams
from shapes import RESOLUTIONS, closest_aspect, frames_for_duration, plan_segments


async def _finish_on_cancel(future):
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        try:
            await asyncio.shield(future)
        except Exception:
            pass
        raise


class Generator:
    def __init__(self, manager, settings):
        self.manager = manager
        self.settings = settings
        self.gpu_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="va-gpu"
        )
        self.encoding_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="va-encode"
        )
        self.gpu_lock = asyncio.Lock()

    def close(self):
        self.gpu_executor.shutdown(wait=True, cancel_futures=True)
        self.encoding_executor.shutdown(wait=True, cancel_futures=True)

    def _fetch(self, source, tasks):
        if source not in tasks:
            tasks[source] = asyncio.create_task(fetch_bytes(source))
        return tasks[source]

    async def resolve(self, req, media_tasks=None) -> ResolvedParams:
        media_tasks = {} if media_tasks is None else media_tasks
        profile = req.profile or self.settings.DEFAULT_PROFILE
        self.manager.pipeline_for(profile, req.mode)
        default_steps = {
            "quality": self.settings.QUALITY_STEPS,
            "fast": self.settings.FAST_STEPS,
            "turbo": self.settings.TURBO_STEPS,
        }
        steps = req.num_inference_steps or default_steps[profile]
        aspect = req.aspect_ratio
        if aspect == "auto":
            if req.first_frame is not None:
                source = req.first_frame.source
            else:
                source = next(
                    item.source for item in req.references if item.type == "image"
                )
            data = await self._fetch(source, media_tasks)
            width, height = await asyncio.to_thread(image_size, data)
            aspect = closest_aspect(width, height)
        width, height = RESOLUTIONS[req.resolution][aspect]
        final_frames = (
            req.num_frames
            if req.num_frames is not None
            else frames_for_duration(req.duration_seconds)
        )
        segments, final_frames = plan_segments(final_frames)
        seed = req.seed if req.seed is not None else secrets.randbits(63)
        return ResolvedParams(
            width=width,
            height=height,
            num_frames=final_frames,
            fps=self.settings.FPS,
            duration_seconds=final_frames / self.settings.FPS,
            seed=seed,
            profile=profile,
            steps=steps,
            mode=req.mode,
            segments=segments,
        )

    async def prepare_media(self, req, params, media_tasks=None):
        media_tasks = {} if media_tasks is None else media_tasks
        key = self.manager.pipeline_for(params.profile, params.mode)
        sample_rate = self.manager.pipelines[key].sample_rate()

        async def image(media):
            if media is None:
                return None
            data = await self._fetch(media.source, media_tasks)
            return await asyncio.to_thread(
                load_image, data, params.width, params.height
            )

        async def reference(item):
            data = await self._fetch(item.source, media_tasks)
            if item.type == "image":
                value = await asyncio.to_thread(
                    load_image, data, params.width, params.height
                )
            elif item.type == "video":
                value = await asyncio.to_thread(
                    load_video_frames,
                    data, params.width, params.height,
                    item.start_seconds, item.end_seconds,
                )
            else:
                value = await asyncio.to_thread(
                    load_audio,
                    data, sample_rate,
                    item.start_seconds, item.end_seconds,
                )
            return {
                "type": item.type,
                "data": value,
                "strength": item.strength,
                "start_seconds": item.start_seconds,
                "end_seconds": item.end_seconds,
            }

        values = await asyncio.gather(
            image(req.first_frame),
            image(req.last_frame),
            *(reference(item) for item in req.references),
        )
        return {
            "first_frame": values[0],
            "last_frame": values[1],
            "references": values[2:],
            "sample_rate": sample_rate,
        }

    def _generate_segments(self, req, params, media, loras):
        key = self.manager.pipeline_for(params.profile, params.mode)
        adapter = self.manager.ensure_resident(key)
        transformer = self.manager.transformers[key]
        cache = self.manager.caches[key]
        tau = {
            "quality": self.settings.QUALITY_REPAIR_TAU,
            "fast": self.settings.FAST_REPAIR_TAU,
            "turbo": self.settings.TURBO_REPAIR_TAU,
        }[params.profile]
        threshold = (
            self.settings.FAST_CACHE_THRESHOLD
            if params.profile == "fast"
            else 0.0
        )
        attention_router.configure(
            self.settings.ATTENTION_BACKEND == "fp8",
            tau,
            self.settings.ATTN_FP8_MIN_SEQ,
        )
        videos, audios = [], []
        previous_frame = None
        try:
            self.manager.loras.activate(transformer, loras)
            with torch.inference_mode(), denoiser_scope(transformer):
                for index, length in enumerate(params.segments):
                    first_frame = (
                        media["first_frame"] if index == 0 else previous_frame
                    )
                    last_frame = (
                        media["last_frame"]
                        if index == len(params.segments) - 1
                        else None
                    )
                    cache.begin(
                        params.steps,
                        threshold,
                        self.settings.CACHE_WARMUP_STEPS,
                        self.settings.CACHE_TAIL_STEPS,
                    )
                    try:
                        video, audio = adapter.run(
                            prompt=req.prompt,
                            height=params.height,
                            width=params.width,
                            num_frames=length,
                            steps=params.steps,
                            generator=torch.Generator(device="cuda").manual_seed(
                                (params.seed + index) % (2**63)
                            ),
                            first_frame=first_frame,
                            last_frame=last_frame,
                            references=media["references"],
                        )
                        frames = to_uint8_frames(video)
                        if req.include_audio and audio is None:
                            raise RuntimeError("The joint pipeline did not return audio")
                        waveform = to_stereo(audio) if req.include_audio else None
                        frames, waveform = trim(
                            frames,
                            waveform,
                            length,
                            media["sample_rate"],
                            params.fps,
                        )
                        videos.append(frames)
                        audios.append(waveform)
                        previous_frame = Image.fromarray(frames[-1])
                        del video, audio
                    finally:
                        cache.end()
            return videos, audios
        finally:
            cache.end()
            self.manager.loras.deactivate(transformer)

    async def run(self, req):
        media_tasks = {}
        profile = req.profile or self.settings.DEFAULT_PROFILE
        key = self.manager.pipeline_for(profile, req.mode)
        lora_task = asyncio.create_task(
            asyncio.to_thread(
                self.manager.loras.prepare,
                self.manager.transformers[key],
                req.loras,
            )
        )
        try:
            params = await self.resolve(req, media_tasks)
            media, loras = await asyncio.gather(
                self.prepare_media(req, params, media_tasks),
                lora_task,
            )
        except BaseException:
            lora_task.cancel()
            for task in media_tasks.values():
                task.cancel()
            await asyncio.gather(
                lora_task, *media_tasks.values(), return_exceptions=True
            )
            raise

        loop = asyncio.get_running_loop()
        async with self.gpu_lock:
            future = loop.run_in_executor(
                self.gpu_executor,
                partial(self._generate_segments, req, params, media, loras),
            )
            videos, audios = await _finish_on_cancel(future)

        def finish():
            frames, audio = stitch_segments(
                videos, audios, media["sample_rate"], params.fps
            )
            return trim(
                frames, audio, params.num_frames, media["sample_rate"], params.fps
            )

        frames, audio = await _finish_on_cancel(
            loop.run_in_executor(self.encoding_executor, finish)
        )
        return frames, audio, media["sample_rate"], params

    async def render_mp4(self, req, out_path):
        frames, audio, sample_rate, params = await self.run(req)
        future = asyncio.get_running_loop().run_in_executor(
            self.encoding_executor,
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
        await _finish_on_cancel(future)
        return params
