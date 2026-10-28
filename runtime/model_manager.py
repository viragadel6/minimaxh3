import copy
import gc
import logging
import importlib
import inspect
import os
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from va_kernels import attention_router
from va_kernels.attention_router import denoiser_scope, instrument_transformer
from va_kernels.fp8_linear import convert_transformer_to_fp8, find_primary_block_list
from runtime.lora import LoraManager, fuse_into_weights, parse_fused_loras
from runtime.pipeline_adapter import PipelineAdapter
from runtime.step_cache import StepResidualCache
from runtime.weights import ensure_weights
from shapes import RESOLUTIONS, normalize_num_frames, validate_num_frames

logger = logging.getLogger(__name__)


def _module_footprint_bytes(mod: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in list(mod.parameters()) + list(mod.buffers()):
        if tensor is None:
            continue
        ptr = tensor.untyped_storage().data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            total += tensor.untyped_storage().nbytes()
    return total


def _is_module_on_cuda(mod: nn.Module) -> bool:
    for tensor in list(mod.parameters()) + list(mod.buffers()):
        if tensor is not None:
            return tensor.is_cuda
    return False


def _pin_module_to_host(mod: nn.Module) -> None:
    mod.to("cpu")
    with torch.no_grad():
        for param in mod.parameters():
            param.data = param.data.cpu().pin_memory()
        for sub in mod.modules():
            for bname, buf in list(sub._buffers.items()):
                if buf is not None and not buf.is_pinned():
                    sub._buffers[bname] = buf.cpu().pin_memory()


def _get_components(pipe) -> dict:
    comps = getattr(pipe, "components", None)
    if isinstance(comps, Mapping):
        return dict(comps)
    return {}


def _parse_warmup_buckets(raw: str) -> list[tuple[int, int, int]]:
    """Parse ``WARMUP_BUCKETS`` into ``(width, height, frames)`` triples.

    Malformed entries are skipped with a warning: this is an optional tuning knob, so a
    typo must never prevent the container from starting.
    """
    bucket_specs: list[tuple[int, int, int]] = []
    for entry in (raw or "").split(","):
        item = entry.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 4:
            logger.warning("ignoring warm-up bucket %r (expected RES:W:H:FRAMES)", item)
            continue
        res_key, ar_w, ar_h, f_str = parts
        aspect = f"{ar_w}:{ar_h}"
        size = RESOLUTIONS.get(res_key, {}).get(aspect)
        if size is None:
            logger.warning("ignoring warm-up bucket %r (unknown resolution/aspect)", item)
            continue
        try:
            requested = int(f_str)
        except ValueError:
            logger.warning("ignoring warm-up bucket %r (frames is not a number)", item)
            continue
        try:
            frames = validate_num_frames(requested)
        except ValueError:
            try:
                frames = normalize_num_frames(requested)
            except ValueError:
                logger.warning("ignoring warm-up bucket %r (invalid frame count)", item)
                continue
            logger.info(
                "warm-up bucket %r uses %d frames instead of %d",
                item,
                frames,
                requested,
            )
        bucket_specs.append((int(size[0]), int(size[1]), int(frames)))
    return bucket_specs


class ModelManager:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.blocks_module = None
        self.transformers: dict[str, nn.Module] = {}
        self.footprints: dict[str, int] = {}
        self.pipelines: dict[str, PipelineAdapter] = {}
        self.caches: dict[str, StepResidualCache] = {}
        self.lora_manager = LoraManager(settings)
        self.resident_lru: OrderedDict[str, None] = OrderedDict()

    def _instantiate_pipeline(
        self,
        base_path: Path,
        subfolder: str,
        shared_components: dict | None = None,
    ):
        from diffusers import ModularPipeline

        pipe = ModularPipeline.from_pretrained(
            str(base_path),
            subfolder=subfolder,
            torch_dtype=torch.bfloat16,
        )
        if shared_components and hasattr(pipe, "update_components"):
            pipe.update_components(**shared_components)

        if hasattr(pipe, "load_components") and "transformer" not in _get_components(pipe):
            sig = inspect.signature(pipe.load_components)
            load_kwargs = {"torch_dtype": torch.bfloat16}
            if shared_components and "names" in sig.parameters:
                expected = set(getattr(pipe, "component_names", ()) or ())
                if expected:
                    remaining = sorted(expected - set(shared_components.keys()))
                    load_kwargs["names"] = remaining
            pipe.load_components(**load_kwargs)
            if shared_components and hasattr(pipe, "update_components"):
                pipe.update_components(**shared_components)

        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        return pipe

    def _prepare_transformer(self, key: str, transformer: nn.Module, fused_specs: list) -> None:
        transformer.eval().requires_grad_(False)
        transformer.to(device="cuda", dtype=torch.bfloat16)

        if fused_specs:
            fuse_into_weights(transformer, fused_specs)

        if self.settings.WEIGHT_PRECISION == "fp8":
            report: list[str] = []
            converted = convert_transformer_to_fp8(
                transformer,
                self.settings.FP8_EXCLUDE_PATTERNS,
                self.settings.FP8_KEEP_EDGE_BLOCKS,
                report=report,
            )
            if report:
                logger.info(
                    "FP8 conversion kept %d layer(s) in their original precision: %s",
                    len(report),
                    "; ".join(report[:5]),
                )
            logger.info("converted %d layer(s) of %r to FP8", converted, key)

        if hasattr(transformer, "set_attention_backend"):
            transformer.set_attention_backend("native")
        instrument_transformer(transformer)

        self.lora_manager.register(transformer)

        compile_mode = (self.settings.COMPILE_MODE or "").strip()
        if compile_mode and compile_mode.lower() not in ("none", "off", "0", "false"):
            torch._dynamo.config.cache_size_limit = 64
            try:
                _, blocks = find_primary_block_list(transformer)
            except ValueError:
                blocks = None
            if blocks is not None:
                for idx, block in enumerate(list(blocks)):
                    compiled_block = torch.compile(
                        block,
                        mode=compile_mode,
                        dynamic=False,
                        fullgraph=False,
                    )
                    blocks[idx] = compiled_block

        # Attach the step cache *after* compiling so the wrappers sit on top of the
        # compiled modules instead of being hidden behind them.
        cache = StepResidualCache()
        try:
            cache.attach(transformer)
        except ValueError as exc:
            logger.warning("step-residual cache disabled for %r: %s", key, exc)
        self.caches[key] = cache

        self.transformers[key] = transformer
        self.footprints[key] = _module_footprint_bytes(transformer)
        self.resident_lru[key] = None
        self.resident_lru.move_to_end(key)

    def _configure_shared_components(self, pipe) -> None:
        for name, comp in _get_components(pipe).items():
            if name == "transformer" or not isinstance(comp, nn.Module):
                continue
            comp.eval().requires_grad_(False)
            comp.to("cuda")
            if self.settings.VAE_TILING and "vae" in name.lower():
                if hasattr(comp, "enable_tiling"):
                    comp.enable_tiling()

    def _enforce_residency_budget(self, keep_key: str) -> None:
        reserve_bytes = int(self.settings.RESIDENCY_RESERVE_GIB) * (1024**3)
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes >= reserve_bytes:
            return
        for candidate in list(self.resident_lru.keys()):
            if candidate == keep_key:
                continue
            self._evict_transformer(candidate)
            free_bytes, _ = torch.cuda.mem_get_info()
            if free_bytes >= reserve_bytes:
                break

    def _evict_transformer(self, key: str) -> None:
        transformer = self.transformers[key]
        if key in self.caches:
            self.caches[key].end()
        self.lora_manager.deactivate(transformer)
        _pin_module_to_host(transformer)
        self.resident_lru.pop(key, None)
        gc.collect()
        torch.cuda.empty_cache()

    def ensure_resident(self, key: str) -> PipelineAdapter:
        transformer = self.transformers[key]
        if not _is_module_on_cuda(transformer):
            reserve_bytes = int(self.settings.RESIDENCY_RESERVE_GIB) * (1024**3)
            needed = self.footprints.get(key, _module_footprint_bytes(transformer)) + reserve_bytes
            free_bytes, _ = torch.cuda.mem_get_info()
            while free_bytes < needed and self.resident_lru:
                oldest = next(iter(self.resident_lru))
                if oldest == key:
                    break
                self._evict_transformer(oldest)
                free_bytes, _ = torch.cuda.mem_get_info()
            transformer.to("cuda", non_blocking=True)
            torch.cuda.current_stream().synchronize()

        self.resident_lru.pop(key, None)
        self.resident_lru[key] = None
        self._enforce_residency_budget(keep_key=key)

        adapter = self.pipelines[key]
        if hasattr(adapter.pipe, "update_components"):
            adapter.pipe.update_components(transformer=transformer)
        return adapter

    def load(self) -> None:
        models_dir = Path(self.settings.MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(models_dir / "inductor_cache")
        os.environ["TRITON_CACHE_DIR"] = str(models_dir / "triton_cache")
        torch.backends.cuda.matmul.allow_tf32 = True

        attention_router.install()
        attention_router.configure(
            enabled=self.settings.ATTENTION_BACKEND == "fp8",
            repair_tau=self.settings.FAST_REPAIR_TAU,
            min_seq=self.settings.ATTN_FP8_MIN_SEQ,
        )

        try:
            self.blocks_module = importlib.import_module(self.settings.MODEL_BLOCKS_MODULE)
        except ImportError:
            self.blocks_module = None

        paths = ensure_weights(self.settings)
        fused_specs = parse_fused_loras(self.settings.FUSED_LORAS)

        fl_pipe = self._instantiate_pipeline(
            paths["base"], self.settings.FL_SUBFOLDER
        )
        self._configure_shared_components(fl_pipe)
        fl_transformer = _get_components(fl_pipe).get(
            "transformer", getattr(fl_pipe, "transformer", None)
        )
        self._prepare_transformer("fl", fl_transformer, fused_specs)
        self.pipelines["fl"] = PipelineAdapter(fl_pipe, self.blocks_module)

        shared_submodules = {
            k: v
            for k, v in _get_components(fl_pipe).items()
            if k != "transformer" and v is not None
        }

        ref_pipe = self._instantiate_pipeline(
            paths["base"],
            self.settings.REF_SUBFOLDER,
            shared_components=shared_submodules,
        )
        self._configure_shared_components(ref_pipe)
        ref_transformer = _get_components(ref_pipe).get(
            "transformer", getattr(ref_pipe, "transformer", None)
        )
        self._prepare_transformer("ref", ref_transformer, fused_specs)
        self.pipelines["ref"] = PipelineAdapter(ref_pipe, self.blocks_module)
        self._enforce_residency_budget(keep_key="fl")

        if "turbo" in paths and self.settings.TURBO_TRANSFORMER_REPO:
            t_cls = type(fl_transformer)
            turbo_transformer = t_cls.from_pretrained(
                str(paths["turbo"]),
                subfolder=self.settings.TURBO_TRANSFORMER_SUBFOLDER or None,
                torch_dtype=torch.bfloat16,
            )
            self._prepare_transformer("turbo", turbo_transformer, fused_specs)
            turbo_pipe = copy.copy(fl_pipe)
            if hasattr(turbo_pipe, "update_components"):
                turbo_pipe.update_components(
                    **shared_submodules, transformer=turbo_transformer
                )
            self.pipelines["turbo"] = PipelineAdapter(turbo_pipe, self.blocks_module)
            self._enforce_residency_budget(keep_key="fl")

        self._run_warmups()
        self.ensure_resident("fl")

    def _run_warmups(self) -> None:
        if self.settings.WARMUP_STEPS <= 0 or not self.settings.WARMUP_BUCKETS.strip():
            return
        bucket_specs = _parse_warmup_buckets(self.settings.WARMUP_BUCKETS)
        if not bucket_specs:
            logger.warning("no usable WARMUP_BUCKETS entry; skipping warm-up")
            return

        for key in list(self.transformers.keys()):
            adapter = self.ensure_resident(key)
            transformer = self.transformers[key]
            cache = self.caches[key]
            tau = (
                self.settings.TURBO_REPAIR_TAU
                if key == "turbo"
                else self.settings.FAST_REPAIR_TAU
            )
            attention_router.configure(
                enabled=self.settings.ATTENTION_BACKEND == "fp8",
                repair_tau=tau,
                min_seq=self.settings.ATTN_FP8_MIN_SEQ,
            )
            for width, height, frames in bucket_specs:
                refs = (
                    [
                        {
                            "type": "image",
                            "data": Image.new("RGB", (width, height), (128, 128, 128)),
                        }
                    ]
                    if key == "ref"
                    else None
                )
                cache.begin(self.settings.WARMUP_STEPS, 0.0, 0, 0)
                try:
                    with torch.inference_mode(), denoiser_scope(transformer):
                        adapter.run(
                            prompt="warmup",
                            height=height,
                            width=width,
                            num_frames=frames,
                            steps=self.settings.WARMUP_STEPS,
                            generator=torch.Generator(device="cuda").manual_seed(0),
                            references=refs,
                        )
                finally:
                    cache.end()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def pipeline_for(self, profile: str, mode: str) -> str:
        if profile == "turbo":
            if "turbo" not in self.transformers and not self.settings.TURBO_TRANSFORMER_REPO:
                raise ValueError("Turbo profile is not configured")
            if mode == "ref2v":
                raise ValueError("Turbo profile does not support ref2v mode")
            return "turbo"
        if mode == "ref2v":
            return "ref"
        return "fl"
