import gc
import importlib
import inspect
import logging
import os
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import torch
from PIL import Image

from kernels import attention_router
from kernels.attention_router import denoiser_scope
from kernels.fp8_linear import convert_transformer_to_fp8, find_block_list
from runtime.lora import LoraManager, fuse_into_weights, fused_specs
from runtime.pipeline_adapter import PipelineAdapter
from runtime.step_cache import StepResidualCache
from runtime.weights import ensure_weights, mark_changed
from shapes import RESOLUTIONS, validate_num_frames


logger = logging.getLogger(__name__)


def _components(pipe):
    components = getattr(pipe, "components", None)
    if not isinstance(components, Mapping):
        raise RuntimeError("The ModularPipeline must expose a components mapping")
    return dict(components)


def _component_names(pipe):
    result = set(_components(pipe))
    for owner in (pipe, getattr(pipe, "blocks", None)):
        if owner is None:
            continue
        for attribute in ("component_names", "expected_components", "component_specs"):
            values = getattr(owner, attribute, None)
            if isinstance(values, Mapping):
                result.update(values)
            elif isinstance(values, (tuple, list, set)):
                for value in values:
                    name = value if isinstance(value, str) else getattr(value, "name", None)
                    if name:
                        result.add(name)
    return result


def _module_bytes(module):
    seen = set()
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        if tensor is None:
            continue
        key = (tensor.device, tensor.untyped_storage().data_ptr())
        if key not in seen:
            seen.add(key)
            total += tensor.untyped_storage().nbytes()
    return total


def _on_cuda(module):
    for tensor in list(module.parameters()) + list(module.buffers()):
        return tensor.is_cuda
    return False


class ModelManager:
    def __init__(self, settings):
        self.settings = settings
        self.transformers = {}
        self.pipelines = {}
        self.caches = {}
        self.resident = OrderedDict()
        self.shared_gpu_ids = set()
        self.loras = LoraManager(settings)
        self.blocks_module = None

    def _load_pipeline(self, directory, shared=None):
        from diffusers import ModularPipeline

        pipe = ModularPipeline.from_pretrained(str(directory))
        if not hasattr(pipe, "load_components") or not hasattr(pipe, "update_components"):
            raise RuntimeError("A compatible diffusers ModularPipeline implementation is required")

        kwargs = {"torch_dtype": torch.bfloat16}
        if shared:
            names = _component_names(pipe)
            reused = {
                name: value for name, value in shared.items()
                if name in names and name != "transformer" and value is not None
            }
            if reused:
                pipe.update_components(**reused)
                signature = inspect.signature(pipe.load_components)
                if "names" not in signature.parameters:
                    raise RuntimeError(
                        "Component sharing requires ModularPipeline.load_components(names=...)"
                    )
                kwargs["names"] = sorted(names.difference(reused))
        pipe.load_components(**kwargs)
        if not isinstance(_components(pipe).get("transformer"), torch.nn.Module):
            raise RuntimeError("The pipeline did not load a transformer component")
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
        return pipe

    def _offload(self, key):
        transformer = self.transformers[key]
        self.caches.get(key, StepResidualCache()).end()
        self.loras.deactivate(transformer)
        transformer.to("cpu")
        with torch.no_grad():
            for parameter in transformer.parameters():
                if not parameter.data.is_pinned():
                    parameter.data = parameter.data.pin_memory()
            for module in transformer.modules():
                for name, buffer in list(module._buffers.items()):
                    if buffer is not None and not buffer.is_pinned():
                        module._buffers[name] = buffer.pin_memory()
        self.resident.pop(key, None)
        gc.collect()
        torch.cuda.empty_cache()

    def _make_room(self, required, protected=()):
        reserve = int(self.settings.RESIDENCY_RESERVE_GIB * 1024**3)
        protected = set(protected)
        free, _ = torch.cuda.mem_get_info()
        if free >= required + reserve:
            return
        self.loras.clear_gpu_cache()
        torch.cuda.empty_cache()
        for key in list(self.resident):
            if key in protected:
                continue
            self._offload(key)
            free, _ = torch.cuda.mem_get_info()
            if free >= required + reserve:
                return
        free, total = torch.cuda.mem_get_info()
        if free < required + reserve:
            raise RuntimeError(
                f"Insufficient GPU memory: free={free / 1024**3:.1f} GiB, "
                f"required={required / 1024**3:.1f} GiB, "
                f"reserve={reserve / 1024**3:.1f} GiB, "
                f"total={total / 1024**3:.1f} GiB"
            )

    def _move_shared(self, pipe):
        for name, component in _components(pipe).items():
            if name == "transformer" or not isinstance(component, torch.nn.Module):
                continue
            if id(component) in self.shared_gpu_ids:
                continue
            component.eval().requires_grad_(False)
            if not _on_cuda(component):
                self._make_room(_module_bytes(component))
                component.to("cuda")
            self.shared_gpu_ids.add(id(component))
            if self.settings.VAE_TILING and "vae" in name:
                enable = getattr(component, "enable_tiling", None)
                if callable(enable):
                    enable()

    def _register_transformer(self, key, transformer, fused):
        transformer.eval().requires_grad_(False)
        self._make_room(_module_bytes(transformer))
        transformer.to("cuda")
        if fused:
            fuse_into_weights(transformer, fused)
        if self.settings.WEIGHT_PRECISION == "fp8":
            count = convert_transformer_to_fp8(
                transformer,
                self.settings.FP8_EXCLUDE_PATTERNS,
                self.settings.FP8_KEEP_EDGE_BLOCKS,
            )
            if count == 0:
                raise RuntimeError("FP8 was requested but no transformer linear layers were converted")
            logger.info("Converted %s linear layers for transformer %s", count, key)

        setter = getattr(transformer, "set_attention_backend", None)
        if callable(setter):
            setter("native")
        self.transformers[key] = transformer
        self.resident[key] = None
        self.loras.register(transformer)
        cache = StepResidualCache().attach(transformer)
        self.caches[key] = cache

        mode = self.settings.COMPILE_MODE.strip()
        if mode.lower() not in {"", "none", "off", "0"}:
            aliases = {
                "maxautotunenocudagraphs": "max-autotune-no-cudagraphs",
                "max-autotune": "max-autotune-no-cudagraphs",
            }
            mode = aliases.get(mode, mode)
            if mode not in {"default", "max-autotune-no-cudagraphs"}:
                raise ValueError(
                    "COMPILE_MODE must disable CUDA graphs for mutable LoRAs and residency swapping"
                )
            torch._dynamo.config.cache_size_limit = 64
            _, blocks = find_block_list(transformer)
            for block in blocks:
                block._va_uncached_forward = torch.compile(
                    block._va_uncached_forward,
                    mode=mode,
                    dynamic=False,
                    fullgraph=False,
                )
        self._make_room(0, protected=(key,))

    def load(self):
        settings = self.settings
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required")
        root = Path(settings.MODELS_DIR)
        root.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(root / "torchinductor_cache")
        os.environ["TRITON_CACHE_DIR"] = str(root / "triton_cache")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_num_threads(settings.CPU_CORES)
        attention_router.install()
        attention_router.configure(
            settings.ATTENTION_BACKEND == "fp8",
            settings.FAST_REPAIR_TAU,
            settings.ATTN_FP8_MIN_SEQ,
        )

        try:
            self.blocks_module = importlib.import_module(settings.MODEL_BLOCKS_MODULE)
        except ImportError as exc:
            raise RuntimeError(
                f"{settings.MODEL_BLOCKS_MODULE} is unavailable. "
                "Install the actual MiniMaxH3-compatible diffusers implementation. "
                "This service does not substitute an unrelated model or pipeline."
            ) from exc

        paths = ensure_weights(settings)
        fused = fused_specs(settings.FUSED_LORAS)

        fl = self._load_pipeline(paths["base"] / settings.FL_SUBFOLDER)
        self._move_shared(fl)
        self._register_transformer("fl", _components(fl)["transformer"], fused)
        self.pipelines["fl"] = PipelineAdapter(fl, self.blocks_module)

        shared = {
            name: component for name, component in _components(fl).items()
            if name != "transformer"
        }
        ref = self._load_pipeline(
            paths["base"] / settings.REF_SUBFOLDER, shared=shared
        )
        self._move_shared(ref)
        self._register_transformer("ref", _components(ref)["transformer"], fused)
        self.pipelines["ref"] = PipelineAdapter(ref, self.blocks_module)

        if "turbo" in paths:
            transformer_cls = type(self.transformers["fl"])
            turbo = transformer_cls.from_pretrained(
                str(paths["turbo"]),
                subfolder=settings.TURBO_TRANSFORMER_SUBFOLDER or None,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            self._register_transformer("turbo", turbo, fused)
            self.pipelines["turbo"] = self.pipelines["fl"]

        for adapter in set(self.pipelines.values()):
            adapter.sample_rate()
        self._warmup()
        mark_changed()

    def pipeline_for(self, profile, mode):
        if profile == "turbo":
            if "turbo" not in self.transformers:
                raise ValueError("The turbo profile is not configured")
            if mode == "ref2v":
                raise ValueError("The FL turbo transformer does not support ref2v")
            return "turbo"
        return "ref" if mode == "ref2v" else "fl"

    def ensure_resident(self, key):
        transformer = self.transformers[key]
        if not _on_cuda(transformer):
            self._make_room(_module_bytes(transformer), protected=(key,))
            transformer.to("cuda", non_blocking=True)
            torch.cuda.current_stream().synchronize()
        self.resident.pop(key, None)
        self.resident[key] = None
        self._make_room(0, protected=(key,))
        adapter = self.pipelines[key]
        adapter.pipe.update_components(transformer=transformer)
        if hasattr(adapter.pipe, "to"):
            adapter.pipe.to("cuda")
        return adapter

    def _warmup(self):
        if self.settings.WARMUP_STEPS == 0 or not self.settings.WARMUP_BUCKETS.strip():
            return
        buckets = []
        for value in self.settings.WARMUP_BUCKETS.split(","):
            parts = value.strip().split(":")
            if len(parts) != 4:
                raise ValueError(f"Invalid warmup bucket: {value}")
            resolution, numerator, denominator, frames = parts
            aspect = f"{numerator}:{denominator}"
            width, height = RESOLUTIONS[resolution][aspect]
            frames = validate_num_frames(int(frames))
            if not 124 <= frames <= 345:
                raise ValueError("Warmup buckets must use native segment lengths")
            buckets.append((width, height, frames))

        for key in self.transformers:
            adapter = self.ensure_resident(key)
            transformer = self.transformers[key]
            cache = self.caches[key]
            tau = (
                self.settings.TURBO_REPAIR_TAU
                if key == "turbo"
                else self.settings.FAST_REPAIR_TAU
            )
            attention_router.configure(
                self.settings.ATTENTION_BACKEND == "fp8",
                tau,
                self.settings.ATTN_FP8_MIN_SEQ,
            )
            for width, height, frames in buckets:
                references = []
                if key == "ref":
                    references = [{
                        "type": "image",
                        "data": Image.new("RGB", (width, height), (127, 127, 127)),
                    }]
                cache.begin(self.settings.WARMUP_STEPS, 0.0, 0, 0)
                try:
                    with torch.inference_mode(), denoiser_scope(transformer):
                        output = adapter.run(
                            prompt="A quiet landscape with gentle ambient sound.",
                            height=height,
                            width=width,
                            num_frames=frames,
                            steps=self.settings.WARMUP_STEPS,
                            generator=torch.Generator(device="cuda").manual_seed(0),
                            references=references,
                        )
                        del output
                finally:
                    cache.end()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
