import fcntl
import functools
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import httpx
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file

from config import load_settings
from kernels.fp8_linear import FP8Linear, add_adapters
from schemas import LoraSpec


MAX_LORA_BYTES = 8 * 1024 * 1024 * 1024

_PAIR_PATTERN = re.compile(
    r"^(.*?)(?:"
    r"\.lora_(A|B)(?:\.[^.]+)?\.weight|"
    r"\.lora_(down|up)\.weight|"
    r"\.lora\.(down|up)\.weight"
    r")$"
)


def fused_specs(value: str) -> list[LoraSpec]:
    if not value.strip():
        return []
    try:
        items = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("FUSED_LORAS must be a JSON array of LoRA specifications") from exc
    if not isinstance(items, list):
        raise ValueError("FUSED_LORAS must be a JSON array")
    return [
        LoraSpec(source=item) if isinstance(item, str) else LoraSpec.model_validate(item)
        for item in items
    ]


def resolve_lora(source, weight_name=None) -> Path:
    settings = load_settings()
    directory = Path(settings.MODELS_DIR) / "lora_cache"
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        json.dumps([source, weight_name], separators=(",", ":")).encode()
    ).hexdigest()
    destination = directory / f"{digest}.safetensors"
    lock_path = directory / f"{digest}.lock"

    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if destination.is_file() and destination.stat().st_size:
            return destination
        descriptor, temporary_name = tempfile.mkstemp(dir=directory, suffix=".part")
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            if source.startswith(("https://", "http://")):
                with httpx.Client(
                    follow_redirects=True, timeout=120, max_redirects=5, trust_env=False
                ) as client:
                    with client.stream("GET", source) as response:
                        response.raise_for_status()
                        total = 0
                        with temporary.open("wb") as output:
                            for chunk in response.iter_bytes(1024 * 1024):
                                total += len(chunk)
                                if total > MAX_LORA_BYTES:
                                    raise ValueError("LoRA exceeds the download size limit")
                                output.write(chunk)
            else:
                repo = source.removeprefix("hf://")
                revision = None
                if "@" in repo:
                    repo, revision = repo.rsplit("@", 1)
                if len(repo.split("/")) != 2:
                    raise ValueError("LoRA source must be a URL or an owner/repository ID")
                filename = weight_name
                if filename is None:
                    candidates = [
                        name for name in list_repo_files(repo, revision=revision)
                        if name.endswith(".safetensors")
                    ]
                    preferred = [
                        name for name in (
                            "pytorch_lora_weights.safetensors",
                            "adapter_model.safetensors",
                        )
                        if name in candidates
                    ]
                    if preferred:
                        filename = preferred[0]
                    elif len(candidates) == 1:
                        filename = candidates[0]
                    else:
                        raise ValueError("Specify weight_name for this LoRA repository")
                cached = hf_hub_download(
                    repo_id=repo, filename=filename, revision=revision
                )
                if Path(cached).stat().st_size > MAX_LORA_BYTES:
                    raise ValueError("LoRA exceeds the download size limit")
                shutil.copyfile(cached, temporary)

            if not temporary.stat().st_size:
                raise ValueError("LoRA file is empty")
            load_file(str(temporary), device="cpu")
            os.replace(temporary, destination)
            from runtime.weights import mark_changed
            mark_changed()
            return destination
        finally:
            temporary.unlink(missing_ok=True)


def _canonical(name):
    prefixes = (
        "base_model.model.", "model.diffusion_model.", "diffusion_model.",
        "module.", "model.", "transformer.", "unet.",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
                break
    for prefix in ("lora_unet_", "lora_transformer_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def parse_lora(path, module_names) -> dict:
    tensors = load_file(str(path), device="cpu")
    aliases = {}
    if module_names is not None:
        for name in module_names:
            canonical = _canonical(name)
            for alias in {name, canonical, canonical.replace(".", "_")}:
                aliases.setdefault(alias, set()).add(name)

    groups = {}
    alphas = {}
    unmatched = []
    for key, value in tensors.items():
        match = _PAIR_PATTERN.match(key)
        if match:
            name = _canonical(match.group(1))
            side = next(item for item in match.groups()[1:] if item is not None)
            side = "A" if side in {"A", "down"} else "B"
            group = groups.setdefault(name, {})
            if side in group:
                unmatched.append(key)
            else:
                group[side] = (key, value)
        elif key.endswith((".alpha", ".lora_alpha")):
            name = _canonical(key.rsplit(".", 1)[0])
            if value.numel() != 1 or name in alphas:
                unmatched.append(key)
            else:
                alphas[name] = (key, float(value.item()))
        else:
            unmatched.append(key)

    result = {}
    for raw_name, group in groups.items():
        if "A" not in group or "B" not in group:
            unmatched.extend(item[0] for item in group.values())
            continue
        a_key, a = group["A"]
        b_key, b = group["B"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            unmatched.extend((a_key, b_key))
            continue

        target = raw_name
        if module_names is not None:
            matches = aliases.get(raw_name, set())
            if not matches:
                matches = aliases.get(raw_name.replace(".", "_"), set())
            if len(matches) != 1:
                unmatched.extend((a_key, b_key))
                continue
            target = next(iter(matches))
        if target in result:
            unmatched.extend((a_key, b_key))
            continue

        factor = 1.0
        if raw_name in alphas:
            factor = alphas.pop(raw_name)[1] / a.shape[0]
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            unmatched.extend((a_key, b_key))
            continue
        if not torch.isfinite(torch.tensor(factor)):
            unmatched.append(raw_name + ".alpha")
            continue
        result[target] = (
            a.to(torch.bfloat16).contiguous(),
            (b.float() * factor).to(torch.bfloat16).contiguous(),
        )

    unmatched.extend(item[0] for item in alphas.values())
    if unmatched:
        raise ValueError("Unmatched or invalid LoRA keys: " + ", ".join(unmatched[:10]))
    if not result:
        raise ValueError("The LoRA file contains no compatible matrix pairs")
    return result


def _linear_modules(transformer):
    return {
        name: module
        for name, module in transformer.named_modules()
        if isinstance(module, (nn.Linear, FP8Linear))
    }


def register_adapters(transformer):
    modules = _linear_modules(transformer)
    for module in modules.values():
        if isinstance(module, FP8Linear):
            continue
        if hasattr(module, "_active_adapters"):
            continue
        module._active_adapters = ()
        original = module.forward

        def wrap(current, original_forward):
            @functools.wraps(original_forward)
            def forward(x):
                output = original_forward(x)
                return add_adapters(x, output, current._active_adapters)
            return forward

        module.forward = wrap(module, original)
    return modules


@dataclass(frozen=True)
class PreparedLora:
    key: tuple
    weights: dict
    scale: float


class LoraManager:
    def __init__(self, settings):
        self.capacity = settings.LORA_GPU_CACHE_SIZE
        self.gpu_cache = OrderedDict()
        self.cpu_cache = OrderedDict()
        self.cpu_lock = threading.Lock()
        self.registries = {}

    def register(self, transformer):
        registry = register_adapters(transformer)
        self.registries[id(transformer)] = registry
        return registry

    def prepare(self, transformer, specs):
        if len(specs) > self.capacity:
            raise ValueError("The request exceeds LORA_GPU_CACHE_SIZE")
        registry = self.registries[id(transformer)]
        names = tuple(sorted(registry))
        prepared = []
        for item in specs:
            spec = item if isinstance(item, LoraSpec) else LoraSpec.model_validate(item)
            path = resolve_lora(spec.source, spec.weight_name)
            key = (str(path), names)
            with self.cpu_lock:
                weights = self.cpu_cache.get(key)
                if weights is not None:
                    self.cpu_cache.move_to_end(key)
            if weights is None:
                weights = parse_lora(path, names)
                for name, (a, b) in weights.items():
                    module = registry[name]
                    if a.shape[1] != module.in_features or b.shape[0] != module.out_features:
                        raise ValueError(f"LoRA dimensions do not match {name}")
                with self.cpu_lock:
                    self.cpu_cache[key] = weights
                    self.cpu_cache.move_to_end(key)
                    while len(self.cpu_cache) > self.capacity * 2:
                        self.cpu_cache.popitem(last=False)
            prepared.append(PreparedLora(key, weights, spec.scale))
        return prepared

    @torch.no_grad()
    def activate(self, transformer, specs):
        registry = self.registries[id(transformer)]
        self.deactivate(transformer)
        if specs and not isinstance(specs[0], PreparedLora):
            specs = self.prepare(transformer, specs)
        active = {name: [] for name in registry}
        try:
            for spec in specs:
                weights = self.gpu_cache.get(spec.key)
                if weights is None:
                    while len(self.gpu_cache) >= self.capacity:
                        self.gpu_cache.popitem(last=False)
                    weights = {
                        name: (
                            a.to("cuda", dtype=torch.bfloat16),
                            b.to("cuda", dtype=torch.bfloat16),
                        )
                        for name, (a, b) in spec.weights.items()
                    }
                    self.gpu_cache[spec.key] = weights
                self.gpu_cache.move_to_end(spec.key)
                for name, (a, b) in weights.items():
                    active[name].append((a, b, float(spec.scale)))
            for name, module in registry.items():
                module._active_adapters = tuple(active[name])
        except BaseException:
            self.deactivate(transformer)
            raise

    def deactivate(self, transformer):
        for module in self.registries.get(id(transformer), {}).values():
            module._active_adapters = ()

    def clear_gpu_cache(self):
        self.gpu_cache.clear()


@torch.no_grad()
def fuse_into_weights(transformer, specs):
    modules = _linear_modules(transformer)
    for item in specs:
        spec = item if isinstance(item, LoraSpec) else LoraSpec.model_validate(item)
        path = resolve_lora(spec.source, spec.weight_name)
        weights = parse_lora(path, modules)
        for name, (a, b) in weights.items():
            module = modules[name]
            if not isinstance(module, nn.Linear) or module.weight is None:
                raise ValueError("LoRAs must be fused before FP8 conversion")
            if a.shape[1] != module.in_features or b.shape[0] != module.out_features:
                raise ValueError(f"LoRA dimensions do not match {name}")
            a = a.to(module.weight.device, dtype=torch.float32)
            b = b.to(module.weight.device, dtype=torch.float32)
            for start in range(0, module.out_features, 512):
                stop = min(start + 512, module.out_features)
                delta = b[start:stop] @ a
                updated = module.weight[start:stop].float() + delta * spec.scale
                module.weight[start:stop].copy_(updated.to(module.weight.dtype))
