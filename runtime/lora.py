import functools
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path

import httpx
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file

from config import Settings, load_settings
from kernels.fp8_linear import FP8Linear, apply_lora_adapters
from schemas import LoraSpec


_PREFIXES = (
    "base_model.model.",
    "diffusion_model.",
    "transformer.",
    "model.",
)

_LORA_KEY_RE = re.compile(
    r"^(?P<base>.+?)\.(?:lora_(?P<ab>A|B)(?:\.[^.]+)?|lora_(?P<du>down|up)|lora\.(?P<du2>down|up))\.weight$"
)


def parse_fused_loras(raw: str) -> list[LoraSpec]:
    text = (raw or "").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        return [
            LoraSpec(source=item) if isinstance(item, str) else LoraSpec.model_validate(item)
            for item in data
        ]
    specs: list[LoraSpec] = []
    for entry in text.split(","):
        part = entry.strip()
        if not part:
            continue
        scale = 1.0
        weight_name = None
        if "::" in part:
            source_part, rest = part.split("::", 1)
            if ":" in rest:
                weight_name, scale_str = rest.rsplit(":", 1)
                scale = float(scale_str)
            else:
                weight_name = rest
            specs.append(LoraSpec(source=source_part, weight_name=weight_name, scale=scale))
        elif ":" in part and not part.startswith(("http://", "https://")):
            source_part, scale_str = part.rsplit(":", 1)
            try:
                scale = float(scale_str)
                specs.append(LoraSpec(source=source_part, scale=scale))
            except ValueError:
                specs.append(LoraSpec(source=part))
        else:
            specs.append(LoraSpec(source=part))
    return specs


def resolve_lora(source: str, weight_name: str | None = None) -> Path:
    settings = load_settings()
    cache_dir = Path(settings.MODELS_DIR) / "lora_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key_str = f"{source}::{weight_name or ''}"
    digest = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
    target_path = cache_dir / f"{digest}.safetensors"
    if target_path.is_file() and target_path.stat().st_size > 0:
        return target_path

    fd, tmp_name = tempfile.mkstemp(dir=str(cache_dir), suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        if source.startswith(("http://", "https://")):
            with httpx.Client(follow_redirects=True, timeout=120) as client:
                resp = client.get(source)
                resp.raise_for_status()
                tmp_path.write_bytes(resp.content)
        elif Path(source).is_file():
            shutil.copyfile(source, tmp_path)
        else:
            repo_id = source.removeprefix("hf://")
            filename = weight_name
            if filename is None:
                files = [
                    f for f in list_repo_files(repo_id) if f.endswith(".safetensors")
                ]
                if not files:
                    raise ValueError(f"No .safetensors file found in {repo_id}")
                preferred = [
                    f
                    for f in files
                    if f in ("pytorch_lora_weights.safetensors", "adapter_model.safetensors")
                ]
                filename = preferred[0] if preferred else files[0]
            downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
            shutil.copyfile(downloaded, tmp_path)
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return target_path


def _strip_prefixes(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in _PREFIXES:
            if key.startswith(p):
                key = key[len(p) :]
                changed = True
                break
    return key


def parse_lora(path: str | Path, module_names: set[str] | list[str] | tuple[str, ...] | dict) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    state_dict = load_file(str(path), device="cpu")
    name_list = list(module_names) if module_names is not None else []
    exact_map: dict[str, str] = {}
    underscore_map: dict[str, str] = {}
    for mod_name in name_list:
        stripped = _strip_prefixes(mod_name)
        exact_map[mod_name] = mod_name
        exact_map[stripped] = mod_name
        us = stripped.replace(".", "_")
        underscore_map[us] = mod_name
        underscore_map[f"lora_unet_{us}"] = mod_name
        underscore_map[f"lora_transformer_{us}"] = mod_name

    pairs: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    alphas: dict[str, tuple[str, float]] = {}
    unmatched: list[str] = []

    for raw_key, tensor in state_dict.items():
        clean_key = _strip_prefixes(raw_key)
        m = _LORA_KEY_RE.match(clean_key)
        if m:
            base = _strip_prefixes(m.group("base"))
            role_raw = m.group("ab") or m.group("du") or m.group("du2")
            role = "A" if role_raw in ("A", "down") else "B"
            entry = pairs.setdefault(base, {})
            entry[role] = (raw_key, tensor)
        elif clean_key.endswith(".alpha") or clean_key.endswith(".lora_alpha"):
            base = _strip_prefixes(clean_key.rsplit(".", 1)[0])
            alphas[base] = (raw_key, float(tensor.item()))
        else:
            unmatched.append(raw_key)

    parsed: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for base, entry in pairs.items():
        if "A" not in entry or "B" not in entry:
            for rk, _ in entry.values():
                unmatched.append(rk)
            continue
        rk_a, A = entry["A"]
        rk_b, B = entry["B"]

        resolved_name: str | None = None
        if not name_list:
            resolved_name = base.removeprefix("lora_unet_").removeprefix("lora_transformer_")
        else:
            if base in exact_map:
                resolved_name = exact_map[base]
            elif base in underscore_map:
                resolved_name = underscore_map[base]
            else:
                trimmed = base.removeprefix("lora_unet_").removeprefix("lora_transformer_")
                if trimmed in exact_map:
                    resolved_name = exact_map[trimmed]
                elif trimmed in underscore_map:
                    resolved_name = underscore_map[trimmed]
                elif trimmed.replace(".", "_") in underscore_map:
                    resolved_name = underscore_map[trimmed.replace(".", "_")]

        if resolved_name is None:
            unmatched.extend([rk_a, rk_b])
            continue

        rank = A.shape[0]
        alpha_entry = alphas.pop(base, None)
        mult = (alpha_entry[1] / rank) if alpha_entry is not None else 1.0
        A_bf16 = A.detach().to(torch.bfloat16).contiguous()
        B_bf16 = (B.detach().float() * mult).to(torch.bfloat16).contiguous()
        parsed[resolved_name] = (A_bf16, B_bf16)

    for rk, _ in alphas.values():
        unmatched.append(rk)

    if unmatched:
        raise ValueError(f"Unmatched LoRA keys: {', '.join(unmatched[:10])}")
    return parsed


def attach_lora_registry(transformer: nn.Module) -> dict[str, nn.Module]:
    registry: dict[str, nn.Module] = {}
    for name, mod in transformer.named_modules():
        if isinstance(mod, (nn.Linear, FP8Linear)):
            registry[name] = mod
            if not hasattr(mod, "adapters") or not isinstance(mod.adapters, list):
                mod.adapters = []
            if isinstance(mod, nn.Linear) and not getattr(mod, "_lora_wrapped", False):
                orig_forward = mod.forward

                def _make_forward(m: nn.Linear, fn):
                    @functools.wraps(fn)
                    def _forward(x: torch.Tensor) -> torch.Tensor:
                        out = fn(x)
                        return apply_lora_adapters(x, out, m.adapters)

                    return _forward

                mod.forward = _make_forward(mod, orig_forward)
                mod._lora_wrapped = True
    return registry


class LoraManager:
    def __init__(self, settings_or_capacity: Settings | int | None = None) -> None:
        if isinstance(settings_or_capacity, int):
            self.capacity = settings_or_capacity
        elif isinstance(settings_or_capacity, Settings):
            self.capacity = settings_or_capacity.LORA_GPU_CACHE_SIZE
        else:
            self.capacity = load_settings().LORA_GPU_CACHE_SIZE
        self.gpu_cache: OrderedDict[tuple[str, int], dict[str, tuple[torch.Tensor, torch.Tensor]]] = OrderedDict()
        self.cpu_cache: OrderedDict[tuple[str, int], dict[str, tuple[torch.Tensor, torch.Tensor]]] = OrderedDict()
        self.registries: dict[int, dict[str, nn.Module]] = {}

    def register(self, transformer: nn.Module) -> dict[str, nn.Module]:
        reg = attach_lora_registry(transformer)
        self.registries[id(transformer)] = reg
        return reg

    def prepare(self, transformer: nn.Module, specs: list[LoraSpec]):
        reg = self.registries.get(id(transformer)) or self.register(transformer)
        mod_names = set(reg.keys())
        prepared = []
        for raw_spec in specs:
            spec = raw_spec if isinstance(raw_spec, LoraSpec) else LoraSpec.model_validate(raw_spec)
            path = resolve_lora(spec.source, spec.weight_name)
            cache_key = (str(path), id(transformer))
            parsed = self.cpu_cache.get(cache_key)
            if parsed is None:
                parsed = parse_lora(path, mod_names)
                self.cpu_cache[cache_key] = parsed
            else:
                self.cpu_cache.move_to_end(cache_key)
            prepared.append((cache_key, parsed, float(spec.scale)))
        return prepared

    @torch.no_grad()
    def activate(self, transformer: nn.Module, specs: list) -> None:
        reg = self.registries.get(id(transformer)) or self.register(transformer)
        self.deactivate(transformer)
        if not specs:
            return
        if not (isinstance(specs[0], tuple) and len(specs[0]) == 3):
            prepared = self.prepare(transformer, specs)
        else:
            prepared = specs

        device = next(
            (
                t.device
                for t in list(transformer.parameters()) + list(transformer.buffers())
                if t is not None
            ),
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        for cache_key, cpu_weights, scale in prepared:
            gpu_weights = self.gpu_cache.get(cache_key)
            if gpu_weights is None:
                while len(self.gpu_cache) >= max(1, self.capacity):
                    self.gpu_cache.popitem(last=False)
                gpu_weights = {
                    m_name: (
                        A.to(device=device, dtype=torch.bfloat16),
                        B.to(device=device, dtype=torch.bfloat16),
                    )
                    for m_name, (A, B) in cpu_weights.items()
                }
                self.gpu_cache[cache_key] = gpu_weights
            else:
                self.gpu_cache.move_to_end(cache_key)

            for m_name, (A_gpu, B_gpu) in gpu_weights.items():
                target_mod = reg.get(m_name)
                if target_mod is not None:
                    target_mod.adapters.append((A_gpu, B_gpu, float(scale)))

    def deactivate(self, transformer: nn.Module) -> None:
        reg = self.registries.get(id(transformer))
        if reg is None:
            return
        for mod in reg.values():
            if hasattr(mod, "adapters") and isinstance(mod.adapters, list):
                mod.adapters.clear()

    def clear_gpu_cache(self) -> None:
        self.gpu_cache.clear()


@torch.no_grad()
def fuse_into_weights(transformer: nn.Module, specs: list[LoraSpec]) -> None:
    if not specs:
        return
    linear_mods = {
        name: mod
        for name, mod in transformer.named_modules()
        if isinstance(mod, nn.Linear) and mod.weight is not None
    }
    for raw_spec in specs:
        spec = raw_spec if isinstance(raw_spec, LoraSpec) else LoraSpec.model_validate(raw_spec)
        path = resolve_lora(spec.source, spec.weight_name)
        parsed = parse_lora(path, set(linear_mods.keys()))
        for mod_name, (A, B) in parsed.items():
            mod = linear_mods[mod_name]
            w = mod.weight
            A_dev = A.to(device=w.device, dtype=torch.float32)
            B_dev = B.to(device=w.device, dtype=torch.float32)
            delta = (B_dev @ A_dev) * float(spec.scale)
            w.copy_((w.float() + delta).to(w.dtype))
