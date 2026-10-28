"""LoRA discovery, parsing, activation and fusing.

Parsing is deliberately strict: ambiguous keys, mismatched tensor shapes, multi
element alpha tensors and duplicate module resolutions all raise instead of silently
picking one of the candidates.  Activation stores adapters in registered
``nn.ModuleList`` holders (so ``.to()``/``state_dict``/optimizers see them) and the
per-module ``forward`` override is installed via a generated subclass, which keeps
``pickle``/``deepcopy`` working.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file

from config import Settings, load_settings
from runtime.lora_keys import (
    _alpha_base,
    _scheme,
    _build_name_maps,
    _match_lora_key,
    _resolve_module_name,
    _strip_prefixes,
    parse_fused_loras,
)
from va_kernels.fp8_linear import (
    FP8Linear,
    add_adapter,
    apply_lora_adapters,
    clear_adapters,
    iter_adapters,
    quantize_rows,
)
from schemas import LoraSpec

logger = logging.getLogger(__name__)

_PREFERRED_WEIGHT_NAMES = (
    "adapter_model.safetensors",
    "pytorch_lora_weights.safetensors",
    "diffusion_pytorch_model.safetensors",
)
_SAFETENSORS_MAGIC_LEN = 8
_MAX_HEADER_BYTES = 100 * 1024 * 1024
_HTTP_RETRIES = 3

_resolve_lock_guard = threading.Lock()
_resolve_locks: dict[str, threading.Lock] = {}


# ---------------------------------------------------------------------------
# LoraSpec parsing
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# resolution / caching
# ---------------------------------------------------------------------------
def _validate_safetensors(path: Path) -> None:
    """Reject empty/partial HTML or JSON bodies before they enter the cache."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            header_len_raw = handle.read(_SAFETENSORS_MAGIC_LEN)
            if len(header_len_raw) != _SAFETENSORS_MAGIC_LEN:
                raise ValueError("file is shorter than the safetensors header")
            header_len = int.from_bytes(header_len_raw, "little")
            if header_len <= 0 or header_len > _MAX_HEADER_BYTES:
                raise ValueError(f"invalid safetensors header length {header_len}")
            if _SAFETENSORS_MAGIC_LEN + header_len > size:
                raise ValueError("truncated safetensors file")
            header = json.loads(handle.read(header_len).decode("utf-8"))
            if not isinstance(header, dict):
                raise ValueError("safetensors header is not an object")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name} is not a valid safetensors file: {exc}") from exc


def _cache_paths(source: str, weight_name: str | None) -> tuple[Path, Path]:
    settings = load_settings()
    cache_dir = Path(settings.MODELS_DIR) / "lora_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Length-prefixed key: unambiguous for values that themselves contain "::".
    weight_key = "" if weight_name is None else f"{len(weight_name)}:{weight_name}"
    key = f"{len(source)}:{source}|{weight_key}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.safetensors", cache_dir / f"{digest}.json"


def _resolve_lock(key: str) -> threading.Lock:
    with _resolve_lock_guard:
        lock = _resolve_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _resolve_locks[key] = lock
        return lock


def _download_http(source: str, target: Path, meta_path: Path) -> None:
    import httpx

    meta = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
    headers = {}
    if meta.get("etag"):
        headers["If-None-Match"] = meta["etag"]
    elif meta.get("last_modified"):
        headers["If-Modified-Since"] = meta["last_modified"]

    last_error: Exception | None = None
    for attempt in range(_HTTP_RETRIES):
        try:
            with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120.0)) as client:
                with client.stream("GET", source, headers=headers) as response:
                    if response.status_code == 304 and target.is_file():
                        return
                    response.raise_for_status()
                    tmp = target.with_suffix(".partial")
                    with tmp.open("wb") as handle:
                        for chunk in response.iter_bytes(1024 * 1024):
                            handle.write(chunk)
                    _validate_safetensors(tmp)
                    tmp.replace(target)
                    meta_path.write_text(
                        json.dumps(
                            {
                                "etag": response.headers.get("etag"),
                                "last_modified": response.headers.get("last-modified"),
                                "fetched_at": time.time(),
                            }
                        ),
                        encoding="utf-8",
                    )
                    return
        except httpx.HTTPStatusError as exc:
            raise ValueError(
                f"HTTP {exc.response.status_code} while downloading {source}"
            ) from exc
        except (httpx.TransportError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < _HTTP_RETRIES:
                time.sleep(0.5 * (2**attempt))
    raise ValueError(f"failed to download {source}: {last_error}")


def _pick_hub_file(repo_id: str, weight_name: str | None) -> str:
    if weight_name:
        return weight_name
    files = [f for f in list_repo_files(repo_id) if f.endswith(".safetensors")]
    if not files:
        raise ValueError(f"no .safetensors file found in {repo_id}")
    by_name = {Path(f).name: f for f in files}
    preferred = [by_name[name] for name in _PREFERRED_WEIGHT_NAMES if name in by_name]
    if preferred:
        return preferred[0]
    if len(files) == 1:
        return files[0]
    raise ValueError(
        f"{repo_id} contains {len(files)} safetensors files; set weight_name to one of "
        f"{sorted(files)[:8]}"
    )


def _load_adapter_config(path: Path) -> dict:
    for candidate in (path.parent / "adapter_config.json", path.with_suffix(".json")):
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                return data
    return {}


def resolve_lora(source: str, weight_name: str | None = None) -> Path:
    """Resolve a LoRA spec to a validated local safetensors file.

    Local files and Hugging Face hub files are used *in place* (no duplicate copy);
    remote HTTP files are streamed into a content-addressed cache under ``MODELS_DIR``.
    """
    source = (source or "").strip()
    if not source:
        raise ValueError("LoRA source must be a non-empty string")
    weight_name = (weight_name or "").strip() or None  # None and "" behave identically

    if source.startswith(("data:", "data;")):
        raise ValueError("LoRA weights cannot be loaded from a data: URI")

    if _scheme(source) in ("http", "https"):
        target, meta_path = _cache_paths(source, weight_name)
        lock = _resolve_lock(str(target))
        with lock:
            if target.is_file():
                try:
                    _validate_safetensors(target)
                    return target
                except ValueError:
                    target.unlink(missing_ok=True)
            _download_http(source, target, meta_path)
        return target

    local = Path(source).expanduser()
    try:
        is_file = local.is_file()
    except OSError:
        is_file = False
    if is_file:
        resolved = local.resolve()
        _validate_safetensors(resolved)
        return resolved

    if "/" in source or not local.exists():
        repo_id = source.removeprefix("hf://")
        filename = _pick_hub_file(repo_id, weight_name)
        downloaded = Path(hf_hub_download(repo_id=repo_id, filename=filename))
        _validate_safetensors(downloaded)
        return downloaded

    raise ValueError(f"unsupported LoRA source: {source}")


# ---------------------------------------------------------------------------
# state dict parsing
# ---------------------------------------------------------------------------
def parse_lora(
    path: str | Path,
    module_names: set[str] | list[str] | tuple[str, ...] | dict,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Load a LoRA file and return ``{module_name: (A, B)}`` ready to apply."""
    name_list = list(module_names) if module_names is not None else []
    if not name_list:
        raise ValueError(
            "parse_lora requires the list of target module names; refusing to return "
            "unvalidated keys"
        )
    exact_map, underscore_map = _build_name_maps(name_list)
    state_dict = load_file(str(path), device="cpu")
    config = _load_adapter_config(Path(path))
    default_mult: float | None = None
    if config:
        alpha = config.get("lora_alpha")
        rank = config.get("r")
        try:
            if alpha is not None and rank:
                default_mult = float(alpha) / float(rank)
        except (TypeError, ValueError):
            default_mult = None

    pairs: dict[tuple[str, str], dict[str, tuple[str, torch.Tensor]]] = {}
    alphas: dict[str, tuple[str, float]] = {}
    unmatched: list[str] = []

    for raw_key, tensor in state_dict.items():
        matched = _match_lora_key(raw_key)
        if matched is not None:
            base, role, adapter = matched
            base = _strip_prefixes(base)
            pairs.setdefault((base, adapter), {})[role] = (raw_key, tensor)
            continue
        alpha_base = _alpha_base(raw_key)
        if alpha_base is not None:
            if tensor.numel() != 1:
                raise ValueError(
                    f"alpha tensor {raw_key} has {tensor.numel()} elements; "
                    "exactly one is required"
                )
            alphas[alpha_base] = (raw_key, float(tensor.item()))
            continue
        unmatched.append(raw_key)

    parsed: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    sources: dict[str, str] = {}
    for (base, _adapter), entry in pairs.items():
        if "A" not in entry or "B" not in entry:
            for raw_key, _ in entry.values():
                unmatched.append(raw_key)
            continue
        raw_key_a, A = entry["A"]
        raw_key_b, B = entry["B"]
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError(
                f"LoRA matrices for {base!r} must be 2D "
                f"(got {tuple(A.shape)} and {tuple(B.shape)})"
            )
        if A.shape[0] == 0:
            raise ValueError(f"LoRA matrices for {base!r} have zero rank")
        if B.shape[1] != A.shape[0]:
            raise ValueError(
                f"LoRA rank mismatch for {base!r}: A has rank {A.shape[0]}, "
                f"B expects rank {B.shape[1]}"
            )
        resolved = _resolve_module_name(base, exact_map, underscore_map)
        if resolved is None:
            unmatched.extend([raw_key_a, raw_key_b])
            continue
        if resolved in parsed:
            raise ValueError(
                f"LoRA keys {sources[resolved]!r} and {raw_key_a!r} both target module "
                f"{resolved!r}; refusing to silently overwrite"
            )
        sources[resolved] = raw_key_a

        rank = int(A.shape[0])
        alpha_entry = alphas.pop(base, None)
        if alpha_entry is not None:
            mult = alpha_entry[1] / rank
        elif default_mult is not None:
            mult = default_mult
        else:
            mult = 1.0

        a_out = A.detach().to(dtype).contiguous()
        if mult == 1.0:
            b_out = B.detach().to(dtype).contiguous()
        else:
            b_out = (B.detach().float() * float(mult)).to(dtype).contiguous()
        parsed[resolved] = (a_out, b_out)

    for raw_key, _ in alphas.values():
        unmatched.append(raw_key)
    if unmatched:
        preview = ", ".join(unmatched[:10])
        suffix = "" if len(unmatched) <= 10 else f" (+{len(unmatched) - 10} more)"
        raise ValueError(f"unmatched LoRA keys: {preview}{suffix}")
    return parsed


# ---------------------------------------------------------------------------
# registry / adapters
# ---------------------------------------------------------------------------
_WRAPPED_SUBCLASSES: dict[type, type] = {}
_registry_lock = threading.RLock()


def _make_lora_forward(base_cls: type):
    def forward(self, *args, **kwargs):
        output = base_cls.forward(self, *args, **kwargs)
        adapters = iter_adapters(self)
        if not adapters:
            return output
        activation = args[0] if args else kwargs.get("input", kwargs.get("x"))
        if not isinstance(activation, torch.Tensor):
            return output
        return apply_lora_adapters(
            activation,
            output,
            adapters,
            inplace=not torch.is_grad_enabled(),
        )

    forward.__name__ = "forward"
    return forward


def _wrap_linear_module(mod: nn.Module) -> None:
    with _registry_lock:
        cls = type(mod)
        if getattr(cls, "_va_lora_wrapped", False):
            return
        sub_cls = _WRAPPED_SUBCLASSES.get(cls)
        if sub_cls is None:
            name = f"{cls.__name__}VaLoraWrapped"
            sub_cls = type(
                name,
                (cls,),
                {
                    "forward": _make_lora_forward(cls),
                    "__module__": __name__,
                    "__qualname__": name,
                    "__doc__": cls.__doc__,
                    "_va_lora_wrapped": True,
                },
            )
            globals()[name] = sub_cls
            _WRAPPED_SUBCLASSES[cls] = sub_cls
        mod.__class__ = sub_cls


def attach_lora_registry(transformer: nn.Module) -> dict[str, nn.Module]:
    """Attach adapter holders to every linear module and return a name map."""
    registry: dict[str, nn.Module] = {}
    for name, mod in transformer.named_modules():
        if not isinstance(mod, (nn.Linear, FP8Linear)):
            continue
        registry[name] = mod
        # FP8Linear applies adapters in its own forward; plain linears need a wrapper.
        if not isinstance(mod, FP8Linear):
            _wrap_linear_module(mod)
    return registry


def _module_device(mod: nn.Module) -> torch.device:
    """First device of ``mod`` (parameters then buffers), scanned lazily."""
    for tensor in mod.parameters():
        return tensor.device
    for tensor in mod.buffers():
        return tensor.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class _PreparedLora:
    path: str
    weights: dict[str, tuple[torch.Tensor, torch.Tensor]]
    scale: float


class LoraManager:
    """Owns the adapter registry and the GPU/CPU LoRA caches."""

    def __init__(self, settings_or_capacity: Settings | int | None = None) -> None:
        if isinstance(settings_or_capacity, int):
            self.capacity = settings_or_capacity
            self.cpu_capacity = 32
        elif isinstance(settings_or_capacity, Settings):
            self.capacity = settings_or_capacity.LORA_GPU_CACHE_SIZE
            self.cpu_capacity = settings_or_capacity.LORA_CPU_CACHE_SIZE
        else:
            settings = load_settings()
            self.capacity = settings.LORA_GPU_CACHE_SIZE
            self.cpu_capacity = settings.LORA_CPU_CACHE_SIZE
        self.gpu_cache: OrderedDict[tuple[str, str], dict[str, tuple[torch.Tensor, torch.Tensor]]] = OrderedDict()
        self.cpu_cache: OrderedDict[Any, dict[str, tuple[torch.Tensor, torch.Tensor]]] = OrderedDict()
        # Weak references: a collected transformer releases its registry, and module
        # ids can never be reused while the module is alive.
        self.registries: "weakref.WeakKeyDictionary[nn.Module, dict[str, nn.Module]]" = (
            weakref.WeakKeyDictionary()
        )
        self._lock = threading.RLock()

    # -- registry --------------------------------------------------------
    def register(self, transformer: nn.Module) -> dict[str, nn.Module]:
        with self._lock:
            existing = self.registries.get(transformer)
            if existing is not None:
                return existing
            registry = attach_lora_registry(transformer)
            self.registries[transformer] = registry
            return registry

    def unregister(self, transformer: nn.Module) -> None:
        with self._lock:
            self.registries.pop(transformer, None)

    # -- preparation -----------------------------------------------------
    def _cpu_weights(self, path: Path, mod_names: Sequence[str]) -> dict:
        key = (str(path), frozenset(mod_names))
        cached = self.cpu_cache.get(key)
        if cached is not None:
            self.cpu_cache.move_to_end(key)
            return cached
        parsed = parse_lora(path, mod_names)
        self.cpu_cache[key] = parsed
        while len(self.cpu_cache) > self.cpu_capacity:
            self.cpu_cache.popitem(last=False)
        return parsed

    def prepare(self, transformer: nn.Module, specs: Sequence) -> list[_PreparedLora]:
        registry = self.registries.get(transformer)
        if registry is None:
            registry = self.register(transformer)
        mod_names = sorted(registry.keys())
        prepared: list[_PreparedLora] = []
        for raw_spec in specs:
            spec = raw_spec if isinstance(raw_spec, LoraSpec) else LoraSpec.model_validate(raw_spec)
            path = resolve_lora(spec.source, spec.weight_name)
            prepared.append(
                _PreparedLora(
                    path=str(path),
                    weights=self._cpu_weights(path, mod_names),
                    scale=float(spec.scale),
                )
            )
        return prepared

    # -- activation ------------------------------------------------------
    def activate(self, transformer: nn.Module, specs: Sequence) -> None:
        registry = self.registries.get(transformer)
        if registry is None:
            registry = self.register(transformer)
        if not specs:
            self.deactivate(transformer)
            return
        if isinstance(specs[0], _PreparedLora) or (
            isinstance(specs[0], tuple) and len(specs[0]) == 3
        ):
            prepared = [
                item
                if isinstance(item, _PreparedLora)
                else _PreparedLora(path=str(item[0]), weights=item[1], scale=float(item[2]))
                for item in specs
            ]
        else:
            # Prepare *before* clearing the current adapters: a failure must not
            # leave the transformer without its previously active LoRAs.
            prepared = self.prepare(transformer, specs)

        module_devices = {name: _module_device(mod) for name, mod in registry.items()}
        plan: list[tuple[_PreparedLora, dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]]]] = []
        for item in prepared:
            by_device: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
            for name, weights in item.weights.items():
                mod = registry.get(name)
                if mod is None:
                    continue
                device = str(module_devices[name])
                by_device.setdefault(device, {})[name] = weights
            plan.append((item, by_device))

        self.deactivate(transformer)
        with self._lock:
            required = [
                (item.path, device)
                for item, by_device in plan
                for device in by_device
            ]
            self._ensure_capacity(required)
            for item, by_device in plan:
                for device, weights_by_name in by_device.items():
                    key = (item.path, device)
                    cached = self.gpu_cache.get(key)
                    if cached is None:
                        cached = {
                            name: (
                                A.to(device=device, dtype=torch.bfloat16),
                                B.to(device=device, dtype=torch.bfloat16),
                            )
                            for name, (A, B) in weights_by_name.items()
                        }
                        self.gpu_cache[key] = cached
                    else:
                        self.gpu_cache.move_to_end(key)
                    for name, (A_gpu, B_gpu) in cached.items():
                        target = registry.get(name)
                        if target is not None:
                            add_adapter(target, A_gpu, B_gpu, item.scale)

    def _ensure_capacity(self, required: Iterable[tuple[str, str]]) -> None:
        """Evict cached LoRAs the current activation does not need."""
        required = list(required)
        for key in required:
            self.gpu_cache.pop(key, None)
        while len(self.gpu_cache) + len(required) > max(1, self.capacity):
            if not self.gpu_cache:
                break
            evicted, _ = self.gpu_cache.popitem(last=False)
            logger.debug("evicted LoRA %s from the GPU cache", evicted)

    def deactivate(self, transformer: nn.Module) -> None:
        registry = self.registries.get(transformer)
        if registry is None:
            return
        for mod in registry.values():
            clear_adapters(mod)

    def clear_gpu_cache(self) -> None:
        with self._lock:
            for transformer in list(self.registries.keys()):
                self.deactivate(transformer)
            self.gpu_cache.clear()


# ---------------------------------------------------------------------------
# fusing
# ---------------------------------------------------------------------------
@torch.no_grad()
def fuse_into_weights(transformer: nn.Module, specs: Sequence[LoraSpec]) -> int:
    """Fuse LoRA deltas into dense weights (all-or-nothing).

    Every delta is accumulated in float32 for every target weight first; the weights
    are only mutated once every spec parsed successfully, so a failure cannot leave a
    partially fused model behind.
    """
    if not specs:
        return 0
    targets: dict[str, nn.Module] = {}
    for name, mod in transformer.named_modules():
        if isinstance(mod, FP8Linear):
            targets[name] = mod
        elif isinstance(mod, nn.Linear) and mod.weight is not None:
            targets[name] = mod
    if not targets:
        raise ValueError("no linear modules are available to fuse LoRA weights into")

    deltas: dict[int, torch.Tensor] = {}
    owners: dict[int, nn.Module] = {}
    skipped: list[str] = []
    for raw_spec in specs:
        spec = raw_spec if isinstance(raw_spec, LoraSpec) else LoraSpec.model_validate(raw_spec)
        path = resolve_lora(spec.source, spec.weight_name)
        parsed = parse_lora(path, set(targets.keys()), dtype=torch.float32)
        for mod_name, (A, B) in parsed.items():
            mod = targets[mod_name]
            if not isinstance(mod, FP8Linear) and hasattr(mod, "parametrizations"):
                skipped.append(mod_name)
                continue
            weight = mod.weight if not isinstance(mod, FP8Linear) else mod.w8
            key = id(weight)
            delta = (B.float() @ A.float()) * float(spec.scale)
            if key in deltas:
                deltas[key] = deltas[key] + delta
            else:
                deltas[key] = delta
                owners[key] = mod

    for key, delta in deltas.items():
        mod = owners[key]
        if isinstance(mod, FP8Linear):
            merged = mod.dequantize_weight(torch.float32) + delta.to(mod.w8.device)
            w8, w_scale_col = quantize_rows(merged.to(torch.bfloat16))
            mod.w8 = w8
            mod.w_scale = w_scale_col.view(1, -1).contiguous()
        else:
            weight = mod.weight
            weight.copy_((weight.float() + delta.to(weight.device)).to(weight.dtype))
        clear_adapters(mod)
    if skipped:
        logger.warning(
            "skipped parametrized weights while fusing LoRA: %s", ", ".join(skipped[:5])
        )
    return len(deltas)


__all__ = [
    "LoraManager",
    "attach_lora_registry",
    "fuse_into_weights",
    "parse_fused_loras",
    "parse_lora",
    "resolve_lora",
]

