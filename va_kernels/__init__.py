"""FP8 kernels and the attention router.

Importing the package itself is cheap: submodules (and therefore torch/triton) are
loaded lazily on first attribute access.  ``kernels`` was renamed to ``va_kernels``
because the former shadows the PyPI ``kernels`` package that diffusers imports.
"""

from __future__ import annotations

import importlib
from typing import Any

_SUBMODULES = ("attention_router", "fp8_attention", "fp8_linear")

_EXPORTS: dict[str, str] = {
    # fp8_linear
    "FP8_E4M3_MAX": "fp8_linear",
    "FP8_TORCH_DTYPE": "fp8_linear",
    "FP8Linear": "fp8_linear",
    "LoraAdapter": "fp8_linear",
    "add_adapter": "fp8_linear",
    "apply_lora_adapters": "fp8_linear",
    "clear_adapters": "fp8_linear",
    "convert_transformer_to_fp8": "fp8_linear",
    "find_primary_block_list": "fp8_linear",
    "find_primary_block_lists": "fp8_linear",
    "fp8_support_error": "fp8_linear",
    "has_adapters": "fp8_linear",
    "is_fp8_supported": "fp8_linear",
    "iter_adapters": "fp8_linear",
    "quantize_rows": "fp8_linear",
    # fp8_attention
    "fp8_attention": "fp8_attention",
    "is_supported": "fp8_attention",
    "requires_grad": "fp8_attention",
    # attention_router
    "RouterConfig": "attention_router",
    "configure": "attention_router",
    "denoiser_scope": "attention_router",
    "get_config": "attention_router",
    "install": "attention_router",
    "instrument_transformer": "attention_router",
    "is_installed": "attention_router",
    "routed_sdpa": "attention_router",
    "uninstall": "attention_router",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    submodule = _EXPORTS.get(name)
    if submodule is None:
        if name in _SUBMODULES:
            return importlib.import_module(f".{name}", __name__)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{submodule}", __name__)
    try:
        value = getattr(module, name)
    except AttributeError as exc:  # pragma: no cover - guards stale __all__ maps
        raise AttributeError(
            f"{submodule!r} does not export {name!r}"
        ) from exc
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_SUBMODULES))
