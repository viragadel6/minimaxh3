import functools
import threading
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from kernels.fp8_attention import fp8_attention, is_supported


enabled: bool = True
repair_tau: float = -1.0
min_seq: int = 4096

_tls = threading.local()
_lock = threading.Lock()
_orig_sdpa = F.scaled_dot_product_attention
_installed: bool = False


def _get_depth() -> int:
    return getattr(_tls, "depth", 0)


def configure(enabled: bool, repair_tau: float, min_seq: int) -> None:
    globals()["enabled"] = bool(enabled)
    globals()["repair_tau"] = float(repair_tau)
    globals()["min_seq"] = int(min_seq)


def _extract_per_key_mask(
    attn_mask: torch.Tensor | None,
    B: int,
    Sk: int,
    device: torch.device,
) -> tuple[bool, torch.Tensor | None, torch.Tensor | None]:
    if attn_mask is None:
        return True, None, None
    if not isinstance(attn_mask, torch.Tensor) or attn_mask.device != device:
        return False, None, None
    if attn_mask.ndim < 2 or attn_mask.ndim > 4:
        return False, None, None
    if attn_mask.shape[-2] != 1 or attn_mask.shape[-1] != Sk:
        return False, None, None
    padded_shape = (1,) * (4 - attn_mask.ndim) + tuple(attn_mask.shape)
    if padded_shape[0] not in (1, B) or padded_shape[1] != 1:
        return False, None, None
    expanded = attn_mask.reshape(padded_shape).expand(B, 1, 1, Sk)[:, 0, 0, :]
    if expanded.dtype == torch.bool:
        return True, expanded.contiguous(), None
    if expanded.is_floating_point():
        return True, None, expanded.to(torch.float32).contiguous()
    return False, None, None


def routed_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    if (
        _get_depth() > 0
        and enabled
        and dropout_p == 0.0
        and not is_causal
        and not enable_gqa
        and isinstance(query, torch.Tensor)
        and isinstance(key, torch.Tensor)
        and isinstance(value, torch.Tensor)
        and query.shape[-1] == key.shape[-1] == value.shape[-1]
        and is_supported(query, key, value)
        and query.shape[2] >= min_seq
    ):
        ok, key_mask, key_bias = _extract_per_key_mask(
            attn_mask, query.shape[0], key.shape[2], query.device
        )
        if ok:
            return fp8_attention(
                query,
                key,
                value,
                scale=scale,
                key_mask=key_mask,
                key_bias=key_bias,
                repair_tau=repair_tau,
            )

    return _orig_sdpa(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


def install() -> None:
    global _installed
    with _lock:
        if _installed:
            return
        F.scaled_dot_product_attention = routed_sdpa
        if hasattr(torch._C, "_nn") and hasattr(
            torch._C._nn, "scaled_dot_product_attention"
        ):
            torch._C._nn.scaled_dot_product_attention = routed_sdpa
        _installed = True


def instrument_transformer(transformer):
    if getattr(transformer, "_denoiser_scope_instrumented", False):
        return transformer
    orig_forward = transformer.forward

    @functools.wraps(orig_forward)
    def _wrapped_forward(*args, **kwargs):
        with denoiser_scope(transformer):
            return orig_forward(*args, **kwargs)

    transformer.forward = _wrapped_forward
    transformer._denoiser_scope_instrumented = True
    return transformer


@contextmanager
def denoiser_scope(transformer=None):
    _tls.depth = _get_depth() + 1
    try:
        yield transformer
    finally:
        _tls.depth = max(0, _get_depth() - 1)
