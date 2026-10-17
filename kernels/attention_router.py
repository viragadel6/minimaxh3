import functools
import threading
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from kernels.fp8_attention import fp8_attention, is_supported


enabled = True
repair_tau = 1.0
min_seq = 4096

_local = threading.local()
_install_lock = threading.Lock()
_original_sdpa = F.scaled_dot_product_attention
_installed = False


def configure(enabled: bool, repair_tau: float, min_seq: int):
    globals()["enabled"] = bool(enabled)
    globals()["repair_tau"] = float(repair_tau)
    globals()["min_seq"] = int(min_seq)


def _per_key_mask(mask, q, k):
    if mask is None:
        return True, None, None
    if not isinstance(mask, torch.Tensor) or mask.device != q.device:
        return False, None, None
    if mask.ndim < 2 or mask.ndim > 4 or mask.shape[-2] != 1:
        return False, None, None
    shape = (1,) * (4 - mask.ndim) + tuple(mask.shape)
    batch, _, _, _ = q.shape
    sk = k.shape[2]
    if shape[0] not in (1, batch) or shape[1] != 1 or shape[-1] != sk:
        return False, None, None
    value = mask.reshape(shape).expand(batch, 1, 1, sk)[:, 0, 0, :]
    if value.dtype == torch.bool:
        return True, value.contiguous(), None
    if value.is_floating_point():
        if bool((torch.isnan(value) | torch.isposinf(value)).any().item()):
            return False, None, None
        return True, None, value.float().contiguous()
    return False, None, None


def routed_sdpa(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    enable_gqa=False,
):
    use_fp8 = (
        enabled
        and getattr(_local, "depth", 0) > 0
        and dropout_p == 0.0
        and not is_causal
        and not enable_gqa
        and is_supported(query, key, value)
        and query.shape[2] >= min_seq
        and not (
            torch.is_grad_enabled()
            and any(tensor.requires_grad for tensor in (query, key, value))
        )
    )
    if use_fp8:
        valid, key_mask, key_bias = _per_key_mask(attn_mask, query, key)
        if valid:
            return fp8_attention(
                query, key, value,
                scale=scale,
                key_mask=key_mask,
                key_bias=key_bias,
                repair_tau=repair_tau,
            )
    return _original_sdpa(
        query, key, value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


def install():
    global _installed
    with _install_lock:
        if _installed:
            return
        F.scaled_dot_product_attention = routed_sdpa
        torch._C._nn.scaled_dot_product_attention = routed_sdpa
        _installed = True


@contextmanager
def denoiser_scope(transformer):
    original = transformer.forward

    @functools.wraps(original)
    def scoped_forward(*args, **kwargs):
        _local.depth = getattr(_local, "depth", 0) + 1
        try:
            return original(*args, **kwargs)
        finally:
            _local.depth -= 1

    transformer.forward = scoped_forward
    try:
        yield transformer
    finally:
        transformer.forward = original
