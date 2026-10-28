"""FP8 (E4M3) flash-attention for Blackwell/Hopper GPUs.

Design notes
------------
* Only shape/dtype *static* properties stay ``tl.constexpr`` (head dim, block
  sizes, boolean feature flags).  Sequence lengths, strides and the softmax scale
  are runtime scalars so a single binary serves every sequence length, and strides
  are ``int64`` inside the kernels so large tensors cannot overflow pointers.
* Quantisation is **per token** for Q and K (a single outlier token can no longer
  degrade 128 other tokens) and per channel for V.
* ``c = softmax_scale * log2(e)`` is folded into the score scaling instead of being
  multiplied into Q before ``amax`` (which used to flush tiny values to zero).
* Key tiles are traversed in order ``0..nk-1``; there is no rotating start offset,
  so numerically identical queries produce identical results.
* Masked entries use a finite ``-1e38`` sentinel that can never dominate a real
  score, probabilities of masked entries are zeroed *before* ``exp2``, and the
  running maximum is monotonic (``alpha <= 1``) so no ``0 * inf`` NaNs are possible.
* ``fp8_attention`` is inference only: it raises if autograd is required rather than
  silently returning detached outputs.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
FP8_E4M3_MAX = 448.0
FLOAT32_MAX = 3.4028234663852886e38
NEG_SENTINEL = -1.0e38
_MIN_FP8_CAPABILITY = (8, 9)
_GRID_MAX = 65535
_HEAD_DIMS = (64, 128)


# ---------------------------------------------------------------------------
# support checks
# ---------------------------------------------------------------------------
def is_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    """Return ``True`` only when :func:`fp8_attention` can run on this input."""
    try:
        if not all(isinstance(t, torch.Tensor) for t in (q, k, v)):
            return False
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            return False
        if q.device != k.device or q.device != v.device:
            return False
        for t in (q, k, v):
            if t.layout != torch.strided or t.is_sparse:
                return False
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            return False
        if q.shape[0] == 0 or q.shape[1] == 0 or q.shape[2] == 0:
            return False
        if k.shape[2] == 0 or k.shape[2] != v.shape[2]:
            return False
        if q.dtype not in (torch.bfloat16, torch.float16):
            return False
        if k.dtype != q.dtype or v.dtype != q.dtype:
            return False
        head_dim = int(q.shape[-1])
        if head_dim not in _HEAD_DIMS:
            return False
        if int(k.shape[-1]) != head_dim or int(v.shape[-1]) != head_dim:
            return False
        if tuple(q.shape[:2]) != tuple(k.shape[:2]) or tuple(q.shape[:2]) != tuple(v.shape[:2]):
            return False
        # ROCm exposes float8e4nv but Triton's CUDA libdevice is unavailable there.
        if getattr(torch.version, "hip", None) is not None:
            return False
        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability(q.device)
        return (major, minor) >= _MIN_FP8_CAPABILITY
    except (RuntimeError, IndexError, AttributeError):  # pragma: no cover - defensive
        return False


def requires_grad(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    return any(t.requires_grad for t in (q, k, v))


# ---------------------------------------------------------------------------
# quantisation kernels
# ---------------------------------------------------------------------------
@triton.jit
def _quantize_q_kernel(
    Q_ptr,
    Q8_ptr,
    ScaleQ_ptr,
    Sq,
    H,
    D: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1).to(tl.int64) + tl.program_id(2).to(tl.int64) * tl.num_programs(
        1
    ).to(tl.int64)
    b = pid_bh // H
    h = pid_bh % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < Sq

    q_ptrs = (
        Q_ptr
        + b * stride_qb
        + h * stride_qh
        + offs_m[:, None].to(tl.int64) * stride_qs
        + offs_d[None, :].to(tl.int64) * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    # Per-token scale (a single outlier token must not degrade its whole tile).
    finite = tl.abs(q) <= FLOAT32_MAX
    amax = tl.max(tl.where(finite, tl.abs(q), 0.0), axis=1)
    scale = tl.where(amax > 0.0, amax / 448.0, 1.0)

    y = q / scale[:, None]
    y = tl.where(finite, y, tl.where(q > 0.0, 448.0, -448.0))
    y = tl.minimum(tl.maximum(y, -448.0), 448.0)
    y = tl.where(q == q, y, q)
    q8 = y.to(tl.float8e4nv)

    out_ptrs = (
        Q8_ptr
        + (pid_bh * Sq + offs_m[:, None].to(tl.int64)) * D
        + offs_d[None, :].to(tl.int64)
    )
    tl.store(out_ptrs, q8, mask=mask_m[:, None])
    tl.store(ScaleQ_ptr + pid_bh * Sq + offs_m, scale, mask=mask_m)


@triton.jit
def _k_mean_kernel(
    K_ptr,
    KMean_ptr,
    Mask_ptr,
    Sk,
    H,
    D,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_mb,
    stride_mn,
    HAS_MASK: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Masked mean over the key axis, computed in one pass (no fp32 copy of K)."""
    pid_d = tl.program_id(0)
    pid_bh = tl.program_id(1).to(tl.int64) + tl.program_id(2).to(tl.int64) * tl.num_programs(
        1
    ).to(tl.int64)
    b = pid_bh // H
    h = pid_bh % H
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    count = tl.zeros((BLOCK_D,), dtype=tl.float32).to(tl.float32)
    for s0 in range(0, Sk, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < Sk
        valid = mask_s
        if HAS_MASK:
            km = tl.load(
                Mask_ptr + b * stride_mb + offs_s.to(tl.int64) * stride_mn,
                mask=mask_s,
                other=False,
            )
            valid = valid & km
        k_ptrs = (
            K_ptr
            + b * stride_kb
            + h * stride_kh
            + offs_s[:, None].to(tl.int64) * stride_ks
            + offs_d[None, :].to(tl.int64) * stride_kd
        )
        k = tl.load(
            k_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0
        ).to(tl.float32)
        finite = tl.abs(k) <= FLOAT32_MAX
        k = tl.where(finite, k, 0.0)
        acc += tl.sum(k, axis=0)
        count += tl.sum(tl.where(valid, 1.0, 0.0), axis=0)
    mean = tl.where(count > 0.0, acc / tl.maximum(count, 1.0), 0.0)
    tl.store(
        KMean_ptr + pid_bh * D + offs_d,
        mean,
        mask=mask_d,
    )


@triton.jit
def _quantize_k_kernel(
    K_ptr,
    KMean_ptr,
    K8_ptr,
    ScaleK_ptr,
    Sk,
    H,
    D: tl.constexpr,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1).to(tl.int64) + tl.program_id(2).to(tl.int64) * tl.num_programs(
        1
    ).to(tl.int64)
    b = pid_bh // H
    h = pid_bh % H
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    mask_n = offs_n < Sk

    k_ptrs = (
        K_ptr
        + b * stride_kb
        + h * stride_kh
        + offs_n[:, None].to(tl.int64) * stride_ks
        + offs_d[None, :].to(tl.int64) * stride_kd
    )
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
    k_mean = tl.load(KMean_ptr + pid_bh * D + offs_d)
    k = tl.where(mask_n[:, None], k - k_mean[None, :], 0.0)

    finite = (tl.abs(k) <= FLOAT32_MAX) & mask_n[:, None]
    amax = tl.max(tl.where(finite, tl.abs(k), 0.0), axis=1)
    scale = tl.where(amax > 0.0, amax / 448.0, 1.0)

    y = k / scale[:, None]
    y = tl.where(finite, y, tl.where(k > 0.0, 448.0, -448.0))
    y = tl.minimum(tl.maximum(y, -448.0), 448.0)
    y = tl.where(k == k, y, k)
    k8 = y.to(tl.float8e4nv)

    out_ptrs = (
        K8_ptr
        + (pid_bh * Sk + offs_n[:, None].to(tl.int64)) * D
        + offs_d[None, :].to(tl.int64)
    )
    tl.store(out_ptrs, k8, mask=mask_n[:, None])
    tl.store(ScaleK_ptr + pid_bh * Sk + offs_n, scale, mask=mask_n)


@triton.jit
def _quantize_v_kernel(
    V_ptr,
    V8T_ptr,
    ScaleV_ptr,
    Sk,
    H,
    D: tl.constexpr,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_v8t,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_bh = tl.program_id(1).to(tl.int64) + tl.program_id(2).to(tl.int64) * tl.num_programs(
        1
    ).to(tl.int64)
    b = pid_bh // H
    h = pid_bh % H
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    amax = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for s0 in range(0, Sk, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < Sk
        v_ptrs = (
            V_ptr
            + b * stride_vb
            + h * stride_vh
            + offs_s[:, None].to(tl.int64) * stride_vs
            + offs_d[None, :].to(tl.int64) * stride_vd
        )
        v_tile = tl.load(
            v_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0
        ).to(tl.float32)
        finite = tl.abs(v_tile) <= FLOAT32_MAX
        amax = tl.maximum(
            amax, tl.max(tl.where(finite, tl.abs(v_tile), 0.0), axis=0)
        )
    scale = tl.where(amax > 0.0, amax / 448.0, 1.0)
    tl.store(ScaleV_ptr + pid_bh * D + offs_d, scale, mask=mask_d)

    for s0 in range(0, Sk, BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask_s = offs_s < Sk
        v_ptrs = (
            V_ptr
            + b * stride_vb
            + h * stride_vh
            + offs_s[:, None].to(tl.int64) * stride_vs
            + offs_d[None, :].to(tl.int64) * stride_vd
        )
        v_tile = tl.load(
            v_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0
        ).to(tl.float32)
        finite = (tl.abs(v_tile) <= FLOAT32_MAX) & mask_s[:, None]
        y = v_tile / scale[None, :]
        y = tl.where(finite, y, tl.where(v_tile > 0.0, 448.0, -448.0))
        y = tl.minimum(tl.maximum(y, -448.0), 448.0)
        y = tl.where(v_tile == v_tile, y, v_tile)
        v8 = y.to(tl.float8e4nv)
        # Rows of V8T are 16-byte aligned via the padded row stride.
        out_ptrs = (
            V8T_ptr
            + (pid_bh * D + offs_d[:, None].to(tl.int64)) * stride_v8t
            + offs_s[None, :].to(tl.int64)
        )
        tl.store(out_ptrs, tl.trans(v8), mask=mask_d[:, None] & mask_s[None, :])


# ---------------------------------------------------------------------------
# attention kernel
# ---------------------------------------------------------------------------
_ATTN_CONFIGS = [
    triton.Config(
        {"BLOCK_N": bn, "NUM_STAGES": ns},
        num_stages=ns,
        num_warps=nw,
    )
    for bn in (64, 128)
    for ns in (2, 3, 4)
    for nw in (4, 8)
]


@triton.autotune(
    configs=_ATTN_CONFIGS,
    # Hardware capability and output dtype are part of the key so cached Triton
    # configurations can never leak across architectures or dtypes.
    key=[
        "D",
        "Sq_bucket",
        "Sk_bucket",
        "HAS_MASK",
        "HAS_BIAS",
        "DO_REPAIR",
        "IS_CC10",
        "OUT_FP16",
    ],
)
@triton.jit
def _fp8_attn_kernel(
    Q8_ptr,
    K8_ptr,
    V8T_ptr,
    ScaleQ_ptr,
    ScaleK_ptr,
    ScaleV_ptr,
    KeyMask_ptr,
    KeyBias_ptr,
    Out_ptr,
    Sq,
    Sk,
    H,
    scale_c,
    stride_mask_b,
    stride_mask_n,
    stride_bias_b,
    stride_bias_n,
    stride_v8t,
    Sq_bucket,
    Sk_bucket,
    D: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    DO_REPAIR: tl.constexpr,
    repair_tau: tl.constexpr,
    OUT_FP16: tl.constexpr,
    IS_CC10: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    BLOCK_M: tl.constexpr = 128,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1).to(tl.int64) + tl.program_id(2).to(tl.int64) * tl.num_programs(
        1
    ).to(tl.int64)
    b = pid_bh // H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < Sq

    q_base = pid_bh * Sq
    q8_ptrs = (
        Q8_ptr
        + (q_base + offs_m[:, None].to(tl.int64)) * D
        + offs_d[None, :].to(tl.int64)
    )
    q8 = tl.load(q8_ptrs, mask=mask_m[:, None], other=0.0)
    scale_q = tl.load(ScaleQ_ptr + q_base + offs_m, mask=mask_m, other=1.0)

    nk = tl.cdiv(Sk, BLOCK_N)
    m = tl.full((BLOCK_M,), NEG_SENTINEL, dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    acc_r = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    v8_row_base = pid_bh * D
    k_base = pid_bh * Sk

    for i in tl.range(0, nk, num_stages=NUM_STAGES):
        offs_n = i * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < Sk
        valid = mask_m[:, None] & mask_n[None, :]

        kb = tl.zeros((BLOCK_N,), dtype=tl.float32)
        if HAS_BIAS:
            kb = tl.load(
                KeyBias_ptr + b * stride_bias_b + offs_n.to(tl.int64) * stride_bias_n,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
            # Only -inf marks a hard mask; finite biases stay additive.
            valid = valid & (kb > -FLOAT32_MAX)
        if HAS_MASK:
            km = tl.load(
                KeyMask_ptr + b * stride_mask_b + offs_n.to(tl.int64) * stride_mask_n,
                mask=mask_n,
                other=False,
            )
            valid = valid & km[None, :]

        n_valid = tl.sum(tl.sum(valid.to(tl.int32), axis=1), axis=0)
        if n_valid > 0:
            k8_ptrs = (
                K8_ptr
                + (k_base + offs_n[:, None].to(tl.int64)) * D
                + offs_d[None, :].to(tl.int64)
            )
            k8 = tl.load(k8_ptrs, mask=mask_n[:, None], other=0.0)
            scale_k = tl.load(ScaleK_ptr + k_base + offs_n, mask=mask_n, other=1.0)

            s = tl.dot(q8, tl.trans(k8)).to(tl.float32)
            s = s * (scale_q * scale_c)[:, None] * scale_k[None, :]
            if HAS_BIAS:
                s = s + tl.where(mask_n, kb, 0.0)[None, :] * 1.4426950408889634
            s = tl.where(valid, s, NEG_SENTINEL)

            rowmax = tl.max(s, axis=1)
            # Monotonic running maximum: alpha = exp2(m - m_new) <= 1 always.
            m_new = tl.where(rowmax > m + 8.0, rowmax, m)
            alpha = tl.math.exp2(m - m_new)
            acc = acc * alpha[:, None]
            l = l * alpha
            if DO_REPAIR:
                acc_r = acc_r * alpha[:, None]
            m = m_new

            p = tl.where(valid, tl.math.exp2(s - m[:, None]), 0.0)
            p8 = p.to(tl.float8e4nv)

            v8_ptrs = (
                V8T_ptr
                + (v8_row_base + offs_d[None, :].to(tl.int64)) * stride_v8t
                + offs_n[:, None].to(tl.int64)
            )
            v8_tile = tl.load(v8_ptrs, mask=mask_n[:, None], other=0.0)

            p_apx = p8.to(tl.float32)
            l = l + tl.sum(p_apx, axis=1)
            acc = tl.dot(p8, v8_tile, acc)

            if DO_REPAIR:
                row_ref = tl.max(p, axis=1)
                delta = tl.where(
                    (p >= repair_tau * row_ref[:, None]) & valid, p - p_apx, 0.0
                )
                delta_bf16 = delta.to(tl.bfloat16)
                acc_r = tl.dot(delta_bf16, v8_tile.to(tl.bfloat16), acc_r)
                # Accumulate `l` from the same values used by the dot product.
                l = l + tl.sum(delta_bf16.to(tl.float32), axis=1)

    scale_v = tl.load(ScaleV_ptr + v8_row_base + offs_d)
    inv_l = 1.0 / tl.maximum(l, 1.0e-12)
    # Divide before scaling to avoid overflowing fp32 for large V magnitudes.
    out = acc * inv_l[:, None]
    if DO_REPAIR:
        out = out + acc_r * inv_l[:, None]
    out = out * scale_v[None, :]

    out_ptrs = (
        Out_ptr
        + (q_base + offs_m[:, None].to(tl.int64)) * D
        + offs_d[None, :].to(tl.int64)
    )
    if OUT_FP16:
        tl.store(out_ptrs, out.to(tl.float16), mask=mask_m[:, None])
    else:
        tl.store(out_ptrs, out.to(tl.bfloat16), mask=mask_m[:, None])


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------
def _bh_grid(bh: int) -> tuple[int, int]:
    """Split ``b * h`` over two grid dimensions (each capped at 65535)."""
    first = min(bh, _GRID_MAX)
    second = triton.cdiv(bh, _GRID_MAX)
    return first, second


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _prepare_key_side(
    tensor: torch.Tensor | None,
    name: str,
    batch: int,
    key_len: int,
    device: torch.device,
    query_dtype: torch.dtype,
    *,
    as_bool: bool,
) -> tuple[torch.Tensor, int, int] | None:
    """Validate/broadcast a per-key mask or bias to a ``(B, S_k)`` *view*.

    Returns ``(view, stride_batch, stride_key)``; ``stride_batch`` is 0 for
    batch-broadcast tensors so no ``B``-fold copy is ever materialised.
    """
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must live on {device}, got {tensor.device}")
    if tensor.layout != torch.strided or tensor.is_sparse:
        raise ValueError(f"{name} must be a dense strided tensor")
    if as_bool:
        tensor = tensor.to(torch.bool)
    else:
        if tensor.is_floating_point() and tensor.dtype != query_dtype:
            raise ValueError(
                f"{name} must match the query dtype ({query_dtype}) when floating point"
            )
        tensor = tensor.to(torch.float32)

    shape = tuple(int(s) for s in tensor.shape)
    if len(shape) == 1:
        if shape[0] != key_len:
            raise ValueError(f"{name} has {shape[0]} keys, expected {key_len}")
        view = tensor.reshape(1, 1, 1, key_len)
    elif len(shape) == 2:
        if shape[-1] != key_len or shape[0] not in (1, batch):
            raise ValueError(f"{name} shape {shape} is not broadcastable to ({batch}, {key_len})")
        view = tensor.reshape(shape[0], 1, 1, key_len)
    elif len(shape) == 3:
        if shape[1] != 1 or shape[2] != key_len or shape[0] not in (1, batch):
            raise ValueError(
                f"{name} shape {shape} is not supported; per-head masks are not "
                "supported by the FP8 kernel"
            )
        view = tensor.reshape(shape[0], 1, 1, key_len)
    elif len(shape) == 4:
        if shape[1] != 1 or shape[2] != 1 or shape[3] != key_len or shape[0] not in (1, batch):
            raise ValueError(
                f"{name} shape {shape} is not supported; expected a per-key mask "
                f"broadcastable to ({batch}, 1, 1, {key_len})"
            )
        view = tensor
    else:
        lead = 1
        for dim in shape[:-3]:
            lead *= dim
        if (
            lead not in (1, batch)
            or shape[-3] != 1
            or shape[-2] != 1
            or shape[-1] != key_len
        ):
            raise ValueError(f"{name} shape {shape} is not broadcastable to ({batch}, {key_len})")
        view = tensor.reshape(lead, 1, 1, key_len)

    expanded = view.expand(batch, 1, 1, key_len)
    return expanded, expanded.stride(0), expanded.stride(3)


def fp8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    key_mask: torch.Tensor | None = None,
    key_bias: torch.Tensor | None = None,
    repair_tau: float = -1.0,
) -> torch.Tensor:
    """FP8 flash attention.

    ``key_mask`` is a boolean per-key keep mask, ``key_bias`` an additive per-key
    bias (``-inf`` acts as a hard mask).  Both must live on ``q.device``.
    """
    if not is_supported(q, k, v):
        raise ValueError("unsupported inputs for fp8_attention")
    if torch.is_grad_enabled() and requires_grad(q, k, v):
        raise RuntimeError(
            "fp8_attention has no backward implementation; use "
            "torch.nn.functional.scaled_dot_product_attention for training"
        )

    B, H, Sq, D = (int(x) for x in q.shape)
    Sk = int(k.shape[2])
    device = q.device
    out_dtype = q.dtype
    scale_value = float(scale) if scale is not None else (1.0 / math.sqrt(D))
    if not math.isfinite(scale_value):
        raise ValueError("scale must be finite")
    c = scale_value * LOG2E

    mask_prepared = _prepare_key_side(
        key_mask, "key_mask", B, Sk, device, out_dtype, as_bool=True
    )
    bias_prepared = _prepare_key_side(
        key_bias, "key_bias", B, Sk, device, out_dtype, as_bool=False
    )
    has_mask = mask_prepared is not None
    has_bias = bias_prepared is not None

    bh_first, bh_second = _bh_grid(B * H)
    block_m = 128
    nq = triton.cdiv(Sq, block_m)
    block_d = 64 if D > 64 else D

    q8 = torch.empty((B, H, Sq, D), dtype=torch.float8_e4m3fn, device=device)
    scale_q = torch.empty((B, H, Sq), dtype=torch.float32, device=device)
    k_mean = torch.empty((B, H, D), dtype=torch.float32, device=device)
    k8 = torch.empty((B, H, Sk, D), dtype=torch.float8_e4m3fn, device=device)
    scale_k = torch.empty((B, H, Sk), dtype=torch.float32, device=device)
    v8t_stride = _round_up(Sk, 16)  # 16-byte aligned rows for vectorised stores
    v8t = torch.empty((B, H, D, v8t_stride), dtype=torch.float8_e4m3fn, device=device)
    scale_v = torch.empty((B, H, D), dtype=torch.float32, device=device)
    out = torch.empty((B, H, Sq, D), dtype=out_dtype, device=device)

    # Dummy pointers for the unused mask/bias arguments (Triton needs a value).
    mask_tensor = mask_prepared[0] if has_mask else q8
    bias_tensor = bias_prepared[0] if has_bias else q8
    mask_stride_b, mask_stride_n = (
        (mask_prepared[1], mask_prepared[2]) if has_mask else (0, 0)
    )
    bias_stride_b, bias_stride_n = (
        (bias_prepared[1], bias_prepared[2]) if has_bias else (0, 0)
    )

    cc_major, cc_minor = torch.cuda.get_device_capability(device)
    is_cc10 = (cc_major, cc_minor) >= (10, 0)
    repair_enabled = float(repair_tau) >= 0.0

    with torch.cuda.device(device):
        _quantize_q_kernel[(nq, bh_first, bh_second)](
            q,
            q8,
            scale_q,
            Sq,
            H,
            D=D,
            stride_qb=q.stride(0),
            stride_qh=q.stride(1),
            stride_qs=q.stride(2),
            stride_qd=q.stride(3),
            BLOCK_M=block_m,
        )

        _k_mean_kernel[(triton.cdiv(D, block_d), bh_first, bh_second)](
            k,
            k_mean,
            mask_tensor,
            Sk,
            H,
            D,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            mask_stride_b,
            mask_stride_n,
            HAS_MASK=has_mask,
            BLOCK_S=256,
            BLOCK_D=block_d,
        )

        _quantize_k_kernel[(triton.cdiv(Sk, 128), bh_first, bh_second)](
            k,
            k_mean,
            k8,
            scale_k,
            Sk,
            H,
            D=D,
            stride_kb=k.stride(0),
            stride_kh=k.stride(1),
            stride_ks=k.stride(2),
            stride_kd=k.stride(3),
            BLOCK_N=128,
        )

        _quantize_v_kernel[(triton.cdiv(D, block_d), bh_first, bh_second)](
            v,
            v8t,
            scale_v,
            Sk,
            H,
            D=D,
            stride_vb=v.stride(0),
            stride_vh=v.stride(1),
            stride_vs=v.stride(2),
            stride_vd=v.stride(3),
            stride_v8t=v8t_stride,
            BLOCK_S=256,
            BLOCK_D=block_d,
        )

        def _grid(meta):
            return (nq, bh_first, bh_second)

        _fp8_attn_kernel[_grid](
            q8,
            k8,
            v8t,
            scale_q,
            scale_k,
            scale_v,
            mask_tensor,
            bias_tensor,
            out,
            Sq,
            Sk,
            H,
            c,
            mask_stride_b,
            mask_stride_n,
            bias_stride_b,
            bias_stride_n,
            v8t_stride,
            Sq // 512,
            Sk // 512,
            D=D,
            HAS_MASK=has_mask,
            HAS_BIAS=has_bias,
            DO_REPAIR=repair_enabled,
            repair_tau=float(repair_tau),
            OUT_FP16=out_dtype == torch.float16,
            IS_CC10=is_cc10,
            BLOCK_M=block_m,
        )
    return out


__all__ = [
    "FP8_E4M3_MAX",
    "is_supported",
    "fp8_attention",
    "requires_grad",
]
