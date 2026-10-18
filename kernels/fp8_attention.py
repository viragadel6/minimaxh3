import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


LOG2E = 1.4426950408889634


def is_supported(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    if not (
        isinstance(q, torch.Tensor)
        and isinstance(k, torch.Tensor)
        and isinstance(v, torch.Tensor)
    ):
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return False
    if q.device != k.device or q.device != v.device:
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    if q.dtype not in (torch.bfloat16, torch.float16):
        return False
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    d = q.shape[-1]
    if d not in (64, 128) or k.shape[-1] != d or v.shape[-1] != d:
        return False
    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
        return False
    if k.shape[2] != v.shape[2] or q.shape[2] == 0 or k.shape[2] == 0:
        return False
    return torch.cuda.get_device_capability(q.device) >= (8, 9)


@triton.jit
def _quantize_q_kernel(
    Q_ptr,
    Q8_ptr,
    ScaleQ_ptr,
    Sq: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    c: tl.constexpr,
    BLOCK_M: tl.constexpr = 128,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < Sq
    q_ptrs = (
        Q_ptr
        + b * stride_qb
        + h * stride_qh
        + offs_m[:, None] * stride_qs
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    q = tl.where(mask_m[:, None], q * c, 0.0)
    amax = tl.max(tl.max(tl.abs(q), axis=1), axis=0)
    scale_q = tl.maximum(amax, 1.0e-12) / 448.0
    q_scaled = tl.minimum(tl.maximum(q / scale_q, -448.0), 448.0)
    q8 = q_scaled.to(tl.float8e4nv)
    out_ptrs = Q8_ptr + (pid_bh * Sq + offs_m[:, None]) * D + offs_d[None, :]
    tl.store(out_ptrs, q8, mask=mask_m[:, None])
    nq = tl.cdiv(Sq, BLOCK_M)
    tl.store(ScaleQ_ptr + pid_bh * nq + pid_m, scale_q)


@triton.jit
def _quantize_k_kernel(
    K_ptr,
    KMean_ptr,
    K8_ptr,
    ScaleK_ptr,
    Sk: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kd: tl.constexpr,
    BLOCK_N: tl.constexpr = 128,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    mask_n = offs_n < Sk
    k_ptrs = (
        K_ptr
        + b * stride_kb
        + h * stride_kh
        + offs_n[:, None] * stride_ks
        + offs_d[None, :] * stride_kd
    )
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
    k_mean = tl.load(KMean_ptr + pid_bh * D + offs_d).to(tl.float32)
    k = tl.where(mask_n[:, None], k - k_mean[None, :], 0.0)
    amax = tl.max(tl.max(tl.abs(k), axis=1), axis=0)
    scale_k = tl.maximum(amax, 1.0e-12) / 448.0
    k_scaled = tl.minimum(tl.maximum(k / scale_k, -448.0), 448.0)
    k8 = k_scaled.to(tl.float8e4nv)
    out_ptrs = K8_ptr + (pid_bh * Sk + offs_n[:, None]) * D + offs_d[None, :]
    tl.store(out_ptrs, k8, mask=mask_n[:, None])
    nk = tl.cdiv(Sk, BLOCK_N)
    tl.store(ScaleK_ptr + pid_bh * nk + pid_n, scale_k)


@triton.jit
def _quantize_v_kernel(
    V_ptr,
    V8T_ptr,
    ScaleV_ptr,
    Sk: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vd: tl.constexpr,
    BLOCK_S: tl.constexpr = 256,
):
    pid_bh = tl.program_id(0)
    b = pid_bh // H
    h = pid_bh % H
    offs_d = tl.arange(0, D)
    amax = tl.zeros((D,), dtype=tl.float32)
    num_blocks = tl.cdiv(Sk, BLOCK_S)
    for blk in range(0, num_blocks):
        offs_s = blk * BLOCK_S + tl.arange(0, BLOCK_S)
        mask_s = offs_s < Sk
        v_ptrs = (
            V_ptr
            + b * stride_vb
            + h * stride_vh
            + offs_s[:, None] * stride_vs
            + offs_d[None, :] * stride_vd
        )
        v_tile = tl.load(v_ptrs, mask=mask_s[:, None], other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(v_tile), axis=0))
    scale_v = tl.maximum(amax, 1.0e-12) / 448.0
    tl.store(ScaleV_ptr + pid_bh * D + offs_d, scale_v)
    for blk in range(0, num_blocks):
        offs_s = blk * BLOCK_S + tl.arange(0, BLOCK_S)
        mask_s = offs_s < Sk
        v_ptrs = (
            V_ptr
            + b * stride_vb
            + h * stride_vh
            + offs_s[:, None] * stride_vs
            + offs_d[None, :] * stride_vd
        )
        v_tile = tl.load(v_ptrs, mask=mask_s[:, None], other=0.0).to(tl.float32)
        v_scaled = tl.minimum(tl.maximum(v_tile / scale_v[None, :], -448.0), 448.0)
        v8_t = tl.trans(v_scaled.to(tl.float8e4nv))
        out_ptrs = V8T_ptr + (pid_bh * D + offs_d[:, None]) * Sk + offs_s[None, :]
        tl.store(out_ptrs, v8_t, mask=mask_s[None, :])


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
    key=["D", "Sq_bucket", "Sk_bucket", "HAS_MASK", "HAS_BIAS", "DO_REPAIR"],
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
    Sq: tl.constexpr,
    Sk: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    Sq_bucket: tl.constexpr,
    Sk_bucket: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    DO_REPAIR: tl.constexpr,
    repair_tau: tl.constexpr,
    IS_CC10: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    BLOCK_M: tl.constexpr = 128,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < Sq

    q8_ptrs = Q8_ptr + (pid_bh * Sq + offs_m[:, None]) * D + offs_d[None, :]
    q8 = tl.load(q8_ptrs, mask=mask_m[:, None], other=0.0)

    nq = tl.cdiv(Sq, BLOCK_M)
    nk = tl.cdiv(Sk, BLOCK_N)
    nk_scale = tl.cdiv(Sk, 128)

    scale_q = tl.load(
        ScaleQ_ptr + pid_bh * nq + (offs_m // 128),
        mask=mask_m,
        other=1.0,
    )

    m = tl.full((BLOCK_M,), -1.0e9, dtype=tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    acc_r = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    j0 = (pid_m * nk) // nq
    offs_n_base = tl.arange(0, BLOCK_N)

    for i in tl.range(0, nk, num_stages=NUM_STAGES, warp_specialize=IS_CC10):
        j = (j0 + i) % nk
        offs_n = j * BLOCK_N + offs_n_base
        mask_n = offs_n < Sk

        k8_ptrs = K8_ptr + (pid_bh * Sk + offs_n[:, None]) * D + offs_d[None, :]
        k8 = tl.load(k8_ptrs, mask=mask_n[:, None], other=0.0)

        scale_k_j = tl.load(ScaleK_ptr + pid_bh * nk_scale + (j * BLOCK_N) // 128)
        s = tl.dot(q8, tl.trans(k8)).to(tl.float32) * scale_q[:, None] * scale_k_j

        valid = mask_m[:, None] & mask_n[None, :]
        if HAS_BIAS:
            kb = tl.load(KeyBias_ptr + b * Sk + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            valid = valid & (kb[None, :] > -1.0e8)
            s = s + kb[None, :] * 1.4426950408889634
        if HAS_MASK:
            km = tl.load(KeyMask_ptr + b * Sk + offs_n, mask=mask_n, other=False)
            valid = valid & km[None, :]

        s = tl.where(valid, s, -1.0e9)

        rowmax = tl.max(s, axis=1)
        m_new = tl.where((rowmax > m + 8.0) | (m == -1.0e9), rowmax, m)
        alpha = tl.math.exp2(m - m_new)
        acc = acc * alpha[:, None]
        acc_r = acc_r * alpha[:, None]
        l = l * alpha
        m = m_new

        cst = 56.0 - 8.0 * m
        bits_f = s * 8.0 + cst[:, None]
        rounded = libdevice.rint(bits_f)
        clamped = tl.minimum(tl.maximum(rounded, 0.0), 126.0)
        bits_u8 = tl.where((bits_f < 8.0) | (~valid), 0.0, clamped).to(tl.uint8)
        p8 = bits_u8.to(tl.float8e4nv, bitcast=True)

        v8_ptrs = V8T_ptr + (pid_bh * D + offs_d[None, :]) * Sk + offs_n[:, None]
        v8_tile = tl.load(v8_ptrs, mask=mask_n[:, None], other=0.0)

        p_apx = p8.to(tl.float32)
        l = l + tl.sum(p_apx, axis=1)
        acc = tl.dot(p8, v8_tile, acc)

        if DO_REPAIR:
            p_exact = tl.where(valid, tl.math.exp2(s - m[:, None]), 0.0)
            delta = tl.where(p_apx >= repair_tau, p_exact - p_apx, 0.0)
            acc_r = tl.dot(delta.to(tl.bfloat16), v8_tile.to(tl.bfloat16), acc_r)
            l = l + tl.sum(delta, axis=1)

    scale_v = tl.load(ScaleV_ptr + pid_bh * D + offs_d)
    out = (acc + acc_r) * scale_v[None, :] / tl.maximum(l[:, None], 1.0e-12)
    out_ptrs = Out_ptr + (pid_bh * Sq + offs_m[:, None]) * D + offs_d[None, :]
    tl.store(out_ptrs, out, mask=mask_m[:, None])


def fp8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    key_mask: torch.Tensor | None = None,
    key_bias: torch.Tensor | None = None,
    repair_tau: float = -1.0,
) -> torch.Tensor:
    if not is_supported(q, k, v):
        raise ValueError("Unsupported inputs for fp8_attention")

    B, H, Sq, D = q.shape
    Sk = k.shape[2]
    device = q.device
    c = (scale if scale is not None else (1.0 / math.sqrt(D))) * LOG2E

    BLOCK_M = 128
    nq = triton.cdiv(Sq, BLOCK_M)
    nk_128 = triton.cdiv(Sk, 128)

    q8 = torch.empty((B, H, Sq, D), dtype=torch.float8_e4m3fn, device=device)
    scale_q = torch.empty((B, H, nq), dtype=torch.float32, device=device)
    _quantize_q_kernel[(nq, B * H)](
        q,
        q8,
        scale_q,
        Sq=Sq,
        H=H,
        D=D,
        stride_qb=q.stride(0),
        stride_qh=q.stride(1),
        stride_qs=q.stride(2),
        stride_qd=q.stride(3),
        c=float(c),
        BLOCK_M=BLOCK_M,
    )

    k_mean = k.float().mean(dim=2, keepdim=True).contiguous()
    k8 = torch.empty((B, H, Sk, D), dtype=torch.float8_e4m3fn, device=device)
    scale_k = torch.empty((B, H, nk_128), dtype=torch.float32, device=device)
    _quantize_k_kernel[(nk_128, B * H)](
        k,
        k_mean,
        k8,
        scale_k,
        Sk=Sk,
        H=H,
        D=D,
        stride_kb=k.stride(0),
        stride_kh=k.stride(1),
        stride_ks=k.stride(2),
        stride_kd=k.stride(3),
        BLOCK_N=128,
    )

    v8t = torch.empty((B, H, D, Sk), dtype=torch.float8_e4m3fn, device=device)
    scale_v = torch.empty((B, H, D), dtype=torch.float32, device=device)
    _quantize_v_kernel[(B * H,)](
        v,
        v8t,
        scale_v,
        Sk=Sk,
        H=H,
        D=D,
        stride_vb=v.stride(0),
        stride_vh=v.stride(1),
        stride_vs=v.stride(2),
        stride_vd=v.stride(3),
    )

    has_mask = key_mask is not None
    has_bias = key_bias is not None
    mask_tensor = key_mask.to(torch.bool).contiguous() if has_mask else q8
    bias_tensor = key_bias.to(torch.float32).contiguous() if has_bias else q8

    cc_major, cc_minor = torch.cuda.get_device_capability(device)
    is_cc10 = (cc_major, cc_minor) >= (10, 0)

    out = torch.empty_like(q)
    _fp8_attn_kernel[(nq, B * H)](
        q8,
        k8,
        v8t,
        scale_q,
        scale_k,
        scale_v,
        mask_tensor,
        bias_tensor,
        out,
        Sq=Sq,
        Sk=Sk,
        H=H,
        D=D,
        Sq_bucket=Sq // 512,
        Sk_bucket=Sk // 512,
        HAS_MASK=has_mask,
        HAS_BIAS=has_bias,
        DO_REPAIR=float(repair_tau) >= 0.0,
        repair_tau=float(repair_tau),
        IS_CC10=is_cc10,
        BLOCK_M=BLOCK_M,
    )
    return out.to(q.dtype)
