import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


LOG2E = 1.4426950408889634


def is_supported(q, k, v) -> bool:
    tensors = (q, k, v)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        return False
    if not all(value.is_cuda and value.ndim == 4 for value in tensors):
        return False
    if not all(value.device == q.device for value in tensors):
        return False
    if q.dtype not in (torch.bfloat16, torch.float16):
        return False
    if not all(value.dtype == q.dtype for value in tensors):
        return False
    if q.shape[-1] not in (64, 128):
        return False
    if k.shape[-1] != q.shape[-1] or v.shape[-1] != q.shape[-1]:
        return False
    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
        return False
    if k.shape[2] != v.shape[2] or min(q.shape[2], k.shape[2]) == 0:
        return False
    return torch.cuda.get_device_capability(q.device) >= (8, 9)


@triton.jit
def _quantize_qk(
    X, MEAN, Y, SCALES,
    S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    X0: tl.constexpr, X1: tl.constexpr,
    X2: tl.constexpr, X3: tl.constexpr,
    FACTOR: tl.constexpr, HAS_MEAN: tl.constexpr,
    BLOCK: tl.constexpr = 128,
):
    block = tl.program_id(0)
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    rows = block * BLOCK + tl.arange(0, BLOCK)
    channels = tl.arange(0, D)
    value = tl.load(
        X + batch * X0 + head * X1
        + rows[:, None] * X2 + channels[None, :] * X3,
        mask=rows[:, None] < S,
        other=0.0,
    ).to(tl.float32)
    if HAS_MEAN:
        mean = tl.load(MEAN + bh * D + channels)
        value = value - mean[None, :]
    value = tl.where(rows[:, None] < S, value * FACTOR, 0.0)
    amax = tl.max(tl.max(tl.abs(value), axis=1), axis=0)
    scale = tl.maximum(amax, 1.0e-12) / 448.0
    quantized = tl.minimum(tl.maximum(value / scale, -448.0), 448.0)
    tl.store(
        Y + (bh * S + rows[:, None]) * D + channels[None, :],
        quantized.to(tl.float8e4nv),
        mask=rows[:, None] < S,
    )
    tl.store(SCALES + bh * tl.cdiv(S, BLOCK) + block, scale)


@triton.jit
def _v_scales(
    V, SCALES,
    S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    V0: tl.constexpr, V1: tl.constexpr,
    V2: tl.constexpr, V3: tl.constexpr,
    BT: tl.constexpr = 256, BC: tl.constexpr = 16,
):
    bh = tl.program_id(0)
    channels = tl.program_id(1) * BC + tl.arange(0, BC)
    batch, head = bh // H, bh % H
    offsets = tl.arange(0, BT)
    amax = tl.full((BC,), 0.0, tl.float32)
    for start in range(tl.cdiv(S, BT)):
        rows = start * BT + offsets
        value = tl.load(
            V + batch * V0 + head * V1
            + rows[:, None] * V2 + channels[None, :] * V3,
            mask=(rows[:, None] < S) & (channels[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(value), axis=0))
    tl.store(
        SCALES + bh * D + channels,
        tl.maximum(amax, 1.0e-12) / 448.0,
        mask=channels < D,
    )


@triton.jit
def _pack_v(
    V, SCALES, VT,
    S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    V0: tl.constexpr, V1: tl.constexpr,
    V2: tl.constexpr, V3: tl.constexpr,
    BT: tl.constexpr = 256, BC: tl.constexpr = 16,
):
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    channels = tl.program_id(1) * BC + tl.arange(0, BC)
    bh = tl.program_id(2)
    batch, head = bh // H, bh % H
    value = tl.load(
        V + batch * V0 + head * V1
        + rows[:, None] * V2 + channels[None, :] * V3,
        mask=(rows[:, None] < S) & (channels[None, :] < D),
        other=0.0,
    ).to(tl.float32)
    scale = tl.load(SCALES + bh * D + channels, mask=channels < D, other=1.0)
    value = tl.minimum(tl.maximum(value / scale[None, :], -448.0), 448.0)
    tl.store(
        VT + (bh * D + channels[None, :]) * S + rows[:, None],
        value.to(tl.float8e4nv),
        mask=(rows[:, None] < S) & (channels[None, :] < D),
    )


_ATTENTION_CONFIGS = [
    triton.Config(
        {"BLOCK_N": block_n, "PIPE_STAGES": stages},
        num_stages=stages,
        num_warps=warps,
    )
    for block_n in (64, 128)
    for stages in (2, 3, 4)
    for warps in (4, 8)
]


@triton.autotune(
    configs=_ATTENTION_CONFIGS,
    key=["D", "Q_BUCKET", "K_BUCKET", "HAS_MASK", "HAS_BIAS", "REPAIR"],
)
@triton.jit
def _attention(
    Q, K, VT, SQ_SCALE, SK_SCALE, SV_SCALE,
    KEY_MASK, KEY_BIAS, OUT,
    SQ: tl.constexpr, SK: tl.constexpr,
    H: tl.constexpr, D: tl.constexpr,
    Q_BUCKET: tl.constexpr, K_BUCKET: tl.constexpr,
    HAS_MASK: tl.constexpr, HAS_BIAS: tl.constexpr,
    REPAIR: tl.constexpr, REPAIR_TAU: tl.constexpr,
    BLACKWELL: tl.constexpr,
    BLOCK_N: tl.constexpr, PIPE_STAGES: tl.constexpr,
    BLOCK_M: tl.constexpr = 128,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // H
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    channels = tl.arange(0, D)
    q = tl.load(
        Q + (bh * SQ + rows[:, None]) * D + channels[None, :],
        mask=rows[:, None] < SQ,
        other=0.0,
    )
    q_scale = tl.load(SQ_SCALE + bh * tl.cdiv(SQ, 128) + pid_m)

    m = tl.full((BLOCK_M,), -1.0e9, tl.float32)
    denominator = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.full((BLOCK_M, D), 0.0, tl.float32)
    acc_r = tl.full((BLOCK_M, D), 0.0, tl.float32)

    nq = tl.cdiv(SQ, BLOCK_M)
    nk = tl.cdiv(SK, BLOCK_N)
    j0 = (pid_m * nk) // nq
    offsets = tl.arange(0, BLOCK_N)

    for index in tl.range(
        0, nk, num_stages=PIPE_STAGES, warp_specialize=BLACKWELL
    ):
        block = (j0 + index) % nk
        columns = block * BLOCK_N + offsets
        k = tl.load(
            K + (bh * SK + columns[:, None]) * D + channels[None, :],
            mask=columns[:, None] < SK,
            other=0.0,
        )
        k_scale = tl.load(
            SK_SCALE + bh * tl.cdiv(SK, 128) + columns // 128,
            mask=columns < SK,
            other=1.0,
        )
        score = tl.dot(q, tl.trans(k)).to(tl.float32)
        score = score * q_scale * k_scale[None, :]
        valid = (rows[:, None] < SQ) & (columns[None, :] < SK)
        if HAS_MASK:
            permitted = tl.load(
                KEY_MASK + batch * SK + columns,
                mask=columns < SK,
                other=False,
            )
            valid = valid & permitted[None, :]
        if HAS_BIAS:
            bias = tl.load(
                KEY_BIAS + batch * SK + columns,
                mask=columns < SK,
                other=0.0,
            ).to(tl.float32)
            valid = valid & (bias[None, :] != float("-inf"))
            score = score + bias[None, :] * 1.4426950408889634
        score = tl.where(valid, score, -1.0e9)

        rowmax = tl.max(score, axis=1)
        m_new = tl.where((rowmax > m + 8.0) | (m == -1.0e9), rowmax, m)
        alpha = tl.math.exp2(m - m_new)
        denominator = denominator * alpha
        acc = acc * alpha[:, None]
        if REPAIR:
            acc_r = acc_r * alpha[:, None]
        m = m_new

        bits_f = score * 8.0 + (56.0 - 8.0 * m)[:, None]
        bits = libdevice.nearbyint(bits_f)
        bits = tl.minimum(tl.maximum(bits, 0.0), 126.0)
        bits = tl.where((bits_f < 8.0) | ~valid, 0.0, bits)
        p8 = bits.to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        p_apx = p8.to(tl.float32)

        v_tile = tl.load(
            VT + (bh * D + channels[None, :]) * SK + columns[:, None],
            mask=columns[:, None] < SK,
            other=0.0,
        )
        denominator = denominator + tl.sum(p_apx, axis=1)
        acc = tl.dot(p8, v_tile, acc)

        if REPAIR:
            p_exact = tl.math.exp2(score - m[:, None])
            delta = tl.where(p_apx >= REPAIR_TAU, p_exact - p_apx, 0.0)
            denominator = denominator + tl.sum(delta, axis=1)
            acc_r = tl.dot(
                delta.to(tl.bfloat16), v_tile.to(tl.bfloat16), acc_r
            )

    v_scale = tl.load(SV_SCALE + bh * D + channels)
    result = (acc + acc_r) * v_scale[None, :]
    result = result / tl.maximum(denominator[:, None], 1.0e-12)
    tl.store(
        OUT + (bh * SQ + rows[:, None]) * D + channels[None, :],
        result,
        mask=rows[:, None] < SQ,
    )


def fp8_attention(q, k, v, scale=None, key_mask=None, key_bias=None, repair_tau=-1.0):
    if not is_supported(q, k, v):
        raise ValueError("Unsupported FP8 attention inputs")
    if not math.isfinite(float(repair_tau)):
        raise ValueError("repair_tau must be finite")

    batch, heads, sq, dimension = q.shape
    sk = k.shape[2]
    factor = (1.0 / math.sqrt(dimension) if scale is None else float(scale)) * LOG2E
    device = q.device

    for name, value in (("key_mask", key_mask), ("key_bias", key_bias)):
        if value is not None:
            if value.shape != (batch, sk) or value.device != device:
                raise ValueError(f"{name} must have shape [B, Sk] on the input device")

    key_mask = None if key_mask is None else key_mask.to(torch.bool).contiguous()
    key_bias = None if key_bias is None else key_bias.float().contiguous()
    q8 = torch.empty(q.shape, dtype=torch.float8_e4m3fn, device=device)
    k8 = torch.empty(k.shape, dtype=torch.float8_e4m3fn, device=device)
    q_scales = torch.empty(
        (batch, heads, triton.cdiv(sq, 128)), dtype=torch.float32, device=device
    )
    k_scales = torch.empty(
        (batch, heads, triton.cdiv(sk, 128)), dtype=torch.float32, device=device
    )
    k_mean = k.float().mean(dim=2, keepdim=True).contiguous()

    _quantize_qk[(triton.cdiv(sq, 128), batch * heads)](
        q, q, q8, q_scales,
        sq, heads, dimension, *q.stride(),
        FACTOR=factor, HAS_MEAN=False, num_warps=8,
    )
    _quantize_qk[(triton.cdiv(sk, 128), batch * heads)](
        k, k_mean, k8, k_scales,
        sk, heads, dimension, *k.stride(),
        FACTOR=1.0, HAS_MEAN=True, num_warps=8,
    )

    v_scales = torch.empty(
        (batch, heads, dimension), dtype=torch.float32, device=device
    )
    v8t = torch.empty(
        (batch, heads, dimension, sk), dtype=torch.float8_e4m3fn, device=device
    )
    _v_scales[(batch * heads, triton.cdiv(dimension, 16))](
        v, v_scales, sk, heads, dimension, *v.stride(), num_warps=4
    )
    _pack_v[(triton.cdiv(sk, 256), triton.cdiv(dimension, 16), batch * heads)](
        v, v_scales, v8t, sk, heads, dimension, *v.stride(), num_warps=4
    )

    output = torch.empty(q.shape, dtype=q.dtype, device=device)
    _attention[(triton.cdiv(sq, 128), batch * heads)](
        q8, k8, v8t, q_scales, k_scales, v_scales,
        q8 if key_mask is None else key_mask,
        q8 if key_bias is None else key_bias,
        output,
        SQ=sq, SK=sk, H=heads, D=dimension,
        Q_BUCKET=sq // 512, K_BUCKET=sk // 512,
        HAS_MASK=key_mask is not None,
        HAS_BIAS=key_bias is not None,
        REPAIR=repair_tau >= 0.0,
        REPAIR_TAU=float(repair_tau),
        BLACKWELL=torch.cuda.get_device_capability(device) >= (10, 0),
    )
    return output
