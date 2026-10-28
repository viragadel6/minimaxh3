"""FP8 (E4M3) row-wise activation quantisation and an FP8 linear layer.

The quantiser is a two-pass, per-row Triton kernel:

* every program owns exactly one row, so the row maximum is exact and no
  cross-block reductions (or atomics) are needed;
* the row is walked in ``BLOCK_K`` chunks, so ``K`` is never forced into a single
  block tensor and ``K`` is a *runtime* argument (one compiled binary, not one per
  feature dimension);
* every global index is computed in ``int64`` so tensors larger than 2^31 elements
  cannot overflow;
* the grid is two dimensional, so more than 2^31 rows can be launched;
* non-finite inputs are handled explicitly (``+-Inf`` saturate, ``NaN`` stays NaN)
  and the scale has no absolute floor: an all-zero row quantises to zeros and a
  tiny-but-representable row is *not* flushed.
"""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

FP8_E4M3_MAX = 448.0
FP8_TORCH_DTYPE = torch.float8_e4m3fn
FLOAT32_MAX = 3.4028234663852886e38
#: CUDA grid dimension 0 supports 2^31-1 blocks; dimensions 1/2 only 65535.
_GRID_MAX = 65535
_BLOCK_K_MAX = 4096
_MIN_FP8_CAPABILITY = (8, 9)


# ---------------------------------------------------------------------------
# device / dtype support
# ---------------------------------------------------------------------------
def fp8_support_error(device: torch.device | str | None = None) -> str | None:
    """Return a human readable reason why FP8 is unavailable, or ``None``."""
    if not torch.cuda.is_available():
        return "FP8 requires CUDA but torch.cuda.is_available() is False"
    if getattr(torch.version, "hip", None) is not None:
        return "FP8 (E4M3) is only supported on NVIDIA CUDA, not ROCm"
    try:
        target = torch.device(device) if device is not None else torch.device("cuda", 0)
    except (TypeError, RuntimeError):  # pragma: no cover - invalid device strings
        return f"invalid CUDA device: {device!r}"
    if target.type != "cuda":
        return f"FP8 requires a CUDA device, got {target}"
    if target.index is not None and target.index >= torch.cuda.device_count():
        return f"CUDA device {target} is not available"
    major, minor = torch.cuda.get_device_capability(target)
    if (major, minor) < _MIN_FP8_CAPABILITY:
        return (
            f"FP8 E4M3 requires compute capability >= 8.9, {target} reports "
            f"{major}.{minor}"
        )
    return None


def is_fp8_supported(device: torch.device | str | None = None) -> bool:
    return fp8_support_error(device) is None


def _require_fp8(device: torch.device) -> None:
    reason = fp8_support_error(device)
    if reason is not None:
        raise RuntimeError(reason)


# ---------------------------------------------------------------------------
# quantisation kernel
# ---------------------------------------------------------------------------
@triton.jit
def _quantize_rows_kernel(
    X_ptr,
    X8_ptr,
    Scale_ptr,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_om,
    stride_ok,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64) + tl.program_id(1).to(tl.int64) * tl.num_programs(
        0
    ).to(tl.int64)
    valid_row = pid < M
    offs = tl.arange(0, BLOCK_K)

    x_row = X_ptr + pid * stride_xm
    o_row = X8_ptr + pid * stride_om

    # Pass 1: exact per-row maximum over finite magnitudes.
    amax = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        idx = k + offs
        mask = valid_row & (idx < K)
        x = tl.load(x_row + idx.to(tl.int64) * stride_xk, mask=mask, other=0.0).to(
            tl.float32
        )
        finite = tl.abs(x) <= FLOAT32_MAX
        amax = tl.maximum(amax, tl.where(finite, tl.abs(x), 0.0))
    row_amax = tl.max(amax, axis=0)
    # No absolute floor: only an exactly-zero row falls back to scale 1.0, which
    # keeps zeros at zero without flushing tiny-but-representable values.
    scale = tl.where(row_amax > 0.0, row_amax / 448.0, 1.0)
    tl.store(Scale_ptr + pid, scale, mask=valid_row)

    # Pass 2: quantise with that scale.
    for k in range(0, K, BLOCK_K):
        idx = k + offs
        mask = valid_row & (idx < K)
        x = tl.load(x_row + idx.to(tl.int64) * stride_xk, mask=mask, other=0.0).to(
            tl.float32
        )
        finite = tl.abs(x) <= FLOAT32_MAX
        y = x / scale
        y = tl.where(finite, y, tl.where(x > 0.0, 448.0, -448.0))
        y = tl.minimum(tl.maximum(y, -448.0), 448.0)
        # NaN must stay NaN; tl.minimum/tl.maximum would otherwise map it to +-448.
        y = tl.where(x == x, y, x)
        tl.store(o_row + idx.to(tl.int64) * stride_ok, y.to(tl.float8e4nv), mask=mask)


def _launch_grid(rows: int) -> tuple[int, int]:
    return (min(rows, _GRID_MAX), triton.cdiv(rows, _GRID_MAX))


def _block_k_for(K: int) -> int:
    return max(128, min(triton.next_power_of_2(K), _BLOCK_K_MAX))


def quantize_rows(x2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise quantise ``x2d`` to FP8 E4M3.

    Returns ``(x8, x_scale)`` with ``x8`` of shape ``(M, K)`` in
    ``torch.float8_e4m3fn`` and ``x_scale`` of shape ``(M, 1)`` in float32.

    The input is never copied: arbitrary row/column strides are passed to the
    kernel.  The kernel is not differentiable; callers that need gradients must
    use the dequantised reference path (see :class:`FP8Linear`).
    """
    if not isinstance(x2d, torch.Tensor):
        raise ValueError("quantize_rows expects a torch.Tensor")
    if x2d.ndim != 2:
        raise ValueError(f"quantize_rows expects a 2D tensor, got {x2d.ndim}D")
    if not x2d.is_cuda:
        raise ValueError("quantize_rows expects a CUDA tensor")
    if x2d.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError(f"quantize_rows does not support dtype {x2d.dtype}")
    _require_fp8(x2d.device)

    M, K = int(x2d.shape[0]), int(x2d.shape[1])
    device = x2d.device
    x8 = torch.empty((M, K), dtype=FP8_TORCH_DTYPE, device=device)
    # `ones` (not `empty`): a K == 0 launch is skipped but the scale is still valid.
    x_scale = torch.ones((M, 1), dtype=torch.float32, device=device)
    if M == 0 or K == 0:
        return x8, x_scale

    grid = _launch_grid(M)
    with torch.cuda.device(device):
        _quantize_rows_kernel[grid](
            x2d,
            x8,
            x_scale,
            M,
            K,
            x2d.stride(0),
            x2d.stride(1),
            x8.stride(0),
            x8.stride(1),
            BLOCK_K=_block_k_for(K),
        )
    return x8, x_scale


# ---------------------------------------------------------------------------
# LoRA adapters (registered as parameters, not free-floating tuples)
# ---------------------------------------------------------------------------
_ADAPTER_ATTR = "_lora_adapters"


class LoraAdapter(nn.Module):
    """A rank-``r`` LoRA adapter stored as trainable-style parameters."""

    def __init__(self, A: torch.Tensor, B: torch.Tensor, scale: float = 1.0) -> None:
        super().__init__()
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("LoRA adapter matrices must be 2D")
        if A.shape[0] != B.shape[1]:
            raise ValueError(
                f"LoRA rank mismatch: A has rank {A.shape[0]}, B expects {B.shape[1]}"
            )
        self.A = nn.Parameter(A.detach().clone(), requires_grad=False)
        self.B = nn.Parameter(B.detach().clone(), requires_grad=False)
        self.scale = float(scale)

    @property
    def rank(self) -> int:
        return int(self.A.shape[0])

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        return self.A, self.B, self.scale

    def extra_repr(self) -> str:
        return f"rank={self.rank}, scale={self.scale}"


def adapter_holder(mod: nn.Module) -> nn.ModuleList:
    """Return (creating if needed) the :class:`nn.ModuleList` owning adapters."""
    holder = mod._modules.get(_ADAPTER_ATTR)
    if holder is None:
        holder = nn.ModuleList()
        mod.add_module(_ADAPTER_ATTR, holder)
    return holder


def add_adapter(mod: nn.Module, A: torch.Tensor, B: torch.Tensor, scale: float) -> None:
    adapter_holder(mod).append(LoraAdapter(A, B, scale))


def clear_adapters(mod: nn.Module) -> None:
    holder = mod._modules.get(_ADAPTER_ATTR)
    if holder is not None:
        holder.clear()


def iter_adapters(mod: nn.Module) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    holder = mod._modules.get(_ADAPTER_ATTR)
    if holder is None:
        return []
    return [module.as_tuple() for module in holder if isinstance(module, LoraAdapter)]


def has_adapters(mod: nn.Module) -> bool:
    holder = mod._modules.get(_ADAPTER_ATTR)
    return bool(holder) if holder is not None else False


def apply_lora_adapters(
    x: torch.Tensor,
    out: torch.Tensor,
    adapters: Sequence[tuple[torch.Tensor, torch.Tensor, float]],
    *,
    inplace: bool = True,
) -> torch.Tensor:
    """Add every LoRA branch of ``adapters`` to ``out``.

    The rank-``r`` inner projection is scaled (not the full ``out_features``
    result), the activation is cast once per dtype group, dtype mismatches between
    ``A`` and ``B`` are promoted instead of raising, device mismatches raise a clear
    error, and accumulation happens in place when ``inplace`` is allowed.
    """
    if not adapters:
        return out
    cached_dtype: torch.dtype | None = None
    x_cast = x
    for A, B, scale in adapters:
        if A.device != x.device or B.device != x.device:
            raise RuntimeError(
                "LoRA adapter device mismatch: adapters live on "
                f"{A.device}/{B.device} but the activation is on {x.device}"
            )
        dtype = A.dtype if A.dtype == B.dtype else torch.promote_types(A.dtype, B.dtype)
        if dtype != cached_dtype:
            x_cast = x.to(dtype)
            cached_dtype = dtype
        inner = (x_cast @ A.to(dtype).t()) * float(scale)
        branch = (inner @ B.to(dtype).t()).to(out.dtype)
        if inplace and not out.requires_grad:
            out.add_(branch)
        else:
            out = out + branch
    return out


# ---------------------------------------------------------------------------
# FP8 linear layer
# ---------------------------------------------------------------------------
def _autocast_output_dtype() -> torch.dtype | None:
    try:
        enabled = torch.is_autocast_enabled("cuda")
        if enabled:
            return torch.get_autocast_dtype("cuda")
        return None
    except (TypeError, AttributeError):  # older torch signatures
        try:
            if torch.is_autocast_enabled():
                return torch.get_autocast_gpu_dtype()
        except (TypeError, AttributeError):  # pragma: no cover
            pass
        return None


_SUPPORTED_OUTPUT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


class FP8Linear(nn.Linear):
    """Drop-in ``nn.Linear`` replacement backed by FP8 weights.

    The class subclasses :class:`torch.nn.Linear` so standard tooling
    (``nn.MultiheadAttention``, PEFT, ...) can read ``weight``/``bias``, while the
    storage stays FP8 (``w8`` buffer + float32 ``w_scale`` buffer).  ``weight`` is a
    lazily dequantised view.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        # Intentionally not calling nn.Linear.__init__: it would register a dense
        # weight Parameter that this class does not use.
        nn.Module.__init__(self)
        if in_features < 0 or out_features < 0:
            raise ValueError("in_features/out_features must be non-negative")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bias_dtype = torch.bfloat16
        self._scaled_mm_ok: bool | None = None
        self._scaled_mm_error: str | None = None
        self.register_buffer(
            "w8", torch.empty((0, 0), dtype=FP8_TORCH_DTYPE), persistent=True
        )
        self.register_buffer(
            "w_scale", torch.empty((1, 0), dtype=torch.float32), persistent=True
        )
        self.register_buffer("bias", None, persistent=True)
        adapter_holder(self)

    # -- construction ---------------------------------------------------
    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "FP8Linear":
        """Build an :class:`FP8Linear` from ``linear`` *without* mutating it."""
        if not isinstance(linear, nn.Linear):
            raise ValueError("from_linear expects an nn.Linear")
        if linear.weight is None:
            raise ValueError("cannot convert a linear layer without a weight")
        in_features = int(linear.in_features)
        out_features = int(linear.out_features)
        if in_features % 16 or out_features % 16:
            raise ValueError(
                "FP8 linear layers require in_features and out_features to be "
                f"multiples of 16 (got {in_features}/{out_features})"
            )
        weight = linear.weight.detach()
        if weight.is_meta:
            raise ValueError("cannot convert meta tensors to FP8")
        target_device = torch.device(device) if device is not None else weight.device
        if target_device.type != "cuda":
            raise ValueError(f"FP8Linear requires a CUDA device, got {target_device}")
        weight = weight.to(device=target_device, dtype=dtype or torch.bfloat16)

        w8, w_scale_col = quantize_rows(weight)
        mod = cls(in_features, out_features)
        mod.to(device=target_device)
        mod.w8 = w8
        mod.w_scale = w_scale_col.view(1, -1).contiguous()
        if linear.bias is not None:
            # Preserve the source dtype: downcasting fp32 to bf16 (or fp16 to bf16)
            # would change numerics for no reason.
            bias = linear.bias.detach().to(device=target_device)
            mod.bias = bias.contiguous()
            mod.bias_dtype = bias.dtype
        else:
            mod.bias = None
        return mod

    # -- registration / casting ----------------------------------------
    def _apply(self, fn, recurse: bool = True):  # type: ignore[override]
        result = super()._apply(fn, recurse)
        self._restore_buffer_dtypes()
        return result

    def _restore_buffer_dtypes(self) -> None:
        """Keep the FP8/float32 invariants after ``.half()``/``.float()``/``.to()``."""
        w8 = self._buffers.get("w8")
        if w8 is not None and w8.numel() and w8.dtype != FP8_TORCH_DTYPE:
            self._buffers["w8"] = w8.to(FP8_TORCH_DTYPE)
        w_scale = self._buffers.get("w_scale")
        if w_scale is not None and w_scale.dtype != torch.float32:
            self._buffers["w_scale"] = w_scale.to(torch.float32)
        bias = self._buffers.get("bias")
        if bias is not None and bias.dtype != self.bias_dtype:
            self._buffers["bias"] = bias.to(self.bias_dtype)

    # -- weight access --------------------------------------------------
    @property
    def weight(self) -> torch.Tensor:  # type: ignore[override]
        """Dequantised BF16 weight of shape ``(out_features, in_features)``."""
        return self.dequantize_weight()

    @weight.setter
    def weight(self, value: torch.Tensor) -> None:  # pragma: no cover - guard rail
        raise RuntimeError(
            "FP8Linear.weight is derived from the w8/w_scale buffers; use "
            "FP8Linear.from_linear() or load_state_dict() instead of assigning weight"
        )

    def is_initialized(self) -> bool:
        w8 = self._buffers.get("w8")
        return w8 is not None and w8.numel() > 0

    def dequantize_weight(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        if not self.is_initialized():
            raise RuntimeError(
                "FP8Linear has no weights: build it with FP8Linear.from_linear()"
            )
        w8 = self._buffers["w8"]
        weight = w8.to(dtype)
        w_scale = self._buffers.get("w_scale")
        if w_scale is not None and w_scale.numel():
            weight = weight * w_scale.reshape(-1, 1).to(dtype)
        return weight

    # -- forward --------------------------------------------------------
    def _preferred_out_dtype(self, x: torch.Tensor) -> torch.dtype:
        autocast_dtype = _autocast_output_dtype()
        if autocast_dtype is not None:
            return autocast_dtype
        return x.dtype if x.dtype in _SUPPORTED_OUTPUT_DTYPES else torch.bfloat16

    def _chunked_dequant_linear(
        self, x2d: torch.Tensor, out_dtype: torch.dtype
    ) -> torch.Tensor:
        """Memory bounded ``x @ W.T`` with the weight dequantised chunk by chunk."""
        w8 = self._buffers["w8"]
        w_scale = self._buffers.get("w_scale")
        x_cast = x2d.to(out_dtype)
        outputs = []
        chunk = 4096
        for start in range(0, self.out_features, chunk):
            end = min(start + chunk, self.out_features)
            w_chunk = w8[start:end].to(out_dtype)
            if w_scale is not None and w_scale.numel():
                w_chunk = w_chunk * w_scale[:, start:end].reshape(-1, 1).to(out_dtype)
            outputs.append(F.linear(x_cast, w_chunk))
        out = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    def _scaled_mm_linear(self, x2d: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        x8, x_scale = quantize_rows(x2d)
        if self._scaled_mm_ok is False:
            raise RuntimeError(self._scaled_mm_error or "scaled_mm unavailable")
        try:
            result = torch._scaled_mm(
                x8,
                self._buffers["w8"].t(),
                scale_a=x_scale,
                scale_b=self._buffers["w_scale"],
                out_dtype=out_dtype,
            )
        except Exception as exc:  # noqa: BLE001 - capability probe
            # Row-wise scale_a + per-column scale_b + bias is not supported by every
            # torch/backend combination; remember it and use the dequantised path.
            self._scaled_mm_ok = False
            self._scaled_mm_error = f"torch._scaled_mm is not usable: {exc}"
            logger.warning("%s; falling back to dequantised matmul", self._scaled_mm_error)
            raise RuntimeError(self._scaled_mm_error) from exc
        self._scaled_mm_ok = True
        if isinstance(result, tuple):
            result = result[0]
        if self.bias is not None:
            result = result + self.bias.to(result.dtype)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if not self.is_initialized():
            raise RuntimeError(
                "FP8Linear has no weights: build it with FP8Linear.from_linear()"
            )
        if not isinstance(x, torch.Tensor):
            raise TypeError("FP8Linear expects a torch.Tensor input")
        if x.ndim < 1:
            raise ValueError("FP8Linear expects at least a 1D input")
        if int(x.shape[-1]) != self.in_features:
            raise ValueError(
                f"FP8Linear expected {self.in_features} input features, got "
                f"{int(x.shape[-1])}"
            )
        orig_shape = x.shape
        out_dtype = self._preferred_out_dtype(x)
        if self.in_features == 0 or self.out_features == 0 or x.numel() == 0:
            return torch.empty(
                (*orig_shape[:-1], self.out_features),
                dtype=out_dtype,
                device=x.device,
            )

        x2d = x.reshape(-1, self.in_features)
        adapters = iter_adapters(self)
        adapter_params = self._modules[_ADAPTER_ATTR].parameters()
        needs_grad = torch.is_grad_enabled() and (
            x2d.requires_grad
            or any(param.requires_grad for param in adapter_params)
            or self._buffers["w8"].requires_grad
        )
        if needs_grad:
            # The Triton quantiser is not differentiable: use the exact reference
            # path so autograd stays correct instead of silently dropping gradients.
            out2d = self._chunked_dequant_linear(x2d, out_dtype)
        else:
            try:
                out2d = self._scaled_mm_linear(x2d, out_dtype)
            except RuntimeError:
                out2d = self._chunked_dequant_linear(x2d, out_dtype)

        out = out2d.reshape(*orig_shape[:-1], self.out_features)
        if out.dtype != x.dtype and _autocast_output_dtype() is None:
            # Outside autocast keep the historical "same dtype as input" contract
            # for fp16/fp32 inputs; autocast semantics are honoured above.
            if x.dtype in _SUPPORTED_OUTPUT_DTYPES:
                out = out.to(x.dtype)
        return apply_lora_adapters(x, out, adapters, inplace=not needs_grad)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, fp8=True"
        )


# ---------------------------------------------------------------------------
# transformer conversion
# ---------------------------------------------------------------------------
def _list_candidates(transformer: nn.Module) -> list[tuple[str, nn.ModuleList, bool, int]]:
    candidates = []
    for name, mod in transformer.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            homogeneous = len({type(child) for child in mod}) == 1
            candidates.append((name, mod, homogeneous, name.count(".") if name else 0))
    return candidates


def find_primary_block_list(transformer: nn.Module) -> tuple[str, nn.ModuleList]:
    """Return the *primary* transformer block list.

    Length is the dominant criterion (a nested homogeneous MoE expert list must
    never beat the longer transformer stack); ties are broken by homogeneity and
    then by shallower nesting.
    """
    candidates = _list_candidates(transformer)
    if not candidates:
        raise ValueError("No non-empty nn.ModuleList found in transformer")
    name, mod, _, _ = max(
        candidates, key=lambda item: (len(item[1]), item[2], -item[3])
    )
    return name, mod


#: A block list is considered "primary" when it holds at least this fraction of the
#: longest list's blocks (dual-stream models expose e.g. ``double_blocks`` and
#: ``single_blocks``).
PRIMARY_LIST_RATIO = 0.5


def find_primary_block_lists(transformer: nn.Module) -> list[tuple[str, nn.ModuleList]]:
    """Return *every* primary block list (handles dual-stream architectures)."""
    candidates = _list_candidates(transformer)
    if not candidates:
        raise ValueError("No non-empty nn.ModuleList found in transformer")
    longest = max(len(item[1]) for item in candidates)
    threshold = max(2, int(longest * PRIMARY_LIST_RATIO))
    selected = [
        (name, mod)
        for name, mod, homogeneous, _ in candidates
        if homogeneous and len(mod) >= threshold
    ]
    if not selected:
        name, mod, _, _ = max(
            candidates, key=lambda item: (len(item[1]), item[2], -item[3])
        )
        selected = [(name, mod)]

    # Drop lists nested inside another selected list (they are sub-blocks).
    selected.sort(key=lambda item: item[0].count("."))
    filtered: list[tuple[str, nn.ModuleList]] = []
    for name, mod in selected:
        if any(
            name.startswith(parent + ".") for parent, _ in filtered if parent
        ):
            continue
        filtered.append((name, mod))
    return filtered


def _join_path(*parts: str) -> str:
    return ".".join(part for part in parts if part)


def _normalize_patterns(exclude_patterns: str | Iterable[str]) -> list[str]:
    if isinstance(exclude_patterns, str):
        raw = exclude_patterns.split(",")
    else:
        raw = list(exclude_patterns)
    return [str(p).strip().lower() for p in raw if str(p).strip()]


def _is_convertible_linear(mod: nn.Module) -> tuple[bool, str]:
    # Exact type check: subclasses may override forward or register hooks and must
    # not be silently replaced by a plain FP8 layer.
    if type(mod) is not nn.Linear:
        return False, "not a plain nn.Linear"
    if mod.weight is None:
        return False, "has no weight"
    if mod.in_features % 16 or mod.out_features % 16:
        return False, (
            f"in_features/out_features not a multiple of 16 "
            f"({mod.in_features}/{mod.out_features})"
        )
    if mod._forward_hooks or mod._forward_pre_hooks or mod._backward_hooks:
        return False, "has hooks"
    if hasattr(mod, "parametrizations"):
        return False, "is parametrized"
    return True, ""


@torch.no_grad()
def convert_transformer_to_fp8(
    transformer: nn.Module,
    exclude_patterns: str | Iterable[str],
    keep_edge_blocks: int,
    *,
    device: torch.device | str | None = None,
    report: list[str] | None = None,
) -> int:
    """Replace eligible ``nn.Linear`` layers inside the primary blocks with FP8.

    ``keep_edge_blocks`` blocks at the start *and* end of every primary list stay in
    their original precision.  Returns the number of converted layers; skipped
    layers are logged and appended to ``report`` when provided.
    """
    patterns = _normalize_patterns(exclude_patterns)
    if isinstance(keep_edge_blocks, bool) or not isinstance(keep_edge_blocks, int):
        raise ValueError("keep_edge_blocks must be an integer")
    if keep_edge_blocks < 0:
        raise ValueError("keep_edge_blocks must be >= 0")

    block_lists = find_primary_block_lists(transformer)
    converted = 0
    for list_name, blocks in block_lists:
        total = len(blocks)
        start_idx = min(int(keep_edge_blocks), total)
        end_idx = max(start_idx, total - int(keep_edge_blocks))
        for idx in range(start_idx, end_idx):
            converted += _convert_block(
                blocks, idx, list_name, patterns, device, report
            )

    if converted:
        torch.cuda.empty_cache()
    return converted


def _convert_block(
    blocks: nn.ModuleList,
    idx: int,
    list_name: str,
    patterns: Sequence[str],
    device: torch.device | str | None,
    report: list[str] | None,
) -> int:
    block = blocks[idx]
    # Snapshot only leaf *paths*; modules are re-resolved before each replacement so
    # a replaced parent can never invalidate (or detach) a pending child.
    targets: list[tuple[str, int]] = []
    for sub_name, sub_mod in block.named_modules():
        ok, _ = _is_convertible_linear(sub_mod)
        if ok:
            depth = 0 if sub_name == "" else sub_name.count(".") + 1
            targets.append((sub_name, depth))
    targets.sort(key=lambda item: -item[1])  # deepest first

    converted = 0
    for sub_name, _depth in targets:
        try:
            current = block if sub_name == "" else block.get_submodule(sub_name)
        except AttributeError:
            # A parent was replaced earlier in this loop; its children no longer exist.
            continue
        ok, reason = _is_convertible_linear(current)
        if not ok:
            continue
        full_path = _join_path(list_name, str(idx), sub_name).lower()
        if any(pattern in full_path for pattern in patterns):
            continue
        try:
            fp8_mod = FP8Linear.from_linear(current, device=device)
        except (ValueError, RuntimeError) as exc:
            message = f"skipped {full_path}: {exc}"
            logger.info(message)
            if report is not None:
                report.append(message)
            continue
        if sub_name == "":
            blocks[idx] = fp8_mod
        else:
            parent_path, _, attr = sub_name.rpartition(".")
            parent = block.get_submodule(parent_path) if parent_path else block
            setattr(parent, attr, fp8_mod)
        converted += 1
    return converted


__all__ = [
    "FP8_E4M3_MAX",
    "FP8_TORCH_DTYPE",
    "FP8Linear",
    "LoraAdapter",
    "add_adapter",
    "apply_lora_adapters",
    "clear_adapters",
    "convert_transformer_to_fp8",
    "find_primary_block_list",
    "find_primary_block_lists",
    "fp8_support_error",
    "has_adapters",
    "is_fp8_supported",
    "iter_adapters",
    "quantize_rows",
]
