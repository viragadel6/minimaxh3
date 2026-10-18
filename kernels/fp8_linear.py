import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _quantize_rows_kernel(
    X_ptr,
    X8_ptr,
    Scale_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < K
    x = tl.load(X_ptr + row * K + offs_k, mask=mask, other=0.0).to(tl.float32)
    rowamax = tl.max(tl.abs(x), axis=0)
    scale = tl.maximum(rowamax, 1.0e-12) / 448.0
    x_q = tl.minimum(tl.maximum(x / scale, -448.0), 448.0).to(tl.float8e4nv)
    tl.store(X8_ptr + row * K + offs_k, x_q, mask=mask)
    tl.store(Scale_ptr + row, scale)


def quantize_rows(x2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x2d.ndim != 2 or not x2d.is_cuda:
        raise ValueError("quantize_rows expects a 2D CUDA tensor")
    x2d = x2d.contiguous()
    M, K = x2d.shape
    x8 = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x2d.device)
    x_scale = torch.empty((M, 1), dtype=torch.float32, device=x2d.device)
    if M > 0 and K > 0:
        _quantize_rows_kernel[(M,)](
            x2d,
            x8,
            x_scale,
            K=K,
            BLOCK_K=triton.next_power_of_2(K),
        )
    return x8, x_scale


def apply_lora_adapters(
    x: torch.Tensor,
    out: torch.Tensor,
    adapters: list[tuple[torch.Tensor, torch.Tensor, float]],
) -> torch.Tensor:
    if not adapters:
        return out
    for A, B, scale in adapters:
        branch = (x.to(A.dtype) @ A.t()) @ B.t()
        out = out + (branch * scale).to(out.dtype)
    return out


class FP8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adapters: list[tuple[torch.Tensor, torch.Tensor, float]] = []

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear) -> "FP8Linear":
        mod = cls(linear.in_features, linear.out_features)
        weight = linear.weight.detach()
        if not weight.is_cuda:
            weight = weight.cuda()
        w8, w_scale_col = quantize_rows(weight)
        w_scale = w_scale_col.view(1, -1).contiguous()
        mod.register_buffer("w8", w8)
        mod.register_buffer("w_scale", w_scale)
        if linear.bias is not None:
            bias = linear.bias.detach().to(device=w8.device, dtype=torch.bfloat16).contiguous()
        else:
            bias = None
        mod.register_buffer("bias", bias)
        if hasattr(linear, "adapters") and isinstance(linear.adapters, list):
            mod.adapters = list(linear.adapters)
        linear.weight = None
        linear.bias = None
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features)
        M = x2d.shape[0]
        pad = (-M) % 16
        if pad:
            x2d = F.pad(x2d, (0, 0, 0, pad))
        x8, x_scale = quantize_rows(x2d)
        out = torch._scaled_mm(
            x8,
            self.w8.t(),
            scale_a=x_scale,
            scale_b=self.w_scale,
            bias=self.bias,
            out_dtype=torch.bfloat16,
        )
        if isinstance(out, tuple):
            out = out[0]
        if pad:
            out = out[:M]
        out = out.reshape(*orig_shape[:-1], self.out_features).to(x.dtype)
        return apply_lora_adapters(x, out, self.adapters)


def find_primary_block_list(transformer: nn.Module) -> tuple[str, nn.ModuleList]:
    homogeneous: list[tuple[str, nn.ModuleList]] = []
    all_lists: list[tuple[str, nn.ModuleList]] = []
    for name, mod in transformer.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > 0:
            all_lists.append((name, mod))
            if len({type(child) for child in mod}) == 1:
                homogeneous.append((name, mod))
    candidates = homogeneous if homogeneous else all_lists
    if not candidates:
        raise ValueError("No non-empty nn.ModuleList found in transformer")
    return max(candidates, key=lambda item: len(item[1]))


def convert_transformer_to_fp8(
    transformer: nn.Module,
    exclude_patterns: str | list[str],
    keep_edge_blocks: int,
) -> int:
    if isinstance(exclude_patterns, str):
        patterns = [p.strip().lower() for p in exclude_patterns.split(",") if p.strip()]
    else:
        patterns = [p.strip().lower() for p in exclude_patterns if p.strip()]

    list_name, blocks = find_primary_block_list(transformer)
    total_blocks = len(blocks)
    start_idx = max(0, int(keep_edge_blocks))
    end_idx = max(start_idx, total_blocks - int(keep_edge_blocks))

    converted = 0
    for idx in range(start_idx, end_idx):
        block = blocks[idx]
        for sub_name, sub_mod in list(block.named_modules()):
            if not isinstance(sub_mod, nn.Linear):
                continue
            if sub_mod.in_features % 16 != 0 or sub_mod.out_features % 16 != 0:
                continue
            full_path = f"{list_name}.{idx}.{sub_name}".lower() if sub_name else f"{list_name}.{idx}".lower()
            if any(pat in full_path for pat in patterns):
                continue
            fp8_mod = FP8Linear.from_linear(sub_mod)
            if not sub_name:
                blocks[idx] = fp8_mod
            else:
                parent_path, _, attr = sub_name.rpartition(".")
                parent = block.get_submodule(parent_path) if parent_path else block
                setattr(parent, attr, fp8_mod)
            converted += 1

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return converted
