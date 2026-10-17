import fnmatch

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _quantize_rows(X, Y, SCALES, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    value = tl.load(X + row * K + columns, mask=columns < K, other=0.0)
    value = value.to(tl.float32)
    amax = tl.max(tl.abs(value), axis=0)
    scale = tl.maximum(amax, 1.0e-12) / 448.0
    quantized = tl.minimum(tl.maximum(value / scale, -448.0), 448.0)
    tl.store(
        Y + row * K + columns,
        quantized.to(tl.float8e4nv),
        mask=columns < K,
    )
    tl.store(SCALES + row, scale)


def quantize_rows(x2d):
    if x2d.ndim != 2 or not x2d.is_cuda:
        raise ValueError("quantize_rows requires a two-dimensional CUDA tensor")
    x2d = x2d.contiguous()
    rows, columns = x2d.shape
    if rows == 0 or columns == 0:
        raise ValueError("Cannot quantize an empty matrix")
    output = torch.empty_like(x2d, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows, 1), dtype=torch.float32, device=x2d.device)
    _quantize_rows[(rows,)](
        x2d, output, scales,
        K=columns, BLOCK=triton.next_power_of_2(columns),
    )
    return output, scales


def add_adapters(x, output, adapters):
    for a, b, scale in adapters:
        branch = F.linear(F.linear(x.to(a.dtype), a), b)
        output = output + branch.to(output.dtype) * scale
    return output


class FP8Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._active_adapters = ()

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear):
        if not linear.weight.is_cuda:
            raise ValueError("FP8 conversion requires CUDA-resident weights")
        module = cls(linear.in_features, linear.out_features)
        w8, w_scale = quantize_rows(linear.weight.detach())
        module.register_buffer("w8", w8)
        module.register_buffer("w_scale", w_scale)
        bias = None
        if linear.bias is not None:
            bias = linear.bias.detach().to(torch.bfloat16).clone()
        module.register_buffer("bias", bias)
        module.train(linear.training)
        linear.weight = None
        linear.bias = None
        return module

    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError("FP8Linear input dimension mismatch")
        if not x.is_cuda or not self.w8.is_cuda:
            raise RuntimeError("FP8Linear must be resident on the GPU")
        shape = x.shape[:-1]
        flat = x.reshape(-1, self.in_features)
        if flat.shape[0] == 0:
            return x.new_empty((*shape, self.out_features))
        rows = flat.shape[0]
        padding = (-rows) % 16
        if padding:
            flat = F.pad(flat, (0, 0, 0, padding))
        x8, x_scale = quantize_rows(flat)
        dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
        output = torch._scaled_mm(
            x8,
            self.w8.t(),
            scale_a=x_scale,
            scale_b=self.w_scale.t().contiguous(),
            out_dtype=dtype,
            use_fast_accum=True,
        )
        if isinstance(output, tuple):
            output = output[0]
        output = output[:rows].reshape(*shape, self.out_features).to(x.dtype)
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return add_adapters(x, output, self._active_adapters)


def find_block_list(transformer):
    candidates = [
        (name, module)
        for name, module in transformer.named_modules()
        if isinstance(module, nn.ModuleList) and len(module)
    ]
    if not candidates:
        raise ValueError("The transformer has no nonempty ModuleList")
    return max(candidates, key=lambda item: len(item[1]))


def convert_transformer_to_fp8(transformer, exclude_patterns, keep_edge_blocks) -> int:
    if isinstance(exclude_patterns, str):
        exclude_patterns = exclude_patterns.split(",")
    patterns = [value.strip().lower() for value in exclude_patterns if value.strip()]
    block_path, blocks = find_block_list(transformer)
    count = 0
    last = len(blocks) - keep_edge_blocks
    for index in range(keep_edge_blocks, max(keep_edge_blocks, last)):
        block = blocks[index]
        for relative, module in list(block.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if module.in_features % 16 or module.out_features % 16:
                continue
            name = f"{block_path}.{index}.{relative}".lower()
            if any(pattern in name or fnmatch.fnmatch(name, pattern) for pattern in patterns):
                continue
            if not relative:
                blocks[index] = FP8Linear.from_linear(module)
            else:
                parent_name, _, child_name = relative.rpartition(".")
                parent = block.get_submodule(parent_name) if parent_name else block
                setattr(parent, child_name, FP8Linear.from_linear(module))
            count += 1
    torch.cuda.empty_cache()
    return count
