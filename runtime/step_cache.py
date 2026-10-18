import functools

import torch
import torch.nn as nn

from kernels.fp8_linear import find_primary_block_list


def _clone_state(state):
    if isinstance(state, torch.Tensor):
        return state.detach().clone()
    if isinstance(state, tuple):
        return tuple(_clone_state(x) for x in state)
    if isinstance(state, list):
        return [_clone_state(x) for x in state]
    return state


def _add_state(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a + b.to(dtype=a.dtype, device=a.device)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return tuple(_add_state(x, y) for x, y in zip(a, b))
    if isinstance(a, list) and isinstance(b, list):
        return [_add_state(x, y) for x, y in zip(a, b)]
    return a


def _sub_state(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a - b.to(dtype=a.dtype, device=a.device)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return tuple(_sub_state(x, y) for x, y in zip(a, b))
    if isinstance(a, list) and isinstance(b, list):
        return [_sub_state(x, y) for x, y in zip(a, b)]
    return a


def _rel_diff(h1, h1_prev) -> float:
    if isinstance(h1, torch.Tensor) and isinstance(h1_prev, torch.Tensor):
        rel = torch.mean(torch.abs(h1.float() - h1_prev.float())) / torch.clamp(
            torch.mean(torch.abs(h1_prev.float())), min=1e-6
        )
        return float(rel.item())
    if isinstance(h1, (tuple, list)) and isinstance(h1_prev, (tuple, list)):
        vals = [
            _rel_diff(x, y)
            for x, y in zip(h1, h1_prev)
            if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
        ]
        return sum(vals) / max(1, len(vals))
    return 0.0


def _extract_input_hidden(args: tuple, kwargs: dict, reference_output=None):
    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    for k in (
        "hidden_states",
        "video_hidden_states",
        "audio_hidden_states",
        "encoder_hidden_states",
        "x",
    ):
        if k in kwargs and isinstance(kwargs[k], torch.Tensor):
            tensors.append(kwargs[k])
    for v in kwargs.values():
        if isinstance(v, torch.Tensor) and all(v is not t for t in tensors):
            tensors.append(v)

    if isinstance(reference_output, tuple):
        count = len(reference_output)
        return tuple(tensors[:count])
    if isinstance(reference_output, list):
        count = len(reference_output)
        return list(tensors[:count])
    if tensors:
        return tensors[0]
    return args[0] if args else next(iter(kwargs.values()))


class StepResidualCache:
    def __init__(self) -> None:
        self.transformer: nn.Module | None = None
        self.blocks: nn.ModuleList | None = None
        self.active: bool = False
        self.num_steps: int = 0
        self.threshold: float = 0.0
        self.warmup: int = 3
        self.tail: int = 2
        self.step: int = -1
        self.acc_rel: float = 0.0
        self.skip_step: bool = False
        self.h1_prev = None
        self.h1_this_step = None
        self.residual = None
        self._output_template = None

    def attach(self, transformer: nn.Module) -> "StepResidualCache":
        self.transformer = transformer
        _, self.blocks = find_primary_block_list(transformer)
        orig_forward = transformer.forward

        @functools.wraps(orig_forward)
        def _transformer_forward(*args, **kwargs):
            if self.active:
                self.step += 1
            return orig_forward(*args, **kwargs)

        transformer.forward = _transformer_forward

        for idx, block in enumerate(self.blocks):
            self.wrap_block(idx, block)
        return self

    def wrap_block(self, idx: int, block: nn.Module) -> None:
        raw_forward = getattr(block, "_uncached_forward", block.forward)
        block._uncached_forward = raw_forward

        @functools.wraps(raw_forward)
        def _block_forward(*args, **kwargs):
            if not self.active or self.blocks is None:
                return block._uncached_forward(*args, **kwargs)

            num_blocks = len(self.blocks)
            last_idx = num_blocks - 1

            if idx == 0:
                h1 = block._uncached_forward(*args, **kwargs)
                self._output_template = h1
                if self.h1_prev is not None:
                    rel = _rel_diff(h1, self.h1_prev)
                    self.acc_rel += rel
                else:
                    self.acc_rel = float("inf")
                self.h1_prev = _clone_state(h1)

                can_skip = (
                    self.warmup <= self.step < (self.num_steps - self.tail)
                    and self.residual is not None
                    and self.acc_rel < self.threshold
                )
                if can_skip:
                    self.skip_step = True
                else:
                    self.skip_step = False
                    self.acc_rel = 0.0
                    self.h1_this_step = _clone_state(h1)
                return h1

            if idx < last_idx:
                if self.skip_step:
                    return _extract_input_hidden(args, kwargs, self._output_template)
                return block._uncached_forward(*args, **kwargs)

            if self.skip_step:
                inp = _extract_input_hidden(args, kwargs, self._output_template)
                return _add_state(inp, self.residual)

            output_hidden = block._uncached_forward(*args, **kwargs)
            if self.h1_this_step is not None:
                self.residual = _clone_state(
                    _sub_state(output_hidden, self.h1_this_step)
                )
                self.h1_this_step = None
            return output_hidden

        block.forward = _block_forward

    def begin(
        self,
        num_steps: int,
        threshold: float,
        warmup: int = 3,
        tail: int = 2,
    ) -> None:
        self.end()
        self.num_steps = int(num_steps)
        self.threshold = float(threshold)
        self.warmup = int(warmup)
        self.tail = int(tail)
        self.active = (
            self.threshold > 0.0
            and self.num_steps > 8
            and self.blocks is not None
            and len(self.blocks) >= 2
        )

    def end(self) -> None:
        self.active = False
        self.step = -1
        self.num_steps = 0
        self.threshold = 0.0
        self.acc_rel = 0.0
        self.skip_step = False
        self.h1_prev = None
        self.h1_this_step = None
        self.residual = None
        self._output_template = None
