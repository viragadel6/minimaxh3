import functools
import inspect

import torch

from kernels.fp8_linear import find_block_list


_STATE_NAMES = {
    "hidden_states", "video_hidden_states", "audio_hidden_states",
    "encoder_hidden_states", "text_hidden_states", "image_hidden_states",
    "x", "x_video", "x_audio", "img", "txt", "video", "audio",
}


def _leaves(value, path=()):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif type(value) in (list, tuple):
        for index, item in enumerate(value):
            yield from _leaves(item, path + (index,))
    elif type(value) is dict:
        for name, item in value.items():
            yield from _leaves(item, path + (name,))


def _lookup(value, path):
    for item in path:
        value = value[item]
    return value


def _layout(output, arguments):
    candidates = []
    for name, value in arguments.items():
        if name in _STATE_NAMES or name.endswith("hidden_states"):
            for path, tensor in _leaves(value):
                candidates.append((name, path, tensor))

    def build(value):
        if isinstance(value, torch.Tensor):
            matches = [
                (name, path)
                for name, path, tensor in candidates
                if tensor.shape == value.shape
                and tensor.dtype == value.dtype
                and tensor.device == value.device
            ]
            if len(matches) != 1:
                raise ValueError("Block state mapping is ambiguous")
            return ("tensor", matches[0], tuple(value.shape))
        if type(value) in (tuple, list):
            return (type(value).__name__, [build(item) for item in value])
        if type(value) is dict:
            return ("dict", {name: build(item) for name, item in value.items()})
        if value is None:
            return ("none",)
        raise ValueError("Unsupported transformer block output structure")

    return build(output)


def _restore(layout, arguments):
    kind = layout[0]
    if kind == "tensor":
        name, path = layout[1]
        value = _lookup(arguments[name], path)
        if tuple(value.shape) != layout[2]:
            raise ValueError("Block state shape changed during a cached trajectory")
        return value
    if kind == "tuple":
        return tuple(_restore(item, arguments) for item in layout[1])
    if kind == "list":
        return [_restore(item, arguments) for item in layout[1]]
    if kind == "dict":
        return {name: _restore(item, arguments) for name, item in layout[1].items()}
    return None


def _binary(left, right, operation):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape:
            raise ValueError("Incompatible cached state shapes")
        return operation(left, right)
    if type(left) is not type(right):
        raise ValueError("Incompatible cached state structures")
    if type(left) in (tuple, list):
        if len(left) != len(right):
            raise ValueError("Incompatible cached state lengths")
        values = [_binary(a, b, operation) for a, b in zip(left, right)]
        return tuple(values) if type(left) is tuple else values
    if type(left) is dict:
        if left.keys() != right.keys():
            raise ValueError("Incompatible cached state keys")
        return {name: _binary(left[name], right[name], operation) for name in left}
    if left is None and right is None:
        return None
    raise ValueError("Unsupported cached state")


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if type(value) is tuple:
        return tuple(_clone(item) for item in value)
    if type(value) is list:
        return [_clone(item) for item in value]
    if type(value) is dict:
        return {name: _clone(item) for name, item in value.items()}
    return None


def _relative(current, previous):
    differences = _binary(
        current,
        previous,
        lambda a, b: (a.float() - b.float()).abs().mean()
        / b.float().abs().mean().clamp_min(1.0e-6),
    )
    values = [value for _, value in _leaves(differences)]
    if not values:
        raise ValueError("No residual tensors were found")
    return float(torch.stack(values).mean().item())


class StepResidualCache:
    def __init__(self):
        self.transformer = None
        self.blocks = None
        self.enabled = False
        self.end()

    def attach(self, transformer):
        if self.transformer is not None:
            raise RuntimeError("This cache is already attached")
        self.transformer = transformer
        _, self.blocks = find_block_list(transformer)
        original = transformer.forward

        @functools.wraps(original)
        def counted_forward(*args, **kwargs):
            if self.enabled:
                self.step += 1
            return original(*args, **kwargs)

        transformer.forward = counted_forward
        for index, block in enumerate(self.blocks):
            block._va_uncached_forward = block.forward
            signature = inspect.signature(block.forward)
            block.forward = self._make_forward(index, block, signature)
        return self

    def _make_forward(self, index, block, signature):
        @functools.wraps(block._va_uncached_forward)
        def forward(*args, **kwargs):
            if not self.enabled:
                return block._va_uncached_forward(*args, **kwargs)
            arguments = signature.bind_partial(*args, **kwargs).arguments
            if index and self.skip:
                state = _restore(self.layouts[index], arguments)
                if index == len(self.blocks) - 1:
                    return _binary(state, self.residual, torch.add)
                return state

            output = block._va_uncached_forward(*args, **kwargs)
            try:
                if index not in self.layouts:
                    self.layouts[index] = _layout(output, arguments)
                state = _restore(self.layouts[index], arguments)

                if index == 0:
                    current = _binary(output, state, torch.sub)
                    rel = float("inf")
                    if self.previous_first is not None:
                        rel = _relative(current, self.previous_first)
                    self.previous_first = _clone(current)
                    self.acc_rel += rel
                    self.skip = (
                        self.warmup <= self.step < self.num_steps - self.tail
                        and self.residual is not None
                        and len(self.layouts) == len(self.blocks)
                        and self.acc_rel < self.threshold
                    )
                    if not self.skip:
                        self.acc_rel = 0.0
                        self.boundary = _clone(output)

                elif index == len(self.blocks) - 1:
                    self.residual = _clone(
                        _binary(output, self.boundary, torch.sub)
                    )
                    self.boundary = None
            except (ValueError, KeyError, TypeError):
                self.enabled = False
                self.residual = None
                self.boundary = None
                self.previous_first = None
                self.skip = False
            return output

        return forward

    def begin(self, num_steps, threshold, warmup, tail):
        self.end()
        self.num_steps = int(num_steps)
        self.threshold = float(threshold)
        self.warmup = int(warmup)
        self.tail = int(tail)
        self.enabled = (
            threshold > 0.0
            and num_steps > 8
            and self.blocks is not None
            and len(self.blocks) >= 3
        )

    def end(self):
        self.enabled = False
        self.step = -1
        self.num_steps = 0
        self.threshold = 0.0
        self.warmup = 0
        self.tail = 0
        self.acc_rel = 0.0
        self.skip = False
        self.layouts = {}
        self.residual = None
        self.boundary = None
        self.previous_first = None
