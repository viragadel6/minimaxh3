"""Step-residual caching for denoising transformer blocks.

The cache skips the tail of the block stack on steps whose first-block activation is
"close enough" to the previous step, re-adding the residual computed at the last
full-compute step.

Correctness notes
-----------------
* The residual is measured against the *input of the first skipped block*, not the
  output of block 0.  Any transformation the outer loop performs between block 0 and
  block 1 is therefore part of the region that is skipped and re-applied through the
  residual, never applied twice.
* Only the container *shape* of the reference output is remembered (no live tensors
  are retained between steps).
* Container handling is generic (``namedtuple``, ``list``, ``Mapping``/``ModelOutput``)
  and every mismatch raises :class:`StateError`, which disables skipping for the run
  instead of silently corrupting activations.
* Steps are counted from the timestep argument when the transformer exposes one, so
  multi-pass CFG no longer counts one denoising step twice.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn

from va_kernels.fp8_linear import find_primary_block_list

logger = logging.getLogger(__name__)

_TIMESTEP_NAMES = frozenset(
    {"timestep", "timesteps", "t", "sigma", "sigmas", "step", "s_idx"}
)
_HIDDEN_NAMES = (
    "hidden_states",
    "video_hidden_states",
    "audio_hidden_states",
    "encoder_hidden_states",
    "x",
    "inputs_embeds",
    "latent",
    "latents",
)


class StateError(RuntimeError):
    """Raised when two cached states cannot be combined safely."""


# ---------------------------------------------------------------------------
# generic container helpers
# ---------------------------------------------------------------------------
def _is_namedtuple(value: Any) -> bool:
    return isinstance(value, tuple) and hasattr(value, "_fields")


def _rebuild_like(template: Any, items: Sequence[Any]):
    """Rebuild a container of the same kind as ``template`` from ``items``."""
    if _is_namedtuple(template):
        return type(template)(*items)
    if isinstance(template, tuple):
        return tuple(items)
    if isinstance(template, list):
        return list(items)
    if isinstance(template, Mapping):
        container_type = type(template)
        try:
            return container_type(items)
        except TypeError:
            rebuilt: dict[Any, Any] = {}
            rebuilt.update(items)
            return rebuilt
    raise StateError(f"unsupported container type {type(template).__name__}")


def _items(state: Any) -> list[Any]:
    if isinstance(state, Mapping):
        return list(state.items())
    if isinstance(state, (tuple, list)):
        return list(state)
    raise StateError(f"unsupported container type {type(state).__name__}")


def _keys(state: Any) -> list[Any]:
    if isinstance(state, Mapping):
        return list(state.keys())
    return []


def _clone_state(state: Any):
    if isinstance(state, torch.Tensor):
        return state.detach().clone()
    if isinstance(state, Mapping):
        return _rebuild_like(state, [(k, _clone_state(v)) for k, v in state.items()])
    if isinstance(state, (tuple, list)):
        return _rebuild_like(state, [_clone_state(v) for v in state])
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        clones = {
            field.name: _clone_state(getattr(state, field.name))
            for field in dataclasses.fields(state)
        }
        return dataclasses.replace(state, **clones)
    if state is None or isinstance(state, (int, float, bool, str)):
        return state
    raise StateError(f"cannot clone state of type {type(state).__name__}")


def _combine_state(a, b, op: str):
    """Element-wise ``a +/- b`` over an arbitrary (nested) container."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            raise StateError(
                f"cannot {op} tensors with shapes {tuple(a.shape)} and {tuple(b.shape)}"
            )
        b = b.to(dtype=a.dtype, device=a.device)
        return a + b if op == "add" else a - b
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if _keys(a) != _keys(b):
            raise StateError(f"cannot {op} mappings with different keys")
        return _rebuild_like(
            a, [(k, _combine_state(a[k], b[k], op)) for k in a.keys()]
        )
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            raise StateError(
                f"cannot {op} containers of length {len(a)} and {len(b)}"
            )
        return _rebuild_like(a, [_combine_state(x, y, op) for x, y in zip(a, b, strict=True)])
    if a is None and b is None:
        return None
    raise StateError(
        f"cannot {op} {type(a).__name__} and {type(b).__name__}"
    )


def _add_state(a, b):
    return _combine_state(a, b, "add")


def _sub_state(a, b):
    return _combine_state(a, b, "sub")


def _rel_diff(h1, h1_prev) -> float:
    """Relative L1 change between two (possibly nested) states.

    Raises :class:`StateError` for unsupported or mismatched states so the caller can
    fall back to full compute instead of treating them as identical.
    """
    if isinstance(h1, torch.Tensor) and isinstance(h1_prev, torch.Tensor):
        if h1.shape != h1_prev.shape:
            raise StateError(
                f"cannot compare tensors with shapes {tuple(h1.shape)} and "
                f"{tuple(h1_prev.shape)}"
            )
        if h1.numel() == 0:
            raise StateError("cannot compare empty tensors")
        current = h1.float()
        previous = h1_prev.to(device=h1.device, dtype=torch.float32)
        numerator = torch.mean(torch.abs(current - previous))
        denominator = torch.clamp(torch.mean(torch.abs(previous)), min=1e-6)
        rel = numerator / denominator
        if not bool(torch.isfinite(rel)):
            raise StateError("relative difference is not finite")
        return float(rel.item())  # single host sync per step
    if isinstance(h1, Mapping) and isinstance(h1_prev, Mapping):
        if _keys(h1) != _keys(h1_prev):
            raise StateError("cannot compare mappings with different keys")
        values = [
            _rel_diff(h1[k], h1_prev[k])
            for k in h1.keys()
            if isinstance(h1[k], torch.Tensor)
        ]
        return _mean_or_zero(values)
    if isinstance(h1, (tuple, list)) and isinstance(h1_prev, (tuple, list)):
        if len(h1) != len(h1_prev):
            raise StateError("cannot compare containers of different length")
        values = [
            _rel_diff(x, y)
            for x, y in zip(h1, h1_prev, strict=True)
            if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor)
        ]
        return _mean_or_zero(values)
    raise StateError(
        f"cannot compare {type(h1).__name__} with {type(h1_prev).__name__}"
    )


def _mean_or_zero(values: list[float]) -> float:
    if not values:
        raise StateError("no comparable tensors in state")
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# reference-output templates
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _StateTemplate:
    kind: str
    container_type: type | None = None
    slots: tuple[bool, ...] = ()
    keys: tuple[Any, ...] = ()


def _describe_state(state: Any) -> _StateTemplate:
    """Describe the *shape* of a block output (never retains tensors)."""
    if isinstance(state, torch.Tensor):
        return _StateTemplate("tensor", None, (True,))
    if isinstance(state, Mapping):
        return _StateTemplate(
            "mapping",
            type(state),
            tuple(isinstance(v, torch.Tensor) for v in state.values()),
            tuple(state.keys()),
        )
    if isinstance(state, tuple):
        return _StateTemplate(
            "tuple", type(state), tuple(isinstance(v, torch.Tensor) for v in state)
        )
    if isinstance(state, list):
        return _StateTemplate(
            "list", type(state), tuple(isinstance(v, torch.Tensor) for v in state)
        )
    return _StateTemplate("scalar")


def _signature_names(func) -> list[str]:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return []
    return [
        name
        for name, param in signature.parameters.items()
        if param.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]


def _collect_tensors(args: tuple, kwargs: dict, names: Sequence[str]) -> list[torch.Tensor]:
    """Collect tensors ordered by the callee's signature (kwargs first)."""
    ordered: list[torch.Tensor] = []
    seen: set[int] = set()

    def push(value):
        if isinstance(value, torch.Tensor) and id(value) not in seen:
            seen.add(id(value))
            ordered.append(value)

    for name in names:
        if name in kwargs:
            push(kwargs[name])
    for name in _HIDDEN_NAMES:
        if name in kwargs:
            push(kwargs[name])
    for value in args:
        push(value)
    for value in kwargs.values():
        push(value)
    return ordered


def _rebuild_from_template(
    template: _StateTemplate, tensors: Sequence[torch.Tensor]
):
    if template.kind == "mapping":
        rebuilt = {}
        cursor = 0
        for key, is_tensor in zip(template.keys, template.slots, strict=True):
            if is_tensor:
                if cursor >= len(tensors):
                    raise StateError("not enough tensor inputs to rebuild the mapping")
                rebuilt[key] = tensors[cursor]
                cursor += 1
            else:
                rebuilt[key] = None
        container = template.container_type or dict
        try:
            return container(rebuilt)
        except TypeError:  # pragma: no cover - exotic mapping types
            return rebuilt
    slots = template.slots or (True,)
    rebuilt_items = []
    cursor = 0
    for is_tensor in slots:
        if not is_tensor:
            rebuilt_items.append(None)
            continue
        if cursor >= len(tensors):
            raise StateError("not enough tensor inputs to rebuild the block output")
        rebuilt_items.append(tensors[cursor])
        cursor += 1
    if template.kind == "list":
        return list(rebuilt_items)
    if template.kind == "tuple" and template.container_type is not None:
        return template.container_type(*rebuilt_items)
    return tuple(rebuilt_items)


def _extract_input_hidden(
    args: tuple,
    kwargs: dict,
    reference_output: Any = None,
    names: Sequence[str] | None = None,
) -> Any:
    """Reconstruct the block input in the container shape of ``reference_output``.

    Tensors are ordered by the block's signature (not by positional order), so a
    positional conditioning tensor can never masquerade as the hidden state, and a
    dual-stream block returning ``(encoder_hidden_states, hidden_states)`` keeps its
    ordering.
    """
    tensors = _collect_tensors(args, kwargs, names or ())
    if not tensors:
        raise StateError("no tensor input found for the skipped block")
    if reference_output is None:
        return tensors[0]
    if isinstance(reference_output, _StateTemplate):
        template = reference_output
    else:
        template = _describe_state(reference_output)
    if template.kind == "tensor" or template.kind == "scalar":
        return tensors[0]
    return _rebuild_from_template(template, tensors)


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
class StepResidualCache:
    """Skip the tail of the block stack when consecutive steps are similar."""

    DEFAULT_WARMUP = 3
    DEFAULT_TAIL = 2
    DEFAULT_MIN_SKIPPABLE = 1

    def __init__(self) -> None:
        self.transformer: nn.Module | None = None
        self._transformer_ref: weakref.ref | None = None
        self.blocks: nn.ModuleList | None = None
        self.list_name: str = ""
        self.last_idx: int = -1
        self.active: bool = False
        self.attached: bool = False
        self.disabled_reason: str | None = None
        #: Set when the module structure itself is unsuitable; unlike
        #: ``disabled_reason`` this is never cleared by a new run.
        self.structural_error: str | None = None
        self.num_steps: int = 0
        self.threshold: float = 0.0
        self.warmup: int = self.DEFAULT_WARMUP
        self.tail: int = self.DEFAULT_TAIL
        self.min_skippable: int = self.DEFAULT_MIN_SKIPPABLE
        self.step: int = -1
        self.acc_rel: float = 0.0
        self.skip_step: bool = False
        self.h1_prev = None
        self.residual = None
        self._region_input = None
        self._template: _StateTemplate | None = None
        self._last_timestep: Any = None
        self._originals: dict[int, tuple[nn.Module, Any]] = {}
        self._transformer_forward_original = None
        self._skips: int = 0

    # -- attach / detach -------------------------------------------------
    def attach(self, transformer: nn.Module) -> "StepResidualCache":
        if self.attached:
            self.detach()
        name, blocks = find_primary_block_list(transformer)
        if blocks is None or len(blocks) < 2:
            raise ValueError("StepResidualCache needs at least two transformer blocks")
        self.transformer = transformer
        self._transformer_ref = weakref.ref(transformer)
        self.list_name = name
        self.blocks = blocks
        self.last_idx = len(blocks) - 1
        self._originals = {}
        seen: dict[int, int] = {}
        for idx, block in enumerate(blocks):
            if id(block) in seen:
                self._disable_structural(
                    f"block {idx} is the same module as block {seen[id(block)]}; "
                    "step caching requires unique block modules"
                )
                continue
            seen[id(block)] = idx
            self._originals[idx] = (block, block.forward)
            block.forward = self._make_block_forward(idx, block, block.forward)
        self._transformer_forward_original = transformer.forward
        transformer.forward = self._make_transformer_forward()
        self.attached = True
        return self

    def detach(self) -> None:
        """Restore every wrapped ``forward`` (safe to call repeatedly)."""
        for idx, (block, forward) in self._originals.items():
            if self.blocks is not None and idx < len(self.blocks) and self.blocks[idx] is block:
                block.forward = forward
        self._originals = {}
        transformer = self._transformer_ref() if self._transformer_ref else None
        if transformer is not None and self._transformer_forward_original is not None:
            transformer.forward = self._transformer_forward_original
        self._transformer_forward_original = None
        self.attached = False
        self.end()
        self.blocks = None
        self.transformer = None
        self._transformer_ref = None

    # -- wrappers --------------------------------------------------------
    def _make_transformer_forward(self):
        cache = self
        original = self._transformer_forward_original
        names = _signature_names(original)
        timestep_index = _timestep_position(original)
        timestep_name = (
            names[timestep_index]
            if timestep_index is not None and timestep_index < len(names)
            else None
        )

        def _transformer_forward(*args, **kwargs):
            if cache.active:
                cache._advance_step(args, kwargs, timestep_index, timestep_name)
            return original(*args, **kwargs)

        return _transformer_forward

    def _advance_step(
        self,
        args: tuple,
        kwargs: dict,
        timestep_index: int | None,
        timestep_name: str | None,
    ) -> None:
        value = None
        found = False
        if timestep_name is not None and timestep_name in kwargs:
            value, found = kwargs[timestep_name], True
        elif timestep_index is not None and timestep_index < len(args):
            value, found = args[timestep_index], True
        if not found:
            for key in _TIMESTEP_NAMES:
                if key in kwargs:
                    value, found = kwargs[key], True
                    break
        if found:
            normalized = _scalarise(value)
            if normalized is not None:
                if normalized == self._last_timestep:
                    # Same timestep: another CFG pass, not a new denoising step.
                    return
                self._last_timestep = normalized
        self.step += 1
        self.skip_step = False

    def _make_block_forward(self, idx: int, block: nn.Module, raw_forward):
        cache = self
        names = _signature_names(raw_forward)

        def _block_forward(*args, **kwargs):
            if not cache.active or cache.blocks is None:
                return raw_forward(*args, **kwargs)
            if idx >= len(cache.blocks) or cache.blocks[idx] is not block:
                cache._disable("block list changed after attach(); skipping disabled")
                return raw_forward(*args, **kwargs)

            if idx == 0:
                output = raw_forward(*args, **kwargs)
                cache._on_first_block(output)
                return output

            if cache.skip_step and cache.residual is not None:
                try:
                    hidden = _extract_input_hidden(args, kwargs, cache._template, names)
                except StateError as exc:
                    cache._disable(f"cannot rebuild skipped block input: {exc}")
                    return raw_forward(*args, **kwargs)
                if idx == cache.last_idx:
                    try:
                        return _add_state(hidden, cache.residual)
                    except StateError as exc:
                        cache._disable(f"cannot apply residual: {exc}")
                        return raw_forward(*args, **kwargs)
                return hidden

            if idx == 1:
                try:
                    cache._region_input = _clone_state(
                        _extract_input_hidden(args, kwargs, cache._template, names)
                    )
                except StateError:
                    cache._region_input = None
            output = raw_forward(*args, **kwargs)
            if idx == cache.last_idx and cache._region_input is not None:
                try:
                    cache.residual = _sub_state(output, cache._region_input)
                except StateError as exc:
                    cache.residual = None
                    cache._disable(f"cannot compute residual: {exc}")
                finally:
                    cache._region_input = None
            return output

        return _block_forward

    # -- per-step bookkeeping -------------------------------------------
    def _on_first_block(self, output) -> None:
        self._template = _describe_state(output)
        try:
            cloned = _clone_state(output)
        except StateError as exc:
            self._disable(f"cannot clone block output: {exc}")
            return

        in_window = self.warmup <= self.step < (self.num_steps - self.tail)
        if not in_window or self.h1_prev is None or self.residual is None:
            self.skip_step = False
            self.acc_rel = 0.0
            self.h1_prev = cloned
            return
        try:
            self.acc_rel += _rel_diff(output, self.h1_prev)
        except StateError as exc:
            self.h1_prev = cloned
            self.skip_step = False
            self.acc_rel = 0.0
            logger.warning("step cache metric failed (%s); running full compute", exc)
            return
        self.h1_prev = cloned
        self.skip_step = self.acc_rel < self.threshold
        if self.skip_step:
            self._skips += 1
        else:
            self.acc_rel = 0.0

    def _disable(self, reason: str) -> None:
        """Stop caching for the current run (a later run gets another chance)."""
        if self.active:
            logger.warning("step cache disabled: %s", reason)
        self.active = False
        self.skip_step = False
        self.disabled_reason = reason

    def _disable_structural(self, reason: str) -> None:
        """Stop caching permanently: the module layout cannot support it."""
        logger.warning("step cache disabled for this transformer: %s", reason)
        self.structural_error = reason
        self.disabled_reason = reason
        self.active = False
        self.skip_step = False

    # -- lifecycle -------------------------------------------------------
    def begin(
        self,
        num_steps: int,
        threshold: float,
        warmup: int | None = None,
        tail: int | None = None,
        min_skippable: int | None = None,
    ) -> None:
        self.end()
        # A previous run's runtime failure must not poison this one.
        self.disabled_reason = None
        if not self.attached or self.blocks is None:
            logger.warning("StepResidualCache.begin() called before attach(); caching off")
            return
        self.num_steps = max(0, int(num_steps))
        self.threshold = float(threshold)
        self.warmup = self.DEFAULT_WARMUP if warmup is None else int(warmup)
        self.tail = self.DEFAULT_TAIL if tail is None else int(tail)
        self.min_skippable = (
            self.DEFAULT_MIN_SKIPPABLE if min_skippable is None else int(min_skippable)
        )
        skippable = self.num_steps - self.warmup - self.tail
        self.active = (
            self.threshold > 0.0
            and len(self.blocks) >= 2
            and skippable >= self.min_skippable
            and self.structural_error is None
        )
        if not self.active:
            return
        self.step = -1
        self._last_timestep = None
        self.acc_rel = 0.0
        self.skip_step = False
        self.h1_prev = None
        self.residual = None
        self._region_input = None
        self._template = None
        self._skips = 0

    def end(self) -> None:
        self.active = False
        if self.structural_error is None:
            self.disabled_reason = None
        self.step = -1
        self.num_steps = 0
        self.threshold = 0.0
        self.acc_rel = 0.0
        self.skip_step = False
        self.h1_prev = None
        self.residual = None
        self._region_input = None
        self._template = None
        self._last_timestep = None

    def __del__(self):  # pragma: no cover - defensive cleanup
        try:
            self.detach()
        except Exception:
            pass


def _scalarise(value) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return value.detach().item()
    if isinstance(value, (int, float, str, bool)):
        return value
    return None


def _timestep_position(forward) -> int | None:
    for position, name in enumerate(_signature_names(forward)):
        if name.lower() in _TIMESTEP_NAMES:
            return position
    return None


__all__ = [
    "StateError",
    "StepResidualCache",
    "_add_state",
    "_clone_state",
    "_extract_input_hidden",
    "_rel_diff",
    "_sub_state",
]
