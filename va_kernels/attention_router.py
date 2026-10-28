"""Opt-in routing of ``F.scaled_dot_product_attention`` to the FP8 kernel.

Guarantees
----------
* :func:`install` / :func:`uninstall` are idempotent, reload safe and only touch
  ``torch.nn.functional.scaled_dot_product_attention`` (never the C builtin, which
  would break pickling of references to it).
* Every fallback goes through the *native* implementation captured on first
  install, so a module reload can never create a self-recursive wrapper.
* Routing is skipped whenever gradients, autocast, GQA, dropout, causal masking or
  unsupported dtypes/shapes are involved, and whenever the current thread/task is
  already inside the router (re-entrancy guard).
* ``denoiser_scope`` uses a :class:`contextvars.ContextVar`, so it is async-task
  local, survives thread hops used by checkpointing and always restores the exact
  previous depth.
"""

from __future__ import annotations

import contextlib
import contextvars
import inspect
import logging
import math
import operator
import threading
import types
import weakref
from dataclasses import dataclass

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:  # package-relative first (works when imported as ``va_kernels.attention_router``)
    from .fp8_attention import _prepare_key_side, fp8_attention, is_supported, requires_grad
except ImportError:  # pragma: no cover - flat sys.path layouts
    from va_kernels.fp8_attention import (  # type: ignore[no-redef]
        _prepare_key_side,
        fp8_attention,
        is_supported,
        requires_grad,
    )


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RouterConfig:
    enabled: bool = True
    repair_tau: float = -1.0
    min_seq: int = 4096


_config = RouterConfig()
_config_lock = threading.Lock()


def configure(
    enabled: bool = True,
    repair_tau: float = -1.0,
    min_seq: int = 4096,
) -> RouterConfig:
    """Atomically replace the router configuration (validated, immutable).

    Returns the *new* snapshot; read :func:`get_config` first if you need to restore
    the previous one.
    """
    global _config
    if not isinstance(enabled, bool):
        raise TypeError(
            f"enabled must be a bool, got {type(enabled).__name__}; "
            "note bool('False') is True"
        )
    try:
        tau = float(repair_tau)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"repair_tau must be a number, got {repair_tau!r}") from exc
    if not math.isfinite(tau):
        raise ValueError("repair_tau must be finite")
    if isinstance(min_seq, bool):
        raise TypeError("min_seq must be an integer")
    try:
        # ``operator.index`` rejects floats (including 1.5) instead of truncating them.
        min_seq_v = operator.index(min_seq)
    except TypeError as exc:
        raise ValueError(f"min_seq must be an integer, got {min_seq!r}") from exc
    if min_seq_v < 1:
        raise ValueError("min_seq must be >= 1")
    new_config = RouterConfig(enabled=enabled, repair_tau=tau, min_seq=min_seq_v)
    with _config_lock:
        _config = new_config
    return new_config


def get_config() -> RouterConfig:
    """Return an immutable configuration snapshot (safe to read concurrently)."""
    return _config


def __getattr__(name: str):  # PEP 562: keep ``attention_router.enabled`` working
    if name in ("enabled", "repair_tau", "min_seq"):
        return getattr(_config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# denoiser scope (task local, restores the previous depth exactly)
# ---------------------------------------------------------------------------
_depth_var: contextvars.ContextVar[int] = contextvars.ContextVar(
    "va_denoiser_depth", default=0
)
_IN_ROUTER: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "va_in_router", default=False
)


def _get_depth() -> int:
    return _depth_var.get()


@contextlib.contextmanager
def denoiser_scope(transformer=None):
    token = _depth_var.set(_depth_var.get() + 1)
    try:
        yield transformer
    finally:
        # Resetting the token restores the *previous* depth, even if an inner scope
        # exited unbalanced (a plain decrement would corrupt outer scopes).
        _depth_var.reset(token)


# ---------------------------------------------------------------------------
# native SDPA resolution (reload safe)
# ---------------------------------------------------------------------------
_installed_lock = threading.RLock()
_native_cache: object | None = None


def _looks_like_router(fn) -> bool:
    return bool(getattr(fn, "_va_router", False))


def _native_sdpa():
    """Return the original SDPA implementation, never a router wrapper."""
    global _native_cache
    current = F.scaled_dot_product_attention
    if _looks_like_router(current):
        native = getattr(current, "_va_native", None)
        if native is None:  # pragma: no cover - only if someone strips the attribute
            raise RuntimeError(
                "routed_sdpa is installed but the native implementation was lost; "
                "restart the process"
            )
        _native_cache = native
        return native
    if _native_cache is None or current is not _native_cache:
        if current is routed_sdpa:  # pragma: no cover - defensive
            raise RuntimeError("refusing to use routed_sdpa as the native implementation")
        _native_cache = current
    return _native_cache


_SDPA_GQA: bool | None = None


def _sdpa_supports_gqa() -> bool:
    global _SDPA_GQA
    if _SDPA_GQA is None:
        try:
            signature = inspect.signature(_native_sdpa())
            _SDPA_GQA = "enable_gqa" in signature.parameters
        except (TypeError, ValueError):  # builtins without introspection
            parts = torch.__version__.split("+")[0].split(".")
            try:
                version = tuple(int(p) for p in parts[:2])
            except ValueError:  # pragma: no cover
                version = (0, 0)
            _SDPA_GQA = version >= (2, 5)
    return _SDPA_GQA


def _call_native(
    query,
    key,
    value,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    enable_gqa,
):
    kwargs = {
        "attn_mask": attn_mask,
        "dropout_p": dropout_p,
        "is_causal": is_causal,
    }
    if scale is not None:
        kwargs["scale"] = scale
    if enable_gqa and _sdpa_supports_gqa():
        kwargs["enable_gqa"] = True
    return _native_sdpa()(query, key, value, **kwargs)


def _autocast_dtype() -> torch.dtype | None:
    try:
        if torch.is_autocast_enabled("cuda"):
            return torch.get_autocast_dtype("cuda")
        return None
    except (TypeError, AttributeError):  # pragma: no cover - older torch
        try:
            if torch.is_autocast_enabled():
                return torch.get_autocast_gpu_dtype()
        except (TypeError, AttributeError):
            return None
        return None


# ---------------------------------------------------------------------------
# mask classification
# ---------------------------------------------------------------------------
def _is_bool_mask(tensor: torch.Tensor) -> bool:
    return tensor.dtype == torch.bool


def _extract_per_key_mask(
    attn_mask: torch.Tensor | None,
    B: int,
    H: int,
    Sq: int,
    Sk: int,
    device: torch.device,
    query_dtype: torch.dtype = torch.bfloat16,
) -> tuple[bool, torch.Tensor | None, torch.Tensor | None]:
    """Classify ``attn_mask`` as an unsupported mask, per-key mask or per-key bias.

    Returns ``(ok, bool_mask, float_bias)``.  Rank-1 masks, trailing broadcast
    dimensions, >4D masks and batch-broadcast masks are all accepted and returned
    as stride-0 views (no ``B``-fold copy is materialised).
    """
    if attn_mask is None:
        return True, None, None
    if not isinstance(attn_mask, torch.Tensor):
        return False, None, None
    if attn_mask.device != device:
        return False, None, None
    if attn_mask.layout != torch.strided:
        return False, None, None
    if _is_bool_mask(attn_mask):
        try:
            prepared = _prepare_key_side(
                attn_mask, "attn_mask", B, Sk, device, query_dtype, as_bool=True
            )
        except ValueError:
            return False, None, None
        return True, prepared[0], None
    if attn_mask.is_floating_point():
        # Native SDPA requires floating masks to match the query dtype.
        if attn_mask.dtype != query_dtype:
            return False, None, None
        try:
            prepared = _prepare_key_side(
                attn_mask, "attn_mask", B, Sk, device, query_dtype, as_bool=False
            )
        except ValueError:
            return False, None, None
        return True, None, prepared[0]
    return False, None, None


# ---------------------------------------------------------------------------
# router
# ---------------------------------------------------------------------------
def routed_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Drop-in replacement for ``F.scaled_dot_product_attention``.

    ``scale``/``enable_gqa`` are accepted positionally *and* by keyword and are
    always forwarded by keyword, so the fast path and the fallback agree on the
    calling convention.
    """
    if kwargs or _should_fallback(query, key, value, dropout_p, is_causal, enable_gqa):
        return _call_native(
            query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
        )

    config = _config
    B, H, Sq, D = (int(x) for x in query.shape)
    Sk = int(key.shape[2])
    token = _IN_ROUTER.set(True)
    try:
        ok, key_mask, key_bias = _extract_per_key_mask(
            attn_mask, B, H, Sq, Sk, query.device, query.dtype
        )
        if ok:
            try:
                return fp8_attention(
                    query,
                    key,
                    value,
                    scale=scale,
                    key_mask=key_mask,
                    key_bias=key_bias,
                    repair_tau=config.repair_tau,
                )
            except (ValueError, RuntimeError) as exc:
                # Unsupported mask / kernel limitation: fall through to native, but
                # keep the reason visible for debugging.
                logger.debug("fp8 attention fell back to native: %s", exc)
    finally:
        _IN_ROUTER.reset(token)
    return _call_native(
        query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
    )


def _should_fallback(
    query,
    key,
    value,
    dropout_p: float,
    is_causal: bool,
    enable_gqa: bool,
) -> bool:
    """Return ``True`` when the native implementation must be used."""
    config = _config
    if not config.enabled:
        return True
    if _IN_ROUTER.get():
        # Re-entrancy guard: fp8_attention must never recurse through the router.
        return True
    if _get_depth() <= 0:
        return True
    if torch.is_grad_enabled() and (
        isinstance(query, torch.Tensor)
        and isinstance(key, torch.Tensor)
        and isinstance(value, torch.Tensor)
        and requires_grad(query, key, value)
    ):
        return True
    if dropout_p != 0.0 or is_causal or enable_gqa:
        return True
    for tensor in (query, key, value):
        if not isinstance(tensor, torch.Tensor):
            return True
        if tensor.ndim != 4:  # only (B, H, L, E) has a sequence axis at index 2
            return True
    if query.shape[2] < config.min_seq:
        return True
    # Long queries with a short key/value cache are cross-attention: not routed.
    if key.shape[2] < config.min_seq or value.shape[2] < config.min_seq:
        return True
    if key.shape[2] != value.shape[2]:
        return True
    autocast_dtype = _autocast_dtype()
    if autocast_dtype is not None:
        # Execute the autocast policy: only route when the inputs already are the
        # autocast dtype, otherwise the Triton kernel would bypass autocast.
        if autocast_dtype not in (torch.bfloat16, torch.float16):
            return True
        if query.dtype != autocast_dtype:
            return True
    return not is_supported(query, key, value)


def install() -> None:
    """Install the router (idempotent, reload safe)."""
    with _installed_lock:
        native = _native_sdpa()
        if native is routed_sdpa:  # pragma: no cover - defensive
            raise RuntimeError("refusing to install routed_sdpa as its own fallback")
        if getattr(F.scaled_dot_product_attention, "_va_router", False):
            # Already installed (possibly by an older module instance): refresh the
            # stored native reference and return.
            F.scaled_dot_product_attention._va_native = native
            return
        routed_sdpa._va_router = True  # type: ignore[attr-defined]
        routed_sdpa._va_native = native  # type: ignore[attr-defined]
        F.scaled_dot_product_attention = routed_sdpa


def uninstall() -> None:
    """Restore the native ``scaled_dot_product_attention``."""
    with _installed_lock:
        current = F.scaled_dot_product_attention
        if not getattr(current, "_va_router", False):
            return
        F.scaled_dot_product_attention = _native_sdpa()


def is_installed() -> bool:
    return bool(getattr(F.scaled_dot_product_attention, "_va_router", False))


# ---------------------------------------------------------------------------
# denoiser scope instrumentation
# ---------------------------------------------------------------------------
_SCOPED_SUBCLASSES: dict[type, type] = {}
_instrument_lock = threading.RLock()


def _scoped_generator(gen, module):
    with denoiser_scope(module):
        yield from gen


def _make_scoped_forward(base_cls: type):
    """Build a ``forward`` override for a *class* (never captures an instance)."""

    def forward(self, *args, **kwargs):
        with denoiser_scope(self):
            result = base_cls.forward(self, *args, **kwargs)
        if isinstance(result, types.GeneratorType):
            # A generator body runs after forward() returns: keep the scope open.
            return _scoped_generator(result, self)
        return result

    forward.__name__ = "forward"
    forward.__doc__ = getattr(base_cls.forward, "__doc__", None)
    return forward


class _WeakForwardWrapper:
    """Fallback wrapper used when ``__class__`` cannot be reassigned."""

    __slots__ = ("_ref", "_func")

    def __init__(self, module, func):
        self._ref = weakref.ref(module)
        self._func = func

    def __call__(self, *args, **kwargs):
        module = self._ref()
        if module is None:  # pragma: no cover - module collected
            raise RuntimeError("instrumented transformer was garbage collected")
        with denoiser_scope(module):
            return self._func(*args, **kwargs)


def instrument_transformer(transformer):
    """Make every call to ``transformer.forward`` run inside a denoiser scope.

    Implemented by swapping the instance's class for a generated subclass: there is
    no closure over the instance (no reference cycle), and ``copy.deepcopy`` /
    ``pickle`` keep working because the generated class is registered in this
    module's globals.
    """
    with _instrument_lock:
        if getattr(transformer, "_denoiser_scope_instrumented", False):
            return transformer
        base_cls = type(transformer)
        if getattr(base_cls, "_va_scoped_forward", False):
            transformer._denoiser_scope_instrumented = True
            return transformer
        scoped_cls = _SCOPED_SUBCLASSES.get(base_cls)
        if scoped_cls is None:
            name = f"{base_cls.__name__}VaDenoiserScoped"
            scoped_cls = type(
                name,
                (base_cls,),
                {
                    "forward": _make_scoped_forward(base_cls),
                    "__module__": __name__,
                    "__qualname__": name,
                    "__doc__": base_cls.__doc__,
                    "_va_scoped_forward": True,
                },
            )
            globals()[name] = scoped_cls  # makes the class picklable
            _SCOPED_SUBCLASSES[base_cls] = scoped_cls
        try:
            transformer.__class__ = scoped_cls
        except TypeError:  # pragma: no cover - extension types
            original = transformer.forward
            transformer.forward = _WeakForwardWrapper(transformer, original)
        transformer._denoiser_scope_instrumented = True
        return transformer


__all__ = [
    "RouterConfig",
    "configure",
    "denoiser_scope",
    "get_config",
    "install",
    "instrument_transformer",
    "is_installed",
    "routed_sdpa",
    "uninstall",
]
