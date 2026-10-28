"""Minimal ``torch``/``triton`` stand-ins used to import-test the GPU modules.

The development sandbox has no torch installation, so the kernel and runtime modules
cannot be executed here.  These stubs are intentionally *shallow*: they exist so
``import`` succeeds and the pure-Python logic (router configuration, scope handling,
budget bookkeeping) can be exercised.  They are never used on a real deployment and
nothing in the package imports them.
"""

from __future__ import annotations

import contextlib
import importlib.machinery
import sys
import types
from typing import Any


def _mark_stub(module: types.ModuleType, name: str) -> None:
    """Give the stub a spec so ``importlib.util.find_spec`` keeps working."""
    if getattr(module, "__spec__", None) is None:
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    module.__loader__ = None


class _DType:
    def __init__(self, name: str, itemsize: int = 2, floating: bool = True) -> None:
        self.name = name
        self.itemsize = itemsize
        self.is_floating_point = floating
        self.is_complex = False

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"torch.{self.name}"


class _Storage:
    def __init__(self, nbytes: int = 0) -> None:
        self._nbytes = nbytes

    def nbytes(self) -> int:
        return self._nbytes

    def data_ptr(self) -> int:
        return id(self)


class _Stream:
    def synchronize(self) -> None:
        return None


class _Generator:
    def __init__(self, device=None) -> None:
        self.device = device

    def manual_seed(self, seed: int):  # noqa: D401 - stub
        self.seed = seed
        return self


class Tensor:
    """Very small tensor stand-in: only shapes, dtypes and a few flags."""

    def __init__(self, shape=(1,), dtype=None, device="cpu", requires_grad=False) -> None:
        self.shape = tuple(shape)
        module = sys.modules.get("torch")
        self.dtype = dtype if dtype is not None else getattr(module, "float32", None)
        self.device = _Device(device)
        self.requires_grad = requires_grad
        self.layout = "strided"
        self.is_sparse = False

    # -- introspection ------------------------------------------------
    @property
    def ndim(self) -> int:
        return len(self.shape)

    def dim(self) -> int:
        return len(self.shape)

    def numel(self) -> int:
        total = 1
        for size in self.shape:
            total *= size
        return total

    def stride(self, index: int | None = None):
        if index is None:
            return tuple(1 for _ in self.shape)
        return 1

    def is_contiguous(self) -> bool:
        return True

    @property
    def is_cuda(self) -> bool:
        return str(self.device).startswith("cuda")

    def is_floating_point(self) -> bool:
        return bool(getattr(self.dtype, "is_floating_point", False))

    def is_pinned(self) -> bool:
        return False

    def untyped_storage(self) -> _Storage:
        return _Storage(self.numel() * getattr(self.dtype, "itemsize", 2))

    def item(self) -> float:
        return 0.0

    def tolist(self):
        return []

    # -- transforms ---------------------------------------------------
    def to(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("dtype")  # noqa: F841 - stub
        clone = Tensor(self.shape, self.dtype, str(self.device), self.requires_grad)
        if isinstance(target, _DType):
            clone.dtype = target
        elif isinstance(target, str):
            clone.device = _Device(target)
        elif isinstance(target, _Device):
            clone.device = target
        return clone

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")

    def float(self):
        return self.to(sys.modules["torch"].float32)

    def detach(self):
        return self

    def clone(self):
        return Tensor(self.shape, self.dtype, str(self.device), self.requires_grad)

    def contiguous(self):
        return self

    def expand(self, *shape):
        return Tensor(shape, self.dtype, str(self.device))

    def permute(self, *order):
        return Tensor(tuple(self.shape[i] for i in order), self.dtype, str(self.device))

    def view(self, *shape):
        return Tensor(shape, self.dtype, str(self.device))

    def reshape(self, *shape):
        return Tensor(shape, self.dtype, str(self.device))

    def t(self):
        return self.permute(*reversed(range(self.ndim)))

    def __getitem__(self, item):
        return self

    def __setitem__(self, item, value):
        return None

    def __len__(self) -> int:
        return self.shape[0] if self.shape else 0

    def numpy(self):  # pragma: no cover - only used by real torch
        raise RuntimeError("stub tensors have no numpy backing")

    # -- arithmetic (no-op) -------------------------------------------
    def __mul__(self, other):
        return self

    __rmul__ = __mul__
    __add__ = __mul__
    __sub__ = __mul__
    __truediv__ = __mul__


class _Device:
    def __init__(self, name: Any = "cpu") -> None:
        if isinstance(name, _Device):
            self.type = name.type
            self.index = name.index
            return
        text = str(name)
        if ":" in text:
            self.type, _, index = text.partition(":")
            self.index = int(index) if index.isdigit() else 0
        else:
            self.type = text
            self.index = 0 if text == "cuda" else None

    def __str__(self) -> str:
        if self.index is None:
            return self.type
        return f"{self.type}:{self.index}"

    def __repr__(self) -> str:  # pragma: no cover
        return f"device(type={self.type!r})"

    def __eq__(self, other) -> bool:
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))


class _Parameter(Tensor):
    def __init__(self, data: Tensor | None = None, requires_grad: bool = True) -> None:
        super().__init__(
            data.shape if data is not None else (1,),
            data.dtype if data is not None else None,
            str(data.device) if data is not None else "cpu",
            requires_grad,
        )
        self.data = data if data is not None else self


class Module:
    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    # -- registration -------------------------------------------------
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, _Parameter):
            self._parameters[name] = value

    def add_module(self, name: str, module) -> None:
        self._modules[name] = module
        object.__setattr__(self, name, module)

    def register_buffer(self, name: str, tensor, persistent: bool = True) -> None:
        self._buffers[name] = tensor
        object.__setattr__(self, name, tensor)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        for name, param in self._parameters.items():
            yield (f"{prefix}.{name}" if prefix else name), param
        if recurse:
            for mod_name, module in self._modules.items():
                sub_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
                yield from module.named_parameters(sub_prefix, recurse)

    def named_modules(self, memo=None, prefix: str = ""):
        yield prefix, self
        for name, module in self._modules.items():
            sub_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_modules(memo, sub_prefix)

    def parameters(self, recurse: bool = True):
        for _, param in self.named_parameters("", recurse):
            yield param

    def buffers(self, recurse: bool = True):
        for _, tensor in self._buffers.items():
            yield tensor
        if recurse:
            for module in self._modules.values():
                yield from module.buffers(recurse)

    def modules(self):
        for _, module in self.named_modules():
            yield module

    def children(self):
        return iter(list(self._modules.values()))

    # -- behaviour ----------------------------------------------------
    def forward(self, *args, **kwargs):  # pragma: no cover - overridden
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def eval(self):
        object.__setattr__(self, "training", False)
        return self

    def train(self, mode: bool = True):
        object.__setattr__(self, "training", mode)
        return self

    def requires_grad_(self, requires_grad: bool = True):
        return self

    def to(self, *args, **kwargs):
        return self

    def state_dict(self, *args, **kwargs):
        return {}

    def load_state_dict(self, state, strict: bool = True):
        return None


class Linear(Module):
    def __init__(self, in_features: int = 4, out_features: int = 4, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = _Parameter(Tensor((out_features, in_features)))
        self.bias = _Parameter(Tensor((out_features,))) if bias else None

    def forward(self, x, *args, **kwargs):
        return x


class ModuleList(Module):
    def __init__(self, modules=None) -> None:
        super().__init__()
        for module in modules or []:
            self.append(module)

    def append(self, module) -> "ModuleList":
        self.add_module(str(len(self._modules)), module)
        return self

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(list(self._modules.values()))

    def __getitem__(self, index):
        return list(self._modules.values())[index]

    def __setitem__(self, index, module) -> None:
        self.add_module(str(index), module)

    def clear(self) -> None:
        self._modules.clear()


class ModuleDict(Module):
    def __init__(self, modules=None) -> None:
        super().__init__()
        for key, value in (modules or {}).items():
            self.add_module(key, value)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self):
        return iter(list(self._modules.keys()))

    def __getitem__(self, key):
        return self._modules[key]

    def items(self):
        return self._modules.items()


def _noop_decorator(fn=None, **kwargs):
    if fn is None:
        return lambda inner: inner
    return fn


@contextlib.contextmanager
def _context_manager(*args, **kwargs):
    yield


def install() -> None:
    """Register the stub modules in ``sys.modules`` (idempotent)."""
    if "torch" in sys.modules and getattr(sys.modules["torch"], "_va_stub", False):
        return

    torch = types.ModuleType("torch")
    torch._va_stub = True
    torch.Tensor = Tensor
    torch.device = _Device
    torch.strided = "strided"
    torch.sparse_coo = "sparse_coo"
    torch.Generator = _Generator
    torch.float32 = _DType("float32", 4)
    torch.float16 = _DType("float16", 2)
    torch.bfloat16 = _DType("bfloat16", 2)
    torch.float64 = _DType("float64", 8)
    torch.float8_e4m3fn = _DType("float8_e4m3fn", 1)
    torch.uint8 = _DType("uint8", 1, floating=False)
    torch.int8 = _DType("int8", 1, floating=False)
    torch.int16 = _DType("int16", 2, floating=False)
    torch.int32 = _DType("int32", 4, floating=False)
    torch.int64 = _DType("int64", 8, floating=False)
    torch.bool = _DType("bool", 1, floating=False)
    torch.long = torch.int64
    torch.promote_types = lambda a, b: a
    torch.no_grad = _noop_decorator
    torch.inference_mode = _noop_decorator
    torch.is_grad_enabled = lambda: False
    torch.is_autocast_enabled = lambda *a, **k: False
    torch.get_autocast_dtype = lambda *a, **k: torch.bfloat16
    torch.compile = _noop_decorator
    def _shape_from_args(*args):
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return tuple(args[0])
        if not args:
            return (1,)
        return tuple(args)

    torch.zeros = lambda *a, **k: Tensor(_shape_from_args(*a), k.get("dtype"))
    torch.ones = lambda *a, **k: Tensor(_shape_from_args(*a), k.get("dtype"))
    torch.empty = lambda *a, **k: Tensor(_shape_from_args(*a), k.get("dtype"))
    torch.randn = lambda *a, **k: Tensor(_shape_from_args(*a), k.get("dtype"))
    torch.tensor = lambda data, **k: Tensor((1,), k.get("dtype"))
    torch.frombuffer = lambda *a, **k: Tensor((1,), k.get("dtype"))
    torch.cat = lambda tensors, **k: Tensor((1,))
    torch.stack = lambda tensors, **k: Tensor((1,))
    torch.nn = types.ModuleType("torch.nn")
    torch.nn.Module = Module
    torch.nn.Linear = Linear
    torch.nn.Parameter = _Parameter
    torch.nn.ModuleList = ModuleList
    torch.nn.ModuleDict = ModuleDict
    torch.nn.Identity = type("Identity", (Module,), {"forward": lambda self, x: x})
    functional = types.ModuleType("torch.nn.functional")
    functional.scaled_dot_product_attention = lambda q, k, v, **kwargs: v
    torch.nn.functional = functional
    torch.nn.init = types.ModuleType("torch.nn.init")

    cuda = types.ModuleType("torch.cuda")
    cuda.is_available = lambda: False
    cuda.device_count = lambda: 0
    cuda.current_device = lambda: 0
    cuda.get_device_capability = lambda *a, **k: (8, 0)
    cuda.get_device_name = lambda *a, **k: "stub"
    cuda.mem_get_info = lambda *a, **k: (0, 0)
    cuda.empty_cache = lambda: None
    cuda.synchronize = lambda: None
    cuda.current_stream = lambda *a, **k: _Stream()
    cuda.device = _context_manager
    torch.cuda = cuda

    amp = types.ModuleType("torch.amp")
    amp.autocast = _context_manager
    torch.amp = amp

    _dynamo = types.ModuleType("torch._dynamo")
    _dynamo.config = types.SimpleNamespace(cache_size_limit=8)
    torch._dynamo = _dynamo

    torch.version = types.SimpleNamespace(hip=None, cuda="13.0")
    torch.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False))
    )
    torch._scaled_mm = None
    torch._C = types.SimpleNamespace(_nn=types.SimpleNamespace())
    _mark_stub(torch, "torch")
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch.nn
    sys.modules["torch.nn.functional"] = functional
    sys.modules["torch.cuda"] = cuda
    sys.modules["torch.amp"] = amp
    sys.modules["torch._dynamo"] = _dynamo

    triton = types.ModuleType("triton")
    triton._va_stub = True

    class _Config:
        def __init__(self, kwargs=None, num_warps: int = 4, num_stages: int = 2, **extra):
            self.kwargs = dict(kwargs or {})
            self.num_warps = num_warps
            self.num_stages = num_stages
            self.extra = extra

    triton.Config = _Config
    triton.jit = _noop_decorator
    triton.autotune = _noop_decorator
    triton.heuristics = _noop_decorator
    triton.cdiv = lambda a, b: (a + b - 1) // b
    triton.next_power_of_2 = lambda n: 1 << max(0, int(n) - 1).bit_length()
    triton.__version__ = "3.0.0"

    language = types.ModuleType("triton.language")

    class _ConstExpr:
        def __init__(self, value=None):
            self.value = value

    language.constexpr = _ConstExpr
    for name in (
        "arange",
        "zeros",
        "full",
        "load",
        "store",
        "dot",
        "trans",
        "maximum",
        "sum",
        "max",
        "where",
        "program_id",
        "num_programs",
        "range",
        "static_range",
        "float32",
        "float8e4nv",
        "bfloat16",
        "int32",
        "int64",
        "abs",
        "exp",
        "exp2",
        "log2",
        "cdiv",
        "minimum",
        "rsqrt",
        "sqrt",
        "erf",
        "sigmoid",
    ):
        setattr(language, name, lambda *a, **k: Tensor((1,)))
    language.math = types.SimpleNamespace(
        exp2=lambda *a, **k: Tensor((1,)), exp=lambda *a, **k: Tensor((1,))
    )
    # ``tl.range(...)`` inside a kernel is a plain Python loop in the stub.
    language.range = range
    language.static_range = range
    triton.language = language
    _mark_stub(triton, "triton")
    _mark_stub(language, "triton.language")
    sys.modules["triton"] = triton
    sys.modules["triton.language"] = language

    # -- huggingface_hub / safetensors --------------------------------
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda **kwargs: kwargs.get("filename", "stub.safetensors")
    hub.list_repo_files = lambda *a, **k: []
    hub.snapshot_download = lambda **kwargs: kwargs.get("local_dir", ".")
    hub.HfApi = type("HfApi", (), {"list_repo_files": lambda self, *a, **k: []})
    _mark_stub(hub, "huggingface_hub")
    sys.modules["huggingface_hub"] = hub

    safetensors = types.ModuleType("safetensors")
    safetensors_torch = types.ModuleType("safetensors.torch")
    safetensors_torch.load_file = lambda *a, **k: {}
    safetensors_torch.save_file = lambda *a, **k: None
    safetensors.torch = safetensors_torch
    _mark_stub(safetensors, "safetensors")
    _mark_stub(safetensors_torch, "safetensors.torch")
    sys.modules["safetensors"] = safetensors
    sys.modules["safetensors.torch"] = safetensors_torch
