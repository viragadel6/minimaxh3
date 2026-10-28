"""Shared test fixtures.

The sandbox used for development has no GPU stack, so a lightweight ``modal`` stub and
an optional ``torch`` guard keep the pure-Python layers testable.  Tests that need
torch are skipped automatically; the stubs only implement what the code imports.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# modal stub
# ---------------------------------------------------------------------------
def _install_modal_stub() -> None:
    if importlib.util.find_spec("modal") is not None:
        return

    modal = types.ModuleType("modal")

    class _Image:
        def __init__(self, *args, **kwargs):
            self.args = args

        def apt_install(self, *args, **kwargs):
            return self

        def pip_install(self, *args, **kwargs):
            return self

        def env(self, *args, **kwargs):
            return self

        def add_local_python_source(self, *args, **kwargs):
            return self

        def run_commands(self, *args, **kwargs):
            return self

        @staticmethod
        def from_registry(*args, **kwargs):
            return _Image()

        @staticmethod
        def debian_slim(*args, **kwargs):
            return _Image()

    class _Volume:
        def __init__(self):
            self.commits = 0
            self.reloads = 0

        @staticmethod
        def from_name(name, create_if_missing=True):
            return _Volume()

        def commit(self):
            self.commits += 1

        def reload(self):
            self.reloads += 1

    class _Dict:
        def __init__(self):
            self.data = {}

        @staticmethod
        def from_name(name, create_if_missing=True):
            return _Dict()

        def put(self, key, value):
            self.data[key] = value

        def get(self, key):
            return self.data.get(key)

        def pop(self, key):
            return self.data.pop(key)

    class _FunctionCall:
        @staticmethod
        def from_id(call_id):
            raise TimeoutError("function call has not finished")

    class _App:
        def __init__(self, name=None):
            self.name = name

        def function(self, **kwargs):
            def decorator(fn):
                fn.app_function_kwargs = kwargs
                return fn

            return decorator

        def cls(self, **kwargs):
            def decorator(cls):
                cls.app_cls_kwargs = kwargs
                return cls

            return decorator

        def local_entrypoint(self, **kwargs):
            def decorator(fn):
                return fn

            return decorator

    def _identity_decorator(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(fn):
            return fn

        return decorator

    exception = types.ModuleType("modal.exception")

    class _ModalTimeout(TimeoutError):
        pass

    exception.TimeoutError = _ModalTimeout
    exception.FunctionTimeoutError = _ModalTimeout

    modal.App = _App
    modal.Image = _Image
    modal.Volume = _Volume
    modal.Dict = _Dict
    modal.FunctionCall = _FunctionCall
    modal.method = _identity_decorator
    modal.enter = _identity_decorator
    modal.exit = _identity_decorator
    modal.asgi_app = _identity_decorator
    modal.concurrent = _identity_decorator
    modal.exception = exception
    sys.modules["modal"] = modal
    sys.modules["modal.exception"] = exception


_install_modal_stub()


def _find_torch() -> bool:
    module = sys.modules.get("torch")
    if module is not None:
        return not getattr(module, "_va_stub", False)
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken environments
        return False


HAS_TORCH = _find_torch()
