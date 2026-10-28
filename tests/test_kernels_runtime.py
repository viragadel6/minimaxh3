"""Import-level and pure-logic tests for the GPU modules.

These run against the shallow torch/triton stubs in ``tests/stubs.py``: they prove the
modules import cleanly, the router's configuration/scope logic is sound, and the
bookkeeping helpers behave, but they cannot validate kernel numerics.
"""

import dataclasses
import math

import pytest

import stubs

stubs.install()

import torch  # noqa: E402  (the stub)

from va_kernels import attention_router  # noqa: E402
from va_kernels.attention_router import (  # noqa: E402
    RouterConfig,
    configure,
    denoiser_scope,
    get_config,
    install,
    instrument_transformer,
    is_installed,
    routed_sdpa,
    uninstall,
)
from va_kernels.fp8_attention import (  # noqa: E402
    _bh_grid,
    _prepare_key_side,
    _round_up,
    is_supported as attention_supported,
    requires_grad as attention_requires_grad,
)
from va_kernels.fp8_linear import (  # noqa: E402
    add_adapter,
    clear_adapters,
    find_primary_block_list,
    has_adapters,
    iter_adapters,
)


@pytest.fixture(autouse=True)
def _restore_router():
    original = configure(
        enabled=True, repair_tau=-1.0, min_seq=4096
    )
    was_installed = is_installed()
    yield
    configure(
        enabled=original.enabled,
        repair_tau=original.repair_tau,
        min_seq=original.min_seq,
    )
    if not was_installed and is_installed():
        uninstall()


# ---------------------------------------------------------------------------
# router configuration
# ---------------------------------------------------------------------------
def test_config_is_frozen_and_defaulted():
    config = get_config()
    assert isinstance(config, RouterConfig)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.enabled = False


def test_configure_returns_the_new_snapshot():
    configure(enabled=False, repair_tau=0.25, min_seq=128)
    snapshot = configure(enabled=True, repair_tau=-1.0, min_seq=4096)
    assert snapshot.enabled is True
    assert snapshot.repair_tau == pytest.approx(-1.0)
    assert snapshot.min_seq == 4096
    assert get_config() is snapshot or get_config() == snapshot


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": 1},
        {"enabled": "yes"},
        {"repair_tau": float("nan")},
        {"repair_tau": float("inf")},
        {"min_seq": 0},
        {"min_seq": -4},
        {"min_seq": 1.5},
    ],
)
def test_configure_rejects_invalid_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        configure(**kwargs)


def test_module_level_attributes_track_the_snapshot():
    configure(enabled=False, repair_tau=0.5, min_seq=64)
    assert attention_router.enabled is False
    assert attention_router.repair_tau == pytest.approx(0.5)
    assert attention_router.min_seq == 64
    with pytest.raises(AttributeError):
        attention_router.does_not_exist  # noqa: B018 - attribute probe


# ---------------------------------------------------------------------------
# installation
# ---------------------------------------------------------------------------
def test_install_and_uninstall_are_idempotent():
    import torch.nn.functional as F

    native = F.scaled_dot_product_attention
    install()
    install()
    assert is_installed() is True
    assert F.scaled_dot_product_attention is routed_sdpa
    uninstall()
    uninstall()
    assert is_installed() is False
    assert F.scaled_dot_product_attention is native


# ---------------------------------------------------------------------------
# scopes
# ---------------------------------------------------------------------------
def test_denoiser_scope_nests_and_unwinds():
    assert attention_router._get_depth() == 0
    with denoiser_scope():
        assert attention_router._get_depth() == 1
        with denoiser_scope():
            assert attention_router._get_depth() == 2
        assert attention_router._get_depth() == 1
    assert attention_router._get_depth() == 0


def test_denoiser_scope_unwinds_on_error():
    with pytest.raises(RuntimeError):
        with denoiser_scope():
            raise RuntimeError("boom")
    assert attention_router._get_depth() == 0


def test_instrument_transformer_is_stable_and_idempotent():
    class FakeTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.depths = []
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(), torch.nn.Linear()])

        def forward(self, x):
            self.depths.append(attention_router._get_depth())
            return x

    transformer = FakeTransformer()
    original_cls = type(transformer)
    instrument_transformer(transformer)
    scoped_cls = type(transformer)
    assert scoped_cls is not original_cls
    assert issubclass(scoped_cls, original_cls)
    assert scoped_cls.__name__.endswith("DenoiserScoped")

    instrument_transformer(transformer)
    assert type(transformer) is scoped_cls  # no new class on a second call

    transformer(torch.zeros(1))
    assert transformer.depths == [1]
    assert attention_router._get_depth() == 0


# ---------------------------------------------------------------------------
# routed_sdpa fallbacks (stub tensors only exercise the decision logic)
# ---------------------------------------------------------------------------
def _sdpa_args(seq=8192, dtype=None):
    q = torch.Tensor((1, 2, seq, 128), dtype or torch.bfloat16)
    k = torch.Tensor((1, 2, seq, 128), dtype or torch.bfloat16)
    v = torch.Tensor((1, 2, seq, 128), dtype or torch.bfloat16)
    return q, k, v


def test_routed_sdpa_uses_native_outside_a_scope():
    configure(enabled=True, repair_tau=-1.0, min_seq=1)
    q, k, v = _sdpa_args()
    assert routed_sdpa(q, k, v) is v  # the stub returns ``v``


def test_routed_sdpa_respects_min_seq():
    configure(enabled=True, repair_tau=-1.0, min_seq=1 << 30)
    q, k, v = _sdpa_args()
    with denoiser_scope():
        assert routed_sdpa(q, k, v) is v


def test_routed_sdpa_falls_back_when_disabled():
    configure(enabled=False, repair_tau=-1.0, min_seq=1)
    q, k, v = _sdpa_args()
    with denoiser_scope():
        assert routed_sdpa(q, k, v) is v


def test_routed_sdpa_falls_back_on_dropout_and_causal():
    configure(enabled=True, repair_tau=-1.0, min_seq=1)
    q, k, v = _sdpa_args()
    with denoiser_scope():
        assert routed_sdpa(q, k, v, dropout_p=0.1) is v
        assert routed_sdpa(q, k, v, is_causal=True) is v
        assert routed_sdpa(q, k, v, enable_gqa=True) is v


def test_routed_sdpa_accepts_positional_arguments():
    configure(enabled=True, repair_tau=-1.0, min_seq=1)
    q, k, v = _sdpa_args()
    with denoiser_scope():
        assert routed_sdpa(q, k, v, None, 0.0, False, 0.5, False) is v


# ---------------------------------------------------------------------------
# fp8_attention helpers
# ---------------------------------------------------------------------------
def test_bh_grid_is_two_dimensional_and_bounded():
    low, high = _bh_grid(4)
    assert high == 1
    low, high = _bh_grid(200_000)
    assert low == 65535
    assert low * high >= 200_000


def test_round_up():
    assert _round_up(5, 16) == 16
    assert _round_up(16, 16) == 16
    assert _round_up(345, 16) == 352


def test_attention_requires_grad_detects_flags():
    q, k, v = _sdpa_args(seq=16)
    assert attention_requires_grad(q, k, v) is False
    q.requires_grad = True
    assert attention_requires_grad(q, k, v) is True


def test_attention_support_requires_cuda_and_good_dims():
    q, k, v = _sdpa_args(seq=16)
    assert attention_supported(q, k, v) is False  # stub has no CUDA
    broken = torch.Tensor((1, 2, 16, 96), torch.bfloat16)
    assert attention_supported(broken, broken, broken) is False


def test_prepare_key_side_rejects_wrong_device():
    mask = torch.Tensor((1, 1, 1, 16), torch.bool)
    with pytest.raises(ValueError) as excinfo:
        _prepare_key_side(
            mask, "attn_mask", 1, 16, torch.device("cuda:0"), torch.bfloat16, as_bool=True
        )
    assert "cuda:0" in str(excinfo.value)


def test_prepare_key_side_accepts_broadcast_masks():
    mask = torch.Tensor((16,), torch.bool)
    view, stride_b, stride_k = _prepare_key_side(
        mask,
        "attn_mask",
        2,
        16,
        torch.device("cpu"),
        torch.bfloat16,
        as_bool=True,
    )
    assert view.shape[-1] == 16
    # Real torch reports a zero batch stride here (broadcast, no copy); the stub
    # cannot model strides, so only the shape contract is asserted.
    assert isinstance(stride_b, int) and isinstance(stride_k, int)

    with pytest.raises(ValueError):
        _prepare_key_side(
            torch.Tensor((2, 3, 16), torch.bool),
            "attn_mask",
            2,
            16,
            torch.device("cpu"),
            torch.bfloat16,
            as_bool=True,
        )


# ---------------------------------------------------------------------------
# fp8_linear helpers
# ---------------------------------------------------------------------------
def test_find_primary_block_list_prefers_the_longest_stack():
    class WithBlocks(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear() for _ in range(4)])
            self.experts = torch.nn.ModuleList([torch.nn.Linear() for _ in range(2)])

    name, blocks = find_primary_block_list(WithBlocks())
    assert name == "blocks"
    assert len(list(blocks)) == 4

    class NoLists(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear()

    with pytest.raises(ValueError):
        find_primary_block_list(NoLists())


def test_lora_adapter_helpers_roundtrip():
    layer = torch.nn.Linear(4, 4)
    assert has_adapters(layer) is False
    assert list(iter_adapters(layer)) == []
    add_adapter(layer, torch.zeros((2, 4)), torch.zeros((4, 2)), 0.5)
    assert has_adapters(layer) is True
    assert len(list(iter_adapters(layer))) == 1
    clear_adapters(layer)
    assert has_adapters(layer) is False


def test_router_config_values_survive_reload():
    configure(enabled=True, repair_tau=0.75, min_seq=512)
    config = get_config()
    assert config.repair_tau == pytest.approx(0.75)
    assert math.isfinite(config.repair_tau)


# ---------------------------------------------------------------------------
# import surface
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "module_name",
    [
        "va_kernels",
        "va_kernels.attention_router",
        "va_kernels.fp8_attention",
        "va_kernels.fp8_linear",
        "runtime.generator",
        "runtime.jobs",
        "runtime.lora",
        "runtime.model_manager",
        "runtime.pipeline_adapter",
        "runtime.step_cache",
        "runtime.weights",
    ],
)
def test_module_imports(module_name):
    import importlib

    assert importlib.import_module(module_name) is not None


def test_va_kernels_lazy_exports():
    import importlib

    package = importlib.import_module("va_kernels")
    assert package.__all__
    assert package.FP8Linear is not None
    assert package.routed_sdpa is not None
    with pytest.raises(AttributeError):
        package.not_a_real_export  # noqa: B018 - attribute probe
