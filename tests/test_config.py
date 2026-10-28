import math

import pytest

from config import Settings, _coerce_bool, load_settings, profile_repair_tau, profile_steps


def test_settings_defaults_are_valid():
    settings = Settings()
    assert settings.FPS > 0
    assert settings.DEFAULT_PROFILE in ("quality", "fast", "turbo")
    assert settings.max_frames() >= 124


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1", True),
        ("0", False),
        ("true", True),
        ("FALSE", False),
        ("yes", True),
        ("no", False),
        ("on", True),
        ("off", False),
    ],
)
def test_bool_coercion(raw, expected):
    assert _coerce_bool(raw) is expected


def test_bool_coercion_rejects_garbage():
    with pytest.raises(ValueError):
        _coerce_bool("maybe")


def test_load_settings_coerces_types_and_rejects_bad_values(monkeypatch):
    monkeypatch.setenv("VAE_TILING", "0")
    monkeypatch.setenv("FAST_CACHE_THRESHOLD", "0.25")
    settings = load_settings()
    assert settings.VAE_TILING == 0
    assert settings.FAST_CACHE_THRESHOLD == pytest.approx(0.25)

    monkeypatch.setenv("FPS", "0")
    with pytest.raises(ValueError):
        load_settings()


def test_settings_reject_negative_edge_blocks():
    with pytest.raises(ValueError):
        Settings(FP8_KEEP_EDGE_BLOCKS=-1)


def test_profile_helpers():
    settings = Settings()
    assert profile_steps(settings, "fast") == settings.FAST_STEPS
    assert profile_repair_tau(settings, "quality") == pytest.approx(
        settings.QUALITY_REPAIR_TAU
    )
    with pytest.raises(ValueError):
        profile_steps(settings, "nope")
    with pytest.raises(ValueError):
        profile_repair_tau(settings, "nope")


def test_timeouts_are_finite():
    settings = Settings()
    assert math.isfinite(settings.SYNC_REQUEST_TIMEOUT)
    assert settings.POLL_TIMEOUT_SECONDS > 0
