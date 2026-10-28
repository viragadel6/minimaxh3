import base64

import numpy as np
import pytest
from PIL import Image

import media_io


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _solid_frames(count, width=8, height=6, value=128):
    return [
        np.full((height, width, 3), value, dtype=np.uint8) for _ in range(count)
    ]


# ---------------------------------------------------------------------------
# source handling
# ---------------------------------------------------------------------------
def test_data_uri_base64_and_percent():
    payload = b"hello\x00world"
    encoded = base64.b64encode(payload).decode()
    assert media_io._decode_data_uri(f"data:application/octet-stream;base64,{encoded}") == payload
    assert media_io._decode_data_uri("data:text/plain,hello%20world") == b"hello world"


def test_data_uri_rejects_missing_comma():
    with pytest.raises(ValueError):
        media_io._decode_data_uri("data:text/plain")


def test_fetch_bytes_reads_local_files(tmp_path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"abc")
    assert media_io._read_local_bytes(str(path)) == b"abc"


def test_fetch_bytes_rejects_unsupported(tmp_path):
    with pytest.raises(ValueError):
        media_io._read_local_bytes(str(tmp_path / "missing.bin"))
    import asyncio

    with pytest.raises(ValueError):
        asyncio.run(media_io.fetch_bytes(""))
    with pytest.raises(ValueError):
        asyncio.run(media_io.fetch_bytes("   "))


def test_fetch_bytes_is_awaitable(tmp_path):
    import asyncio

    path = tmp_path / "b.bin"
    path.write_bytes(b"xyz")
    assert asyncio.run(media_io.fetch_bytes(str(path))) == b"xyz"


# ---------------------------------------------------------------------------
# ffmpeg argument construction
# ---------------------------------------------------------------------------
def test_time_args_rejects_negative_start():
    with pytest.raises(ValueError):
        media_io._time_args(-1.0, None)


def test_time_args_rejects_nonpositive_window():
    with pytest.raises(ValueError):
        media_io._time_args(2.0, 2.0)
    with pytest.raises(ValueError):
        media_io._time_args(2.0, 1.0)


def test_time_args_rejects_nan_and_inf():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            media_io._time_args(None, bad)


def test_time_args_never_uses_scientific_notation():
    args = media_io._time_args(1e-7, 1e-7 + 1e-6)
    joined = " ".join(args)
    assert "e-" not in joined.lower()


def test_time_args_empty_when_unbounded():
    assert media_io._time_args(None, None) == []


# ---------------------------------------------------------------------------
# frame normalisation
# ---------------------------------------------------------------------------
def test_uint8_input_is_passed_through():
    frames = np.full((4, 6, 8, 3), 7, dtype=np.uint8)
    out = media_io.to_uint8_frames(frames)
    assert out.dtype == np.uint8
    assert out.shape == (4, 6, 8, 3)
    assert int(out.max()) == 7


def test_channel_first_and_batch_layouts():
    channels_first = np.zeros((7, 3, 6, 8), dtype=np.uint8)
    assert media_io.to_uint8_frames(channels_first).shape == (7, 6, 8, 3)

    # A leading singleton is a batch dimension, not a channel dimension.
    batched = np.zeros((1, 7, 6, 8, 3), dtype=np.uint8)
    assert media_io.to_uint8_frames(batched).shape == (7, 6, 8, 3)


def test_batched_video_is_rejected():
    with pytest.raises(ValueError):
        media_io.to_uint8_frames(np.zeros((2, 4, 6, 8, 3), dtype=np.uint8))


def test_empty_video_is_rejected():
    with pytest.raises(ValueError):
        media_io.to_uint8_frames(np.zeros((0, 6, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        media_io.to_uint8_frames([])


def test_float_zero_to_one_scales_to_255():
    frames = np.linspace(0.0, 1.0, 4 * 2 * 2 * 3, dtype=np.float32).reshape(4, 2, 2, 3)
    out = media_io.to_uint8_frames(frames)
    assert out.min() == 0 and out.max() == 255


def test_float_minus_one_to_one_scales_symmetrically():
    values = np.array([-1.0, 0.0, 1.0], dtype=np.float32).reshape(1, 1, 1, 3)
    frames = np.repeat(np.repeat(values, 2, axis=1), 2, axis=2)
    out = media_io.to_uint8_frames(frames)
    assert int(out.min()) == 0
    assert 126 <= int(out.mean()) <= 129
    assert int(out.max()) == 255


def test_float_out_of_range_is_clipped():
    frames = np.array([[[[5.0, 5.0, 255.0]]]], dtype=np.float32)
    out = media_io.to_uint8_frames(frames)
    assert out.max() <= 255 and out.min() >= 0


def test_uint16_is_rescaled_not_truncated():
    frames = np.full((2, 2, 2, 3), 65535, dtype=np.uint16)
    assert int(media_io.to_uint8_frames(frames).max()) == 255
    frames = np.full((2, 2, 2, 3), 256, dtype=np.uint16)
    out = media_io.to_uint8_frames(frames)
    assert 0 < int(out.mean()) < 5


def test_nan_and_inf_do_not_produce_garbage():
    frames = np.zeros((2, 2, 2, 3), dtype=np.float32)
    frames[0, 0, 0, 0] = np.nan
    frames[0, 0, 1, 0] = np.inf
    frames[0, 1, 0, 0] = -np.inf
    out = media_io.to_uint8_frames(frames)
    assert out.dtype == np.uint8
    assert np.isfinite(out).all()


def test_single_channel_is_expanded():
    frames = np.zeros((3, 4, 5, 1), dtype=np.uint8)
    assert media_io.to_uint8_frames(frames).shape == (3, 4, 5, 3)


def test_rgba_drops_alpha():
    frames = np.zeros((3, 4, 5, 4), dtype=np.uint8)
    assert media_io.to_uint8_frames(frames).shape == (3, 4, 5, 3)


def test_pil_list_is_accepted():
    frames = [Image.new("RGB", (5, 4), (1, 2, 3)) for _ in range(3)]
    out = media_io.to_uint8_frames(frames)
    assert out.shape == (3, 4, 5, 3)


def test_single_frame_is_accepted():
    out = media_io.to_uint8_frames(np.zeros((4, 5, 3), dtype=np.uint8))
    assert out.shape == (1, 4, 5, 3)


def test_ambiguous_layout_raises():
    # (4, 3, 6, 8) can be read as (F, C, H, W) or (C, F, H, W); refuse to guess.
    with pytest.raises(ValueError) as excinfo:
        media_io.to_uint8_frames(np.zeros((4, 3, 6, 8), dtype=np.uint8))
    assert "layout" in str(excinfo.value)


def test_explicit_layout_resolves_ambiguity():
    data = np.zeros((4, 3, 6, 8), dtype=np.uint8)
    assert media_io.to_uint8_frames(data, layout="channel_first").shape == (4, 6, 8, 3)
    # (C, F, H, W) with C in {3, 4} loses a channel in the RGB conversion.
    stacked = media_io.to_uint8_frames(data, layout="channel_first_stacked")
    assert stacked.shape == (3, 6, 8, 3)


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------
def test_to_stereo_mono_and_stereo():
    mono = np.linspace(-1, 1, 100, dtype=np.float32)
    stereo = media_io.to_stereo(mono)
    assert stereo.shape == (100, 2)
    stereo_in = np.stack([mono, mono], axis=1)
    assert media_io.to_stereo(stereo_in).shape == (100, 2)


def test_to_stereo_channel_first():
    data = np.zeros((2, 50), dtype=np.float32)
    assert media_io.to_stereo(data).shape == (50, 2)


def test_to_stereo_downmixes_many_channels():
    data = np.zeros((10, 6), dtype=np.float32)
    assert media_io.to_stereo(data).shape == (10, 2)


def test_to_stereo_handles_none():
    out = media_io.to_stereo(None)
    assert out.shape == (0, 2)


def test_to_stereo_normalises_integer_pcm():
    pcm = np.full((10, 2), 32767, dtype=np.int16)
    out = media_io.to_stereo(pcm)
    assert out.max() <= 1.0 + 1e-6


def test_to_stereo_rejects_bad_rank():
    with pytest.raises(ValueError):
        media_io.to_stereo(np.zeros((2, 2, 2), dtype=np.float32))


# ---------------------------------------------------------------------------
# stitching and trimming
# ---------------------------------------------------------------------------
def test_stitch_drops_overlap_frames_only():
    segments = [_solid_frames(5), _solid_frames(5)]
    audio = [np.zeros((100, 2), np.float32), np.zeros((100, 2), np.float32)]
    frames, stitched = media_io.stitch_segments(segments, audio, 44100, 24)
    assert frames.shape[0] == 9
    # Audio is rebuilt to match the stitched duration, not the sum of the inputs.
    assert stitched is not None
    assert stitched.shape[0] == round(9 * 44100 / 24)


def test_stitch_returns_no_audio_when_every_segment_is_silent():
    frames, stitched = media_io.stitch_segments(
        [_solid_frames(5), _solid_frames(5)], [None, None], 44100, 24
    )
    assert frames.shape[0] == 9
    assert stitched is None


def test_stitch_pads_a_missing_segment_with_silence():
    audio = [np.full((100, 2), 0.5, np.float32), None]
    frames, stitched = media_io.stitch_segments(
        [_solid_frames(5), _solid_frames(5)], audio, 44100, 24
    )
    assert frames.shape[0] == 9
    assert stitched is not None
    # The first segment carries audio, the second is silent padding.
    assert stitched[:100].max() > 0.4
    assert np.allclose(stitched[200:], 0.0, atol=1e-6)


def test_stitch_rejects_audio_length_mismatch():
    with pytest.raises(ValueError):
        media_io.stitch_segments([_solid_frames(5)], [None, None], 44100, 24)


def test_stitch_requires_segments():
    with pytest.raises(ValueError):
        media_io.stitch_segments([], [], 44100, 24)


def test_stitch_rejects_too_short_segment():
    with pytest.raises(ValueError):
        media_io.stitch_segments([_solid_frames(1), _solid_frames(1)], [None, None], 44100, 24)


def test_trim_pads_and_crops():
    frames = np.zeros((10, 4, 4, 3), dtype=np.uint8)
    audio = np.zeros((100, 2), dtype=np.float32)
    trimmed = media_io.trim(frames, audio, 5, 44100, 24)
    frames_out, audio_out = trimmed
    assert frames_out.shape[0] == 5
    assert audio_out.shape[0] == pytest.approx(round(5 * 44100 / 24), abs=2)


def test_trim_handles_none_audio():
    frames = np.zeros((10, 4, 4, 3), dtype=np.uint8)
    frames_out, audio_out = media_io.trim(frames, None, 4, 44100, 24)
    assert frames_out.shape[0] == 4
    assert audio_out is None


def test_trim_rejects_bad_frame_count():
    frames = np.zeros((10, 4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        media_io.trim(frames, None, -1, 44100, 24)


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------
def test_encode_mp4_rejects_empty_frames(tmp_path):
    with pytest.raises(ValueError):
        media_io.encode_mp4([], None, 44100, 24, tmp_path / "out.mp4")


def test_encode_mp4_rejects_directory_target(tmp_path):
    with pytest.raises(ValueError):
        media_io.encode_mp4(_solid_frames(2), None, 44100, 24, tmp_path)


def test_encode_mp4_roundtrip(tmp_path):
    out = tmp_path / "clip.mp4"
    audio = np.zeros((4410, 2), dtype=np.float32)
    try:
        media_io.encode_mp4(_solid_frames(5, 32, 32), audio, 44100, 24, out)
    except (RuntimeError, FileNotFoundError) as exc:  # pragma: no cover
        pytest.skip(f"ffmpeg unavailable: {exc}")
    assert out.is_file() and out.stat().st_size > 0


def test_odd_dimensions_are_padded():
    frames = np.zeros((2, 5, 7, 3), dtype=np.uint8)
    padded = media_io._pad_to_even(frames)
    assert padded.shape[1] % 2 == 0 and padded.shape[2] % 2 == 0
