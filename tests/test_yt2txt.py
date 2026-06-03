"""Unit tests for the pure, deterministic helpers in yt2txt.

Scope: slug, fmt_timestamps, pick_backend, DEFAULT_MODEL.
Download/transcription paths (download_audio, transcribe_*, main) are NOT
tested here — they require network, ffmpeg, and the heavy whisper backends.

Boundary Value Analysis (BVA) is applied to every input domain with a
meaningful boundary; each BVA case is labelled B-1 / B / B+1 in a comment.

Runs with a plain `import yt2txt` because pyproject.toml sets
pythonpath = ["."] for pytest.
"""
import sys

import pytest

import yt2txt


# ---------------------------------------------------------------------------
# slug(s) -> str
# ---------------------------------------------------------------------------

def test_slug_keeps_allowed_chars_unchanged():
    # a-z A-Z 0-9 . _ - are all allowed and must pass through verbatim
    assert yt2txt.slug("aZ0._-") == "aZ0._-"


def test_slug_replaces_run_of_invalid_with_single_underscore():
    # a contiguous run of invalid chars collapses to ONE underscore
    assert yt2txt.slug("a   b") == "a_b"
    assert yt2txt.slug("a@#$b") == "a_b"


def test_slug_strips_leading_and_trailing_underscores():
    # leading/trailing invalid runs become underscores then get stripped
    assert yt2txt.slug("  hello  ") == "hello"
    assert yt2txt.slug("***hello***") == "hello"


def test_slug_empty_string_returns_video():
    # empty input -> fallback "video"  (empty-domain boundary)
    assert yt2txt.slug("") == "video"


def test_slug_all_invalid_returns_video():
    # all-invalid input collapses to "_", strips to "", -> "video"
    assert yt2txt.slug("***") == "video"


def test_slug_unicode_run_collapses_to_video():
    # non-ASCII letters are NOT in [a-zA-Z0-9._-]; a pure run -> "video"
    assert yt2txt.slug("日本語") == "video"


# --- BVA: truncation boundary B = 80 chars -------------------------------
# Inputs are pure valid chars (no leading/trailing "_", no invalid runs),
# so the only transform that matters is the [:80] truncation.

def test_slug_truncation_b_minus_1_length_79():
    # B-1: 79 valid chars -> length stays 79 (below the cut)
    out = yt2txt.slug("a" * 79)
    assert len(out) == 79
    assert out == "a" * 79


def test_slug_truncation_b_exactly_80():
    # B: exactly 80 valid chars -> length stays 80 (at the cut, untouched)
    out = yt2txt.slug("a" * 80)
    assert len(out) == 80
    assert out == "a" * 80


def test_slug_truncation_b_plus_1_length_81_truncated_to_80():
    # B+1: 81 valid chars -> truncated to 80
    out = yt2txt.slug("a" * 81)
    assert len(out) == 80
    assert out == "a" * 80


# ---------------------------------------------------------------------------
# fmt_timestamps(segments) -> str
# ---------------------------------------------------------------------------

def test_fmt_timestamps_empty_returns_single_newline():
    # empty-domain boundary: no segments -> just the trailing "\n"
    assert yt2txt.fmt_timestamps([]) == "\n"


def test_fmt_timestamps_single_segment_format():
    out = yt2txt.fmt_timestamps([{"start": 5.0, "text": "hello"}])
    assert out == "[00:05] hello\n"


def test_fmt_timestamps_strips_segment_text():
    # each segment's text is .strip()-ed
    out = yt2txt.fmt_timestamps([{"start": 0.0, "text": "  spaced  "}])
    assert out == "[00:00] spaced\n"


def test_fmt_timestamps_multiple_segments_joined_with_newline():
    segs = [
        {"start": 0.0, "text": "first"},
        {"start": 75.0, "text": "second"},
    ]
    out = yt2txt.fmt_timestamps(segs)
    assert out == "[00:00] first\n[01:15] second\n"


def test_fmt_timestamps_truncates_float_start_toward_zero():
    # start is int()-truncated, not rounded: 59.9 -> 59
    out = yt2txt.fmt_timestamps([{"start": 59.9, "text": "x"}])
    assert out == "[00:59] x\n"


def test_fmt_timestamps_minutes_and_seconds_zero_padded():
    # mm and ss each zero-padded to 2 digits (here 3661s -> 61:01)
    out = yt2txt.fmt_timestamps([{"start": 3661.0, "text": "x"}])
    assert out == "[61:01] x\n"


# --- BVA: minute rollover boundary B = 60 seconds ------------------------

def test_fmt_timestamps_rollover_zero():
    # below the boundary entirely: 0s -> [00:00]
    out = yt2txt.fmt_timestamps([{"start": 0.0, "text": "x"}])
    assert out == "[00:00] x\n"


def test_fmt_timestamps_rollover_b_minus_1_59s():
    # B-1: 59s -> [00:59]  (still minute 0)
    out = yt2txt.fmt_timestamps([{"start": 59.0, "text": "x"}])
    assert out == "[00:59] x\n"


def test_fmt_timestamps_rollover_b_60s():
    # B: 60s -> [01:00]  (rolls over to minute 1, seconds reset)
    out = yt2txt.fmt_timestamps([{"start": 60.0, "text": "x"}])
    assert out == "[01:00] x\n"


def test_fmt_timestamps_rollover_b_plus_1_61s():
    # B+1: 61s -> [01:01]
    out = yt2txt.fmt_timestamps([{"start": 61.0, "text": "x"}])
    assert out == "[01:01] x\n"


# ---------------------------------------------------------------------------
# pick_backend(arg) -> str
#
# Platform-independent: is_apple_silicon is monkeypatched, and the
# importability of mlx_whisper is simulated via sys.modules so the real
# host arch / installed packages never affect the result.
# ---------------------------------------------------------------------------

def _set_mlx_importable(monkeypatch):
    """Make `import mlx_whisper` succeed by injecting a dummy module."""
    import types
    monkeypatch.setitem(sys.modules, "mlx_whisper", types.ModuleType("mlx_whisper"))


def _set_mlx_missing(monkeypatch):
    """Make `import mlx_whisper` raise ImportError."""
    # None in sys.modules forces ImportError on `import mlx_whisper`.
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)


def test_pick_backend_mlx_passthrough(monkeypatch):
    # arg != "auto" returns arg unchanged regardless of platform
    monkeypatch.setattr(yt2txt, "is_apple_silicon", lambda: False)
    assert yt2txt.pick_backend("mlx") == "mlx"


def test_pick_backend_faster_passthrough(monkeypatch):
    monkeypatch.setattr(yt2txt, "is_apple_silicon", lambda: True)
    assert yt2txt.pick_backend("faster") == "faster"


def test_pick_backend_arbitrary_passthrough(monkeypatch):
    # any non-"auto" value is returned verbatim
    monkeypatch.setattr(yt2txt, "is_apple_silicon", lambda: True)
    assert yt2txt.pick_backend("anything") == "anything"


def test_pick_backend_auto_apple_silicon_mlx_importable(monkeypatch):
    # auto + Apple Silicon + mlx-whisper importable -> "mlx"
    monkeypatch.setattr(yt2txt, "is_apple_silicon", lambda: True)
    _set_mlx_importable(monkeypatch)
    assert yt2txt.pick_backend("auto") == "mlx"


def test_pick_backend_auto_apple_silicon_mlx_missing(monkeypatch):
    # auto + Apple Silicon + mlx-whisper NOT importable -> "faster"
    monkeypatch.setattr(yt2txt, "is_apple_silicon", lambda: True)
    _set_mlx_missing(monkeypatch)
    assert yt2txt.pick_backend("auto") == "faster"


def test_pick_backend_auto_not_apple_silicon(monkeypatch):
    # auto + not Apple Silicon -> "faster" (mlx never even attempted)
    monkeypatch.setattr(yt2txt, "is_apple_silicon", lambda: False)
    # even if mlx were importable, non-Apple-Silicon must yield "faster"
    _set_mlx_importable(monkeypatch)
    assert yt2txt.pick_backend("auto") == "faster"


# ---------------------------------------------------------------------------
# DEFAULT_MODEL
# ---------------------------------------------------------------------------

def test_default_model_has_both_keys():
    assert set(yt2txt.DEFAULT_MODEL) >= {"mlx", "faster"}


@pytest.mark.parametrize("key", ["mlx", "faster"])
def test_default_model_values_non_empty_strings(key):
    value = yt2txt.DEFAULT_MODEL[key]
    assert isinstance(value, str)
    assert value.strip() != ""
