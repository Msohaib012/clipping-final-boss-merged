"""Unit tests for the ported A/V sync module (clipper/av_align.py) and the
render_clip av_offset_s dual-input command."""

import json
import types
from pathlib import Path

import pytest

from clipper import av_align
from clipper.av_align import (
    _keys,
    _keys_match,
    _ocr_band,
    channel_key_for,
    load_clock_fix,
    measure_av_offset,
    measure_av_offset_from_transcript,
    save_clock_fix,
)


# --- _keys / _keys_match (substring-tolerant matching) ---

def test_keys_normalises_and_drops_short_words():
    assert _keys("Hello world, hello!") == {"hello", "world"}
    assert _keys("a an of to") == set()  # all < 3 chars after normalising
    assert _keys("") == set()
    assert _keys(None) == set()


def test_keys_match_equal():
    assert _keys_match("hello", "hello") is True


def test_keys_match_substring_tolerant():
    # Donor example: subtitle 'uh-huh' (-> 'uhhuh') vs transcript '-huh.'
    # (-> 'huh'): 'huh' is a substring of 'uhhuh', so they match.
    assert _keys_match("huh", "uhhuh") is True
    assert _keys_match("uhhuh", "huh") is True


def test_keys_match_unrelated_and_short():
    assert _keys_match("hello", "world") is False
    assert _keys_match("ab", "abc") is False  # len < 3 on one side
    assert _keys_match("", "hello") is False


# --- channel_key_for ---

def test_channel_key_for_youtube_watch_url():
    assert channel_key_for("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    # id embedded in a longer path still wins (11-char YouTube id)
    assert channel_key_for("/tmp/seg_watch?v=AbC123_-xYz.mp4") == "AbC123_-xYz"


def test_channel_key_for_twitch_seg_name():
    assert channel_key_for("seg_kaicenat_20261119-120000.mp4") == "kaicenat"
    # channel name ending in underscore (e.g. caseoh_) is preserved
    assert channel_key_for("seg_caseoh__20261119-120000.mp4") == "caseoh_"


def test_channel_key_for_fallback_is_basename_stem():
    assert channel_key_for("/tmp/my_video.mp4") == "my_video"
    assert channel_key_for("") == "unknown"


# --- load/save clock fix round-trip ---

def test_clock_fix_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(av_align, "CLOCK_FIX_FILE", tmp_path / "av_clock_fix.json")
    assert load_clock_fix("kaicenat", 0.0) == 0.0  # missing file -> default
    save_clock_fix("kaicenat", 4.17)
    assert load_clock_fix("kaicenat", 0.0) == 4.17
    save_clock_fix("kaicenat", 2.5)
    assert load_clock_fix("kaicenat", 0.0) == 2.5  # overwrite
    assert load_clock_fix("other_channel", 7.0) == 7.0  # unknown key -> default
    # on-disk JSON shape is {"channel_key": offset}
    data = json.loads((tmp_path / "av_clock_fix.json").read_text(encoding="utf-8"))
    assert data == {"kaicenat": 2.5}


def test_clock_fix_load_corrupt_file_returns_default(tmp_path, monkeypatch):
    bad = tmp_path / "av_clock_fix.json"
    bad.write_text("not json{", encoding="utf-8")
    monkeypatch.setattr(av_align, "CLOCK_FIX_FILE", bad)
    assert load_clock_fix("kaicenat", 3.0) == 3.0


# --- measure_av_offset_from_transcript: seed fallbacks ---

def test_measure_returns_seed_with_no_transcript_words_or_segments():
    # Empty transcript and CFB-shaped transcript with no segments.
    offset, n, details = measure_av_offset_from_transcript(
        "seg_kaicenat_20261119-120000.mp4",
        {"segments": [], "words": []}, 0.0, 60.0, seed=1.5, verbose=False)
    assert offset == 1.5
    assert n == 0
    assert details["reason"] == "no-transcript"

    offset, n, details = measure_av_offset_from_transcript(
        "seg_kaicenat_20261119-120000.mp4",
        {"segments": [], "full_text": ""}, 0.0, 60.0, seed=0.0, verbose=False)
    assert offset == 0.0
    assert n == 0


def test_measure_returns_seed_when_tesseract_missing(monkeypatch, tmp_path):
    # tesseract unavailable -> seed, without ever touching ffmpeg/video.
    monkeypatch.setattr(av_align.shutil, "which", lambda *a, **k: None)
    transcript = {
        "segments": [
            {"start": 1.0, "end": 3.0, "text": "hello world today",
             "words": [{"word": "hello", "start": 1.0, "end": 1.5}]},
        ],
    }
    offset, n, details = measure_av_offset_from_transcript(
        "seg_kaicenat_20261119-120000.mp4", transcript, 0.0, 60.0,
        seed=2.25, verbose=False)
    assert offset == 2.25
    assert n == 0
    assert details["reason"] == "tesseract-missing"


def test_ocr_band_returns_empty_when_tesseract_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(av_align.shutil, "which", lambda *a, **k: None)
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"not a real image, never read")
    assert _ocr_band(str(frame)) == ""


def test_measure_cfb_transcript_shape_reaches_frame_sampling(monkeypatch):
    # CFB transcribe() has NO top-level "words" (segments carry them).
    # The wrapper must flatten them instead of bailing as no-transcript:
    # with _sample_frames stubbed to return no frames, the reason must be
    # "too-few-frames", not "no-transcript".
    monkeypatch.setattr(
        av_align.shutil, "which",
        lambda name, *a, **k: "/usr/bin/tesseract" if name == "tesseract" else None)
    monkeypatch.setattr(av_align, "_sample_frames", lambda *a, **k: [])
    transcript = {
        "segments": [
            {"start": 1.0, "end": 3.0, "text": "hello world today",
             "words": [{"word": "hello", "start": 1.0, "end": 1.5}]},
        ],
    }
    offset, n, details = measure_av_offset_from_transcript(
        "seg_x.mp4", transcript, 0.0, 60.0, seed=1.0, verbose=False)
    assert offset == 1.0
    assert details["reason"] == "too-few-frames"


def test_measure_never_raises_on_bad_inputs(monkeypatch):
    # None transcript, garbage segment times, missing video file: seed.
    offset, n, details = measure_av_offset_from_transcript(
        "/nonexistent/video.mp4", None, 0.0, 60.0, seed=0.75, verbose=False)
    assert offset == 0.75
    assert n == 0


def test_measure_av_offset_file_wrapper_returns_seed_for_empty_transcript(tmp_path):
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(json.dumps({"segments": [], "words": []}),
                               encoding="utf-8")
    offset, n, details = measure_av_offset(
        "seg_x.mp4", str(transcript_path), 0.0, 60.0, seed=3.0, verbose=False)
    assert offset == 3.0
    assert n == 0
    assert details["reason"] == "no-transcript"


def test_measure_av_offset_file_wrapper_missing_file_returns_seed(tmp_path):
    offset, n, details = measure_av_offset(
        "seg_x.mp4", str(tmp_path / "does_not_exist.json"),
        0.0, 60.0, seed=1.25, verbose=False)
    assert offset == 1.25
    assert n == 0
    assert details["reason"] == "transcript-load-failed"


# --- render_clip: av_offset_s dual-input command ---

def _fake_ffmpeg_run_factory(captured):
    """Fake subprocess.run: records the cmd, materialises the staging
    output file so render_clip's success check passes."""
    def fake_run(cmd, **kwargs):
        captured.append(list(cmd))
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x" * 2000)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    return fake_run


def test_render_clip_zero_offset_keeps_single_input(monkeypatch, tmp_path):
    from clipper import renderer
    captured = []
    monkeypatch.setattr(renderer.subprocess, "run",
                        _fake_ffmpeg_run_factory(captured))
    out = tmp_path / "out.mp4"
    ok = renderer.render_clip(
        input_video=tmp_path / "in.mp4", output_video=out,
        aspect_ratio="9:16", framing="blur_fit", captions_on=False,
        trim_start=1.0, trim_end=4.0, av_offset_s=0.0)
    assert ok is True
    assert len(captured) == 1
    cmd = captured[0]
    assert cmd.count("-i") == 1
    assert cmd.count("0:a?") == 1 and "1:a?" not in cmd
    assert cmd[cmd.index("-ss") + 1] == "1.0"


def test_render_clip_nonzero_offset_uses_dual_input(monkeypatch, tmp_path):
    from clipper import renderer
    captured = []
    monkeypatch.setattr(renderer.subprocess, "run",
                        _fake_ffmpeg_run_factory(captured))
    out = tmp_path / "out.mp4"
    ok = renderer.render_clip(
        input_video=tmp_path / "in.mp4", output_video=out,
        aspect_ratio="9:16", framing="blur_fit", captions_on=False,
        trim_start=1.0, trim_end=4.0, av_offset_s=2.5)
    assert ok is True
    assert len(captured) == 1
    cmd = captured[0]
    # two inputs of the same file; video window shifted, audio unshifted
    assert cmd.count("-i") == 2
    ss_values = [cmd[i + 1] for i, v in enumerate(cmd) if v == "-ss"]
    assert ss_values == ["3.5", "1.0"]
    assert "1:a?" in cmd and "0:a?" not in cmd
    # filtergraph still reads the video input
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert fc.startswith("[0:v]")


def test_render_clip_nonzero_offset_retry_also_dual_input(monkeypatch, tmp_path):
    """When the first ffmpeg call fails with subtitles on, the retry
    without subtitles must use the same dual-input layout."""
    from clipper import renderer
    import subprocess
    captured = []
    calls = {"n": 0}

    def flaky_run(cmd, **kwargs):
        captured.append(list(cmd))
        calls["n"] += 1
        if calls["n"] == 1:
            raise subprocess.CalledProcessError(1, cmd, stderr="ass failed")
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x" * 2000)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(renderer.subprocess, "run", flaky_run)
    out = tmp_path / "out.mp4"
    ok = renderer.render_clip(
        input_video=tmp_path / "in.mp4", output_video=out,
        aspect_ratio="9:16", framing="blur_fit",
        words=[{"word": "hi", "start": 1.1, "end": 1.5}],
        captions_on=True, hook_text="HOOK",
        trim_start=0.5, trim_end=3.5, av_offset_s=1.5)
    assert ok is True
    assert len(captured) == 2
    for cmd in captured:
        assert cmd.count("-i") == 2
        assert "1:a?" in cmd
        ss_values = [cmd[i + 1] for i, v in enumerate(cmd) if v == "-ss"]
        assert ss_values == ["2.0", "0.5"]
    # retry drops the ASS filter
    assert "ass=" not in captured[1][captured[1].index("-filter_complex") + 1]
