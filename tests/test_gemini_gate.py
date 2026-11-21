"""Tests for the optional Gemini wording gate (Port 5).

score_moment is MONKEYPATCHED in the picker tests — no network is hit.
The real score_moment is only exercised for its no-key / too-little-speech
early returns, which never reach the network.
"""

import numpy as np
import pytest

from clipper import candidate_picker
from clipper.candidate_picker import find_candidate_moments
from clipper.config import settings
from clipper.gemini_pick import GEMINI_MIN_SCORE, score_moment


def _segments(duration=60.0, seg_len=5.0):
    """Transcript segments with plenty of words, spread over duration."""
    segs = []
    t = 0.0
    i = 0
    while t < duration:
        end = min(duration, t + seg_len)
        segs.append({
            "start": t,
            "end": end,
            "text": f"This is segment number {i} where something really "
                    f"interesting and unexpected happens on the stream today.",
        })
        t = end
        i += 1
    return segs


def _loudness(monkeypatch, value):
    """Force analyze_audio_loudness to a constant level (0-100)."""
    monkeypatch.setattr(
        candidate_picker, "analyze_audio_loudness",
        lambda video_path, duration: np.full(
            max(1, int(np.ceil(duration))), float(value), dtype=np.float32),
    )


@pytest.fixture(autouse=True)
def _gate_off(monkeypatch):
    """Gate flag is off unless a test turns it on; always restored."""
    monkeypatch.setattr(settings, "gemini_gate_enabled", False)
    yield
    monkeypatch.setattr(settings, "gemini_gate_enabled", False)


def _run(monkeypatch, tmp_path, loudness_value=50.0, chat_events=None):
    # No real video needed: loudness analysis is monkeypatched, so the
    # path is never opened. (The shared test_video_path fixture needs a
    # data/real_speech_5min_final.mp4 that this merged copy does not ship.)
    _loudness(monkeypatch, loudness_value)
    transcript = {"segments": _segments(), "duration": 60.0}
    return find_candidate_moments(
        tmp_path / "dummy.mp4", 60.0, transcript=transcript,
        chat_events=chat_events, target_count=2,
        min_clip_len=20.0, max_clip_len=60.0,
    )


def test_gate_off_by_default_leaves_candidates_unchanged(monkeypatch, tmp_path):
    # Even a score_moment that would veto everything must be ignored.
    monkeypatch.setattr(
        candidate_picker, "score_moment",
        lambda **kw: {"score": 1.0, "category": "none", "reason": "boring",
                      "quote": "", "start": None, "end": None},
    )
    assert settings.gemini_gate_enabled is False
    result = _run(monkeypatch, tmp_path)
    assert len(result) == 2
    assert all(c.gemini_score is None for c in result)
    assert all(c.gemini_category == "none" for c in result)
    assert all(c.gemini_reason == "" for c in result)


def test_gate_veto_drops_low_score_untriggered_candidate(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "gemini_gate_enabled", True)
    monkeypatch.setattr(
        candidate_picker, "score_moment",
        lambda **kw: {"score": 4.0, "category": "none",
                      "reason": "routine talk, nothing special",
                      "quote": "", "start": None, "end": None},
    )
    # Loudness 50 (< 70) and no chat (0 < 60): nothing triggers, so the
    # sub-7.0 wording score vetoes every selected candidate.
    result = _run(monkeypatch, tmp_path, loudness_value=50.0)
    assert result == []


def test_gate_keeps_loudness_triggered_candidate(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "gemini_gate_enabled", True)
    monkeypatch.setattr(
        candidate_picker, "score_moment",
        lambda **kw: {"score": 4.0, "category": "none",
                      "reason": "routine talk, nothing special",
                      "quote": "", "start": None, "end": None},
    )
    # Loudness 85 >= 70: loudness-triggered, so the low wording score
    # does NOT veto it — the audience reaction is the evidence.
    result = _run(monkeypatch, tmp_path, loudness_value=85.0)
    assert len(result) == 2
    assert all(c.loudness_score >= 70 for c in result)
    assert all(c.gemini_score == 4.0 for c in result)
    assert all(c.gemini_category == "none" for c in result)


def test_gate_keeps_chat_triggered_candidate(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "gemini_gate_enabled", True)
    monkeypatch.setattr(
        candidate_picker, "score_moment",
        lambda **kw: {"score": 4.0, "category": "none",
                      "reason": "routine talk, nothing special",
                      "quote": "", "start": None, "end": None},
    )
    # Uniform chat across the whole duration: every window's chat_score
    # is 100 (>= 60), so all candidates are chat-triggered and kept.
    chat_events = [{"time": t} for t in range(60) for _ in range(5)]
    result = _run(monkeypatch, tmp_path, loudness_value=50.0,
                  chat_events=chat_events)
    assert len(result) == 2
    assert all(c.chat_score >= 60 for c in result)
    assert all(c.gemini_score == 4.0 for c in result)


def test_gate_failure_keeps_candidate(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "gemini_gate_enabled", True)

    def _boom(**kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(candidate_picker, "score_moment", _boom)
    result = _run(monkeypatch, tmp_path, loudness_value=50.0)
    assert len(result) == 2
    assert all(c.gemini_score is None for c in result)


def test_score_moment_returns_none_without_key():
    # Real function, no key: early return, no network hit.
    transcript = {"segments": _segments(), "duration": 60.0}
    assert score_moment(transcript, api_key=None) is None
    assert score_moment(transcript, api_key="") is None


def test_score_moment_returns_none_on_too_little_speech():
    # Real function, key present but <10 words: early return, no network.
    tiny = {"segments": [{"start": 0.0, "end": 2.0, "text": "hello world"}],
            "duration": 5.0}
    assert score_moment(tiny, api_key="fake-key-never-sent") is None
    # Same via a bare segments list.
    assert score_moment(tiny["segments"], api_key="fake-key-never-sent") is None


def test_gemini_min_score_threshold():
    assert GEMINI_MIN_SCORE == 7.0
