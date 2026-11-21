"""Port 3 tests: Roman captions + word retiming (clipper.roman_captions).

Ported behaviour from the gta6-live-clipper donor (transliterate.py +
vod.py). No network, no Gemini key, no model downloads — the local
(vendored _ck) path is exercised; indic-transliteration is NOT installed
in the merged venv, so devanagari_to_roman runs on the stdlib engine.
"""

from pathlib import Path

import pytest

from clipper.roman_captions import (
    ROMAN_OVERRIDES,
    _natural_word_dur,
    _parse_vtt,
    _retime_words_natural,
    apply_caption_script,
    autosubs_to_transcript,
    devanagari_to_roman,
    urdu_to_roman,
)
from clipper.renderer import generate_ass_subtitles


# --- devanagari_to_roman -------------------------------------------------

def test_devanagari_mein():
    assert devanagari_to_roman("मैं") == "mein"


def test_devanagari_nahin():
    assert devanagari_to_roman("नहीं") == "nahin"


def test_devanagari_karna():
    assert devanagari_to_roman("करना") == "karna"


def test_devanagari_sentence_keeps_punctuation_and_latin():
    out = devanagari_to_roman("मैं जा रहा हूँ, bro!")
    assert out == "mein ja raha hoon, bro!"
    assert all(ord(c) < 0x250 for c in out if c.isalpha())


# --- urdu_to_roman ---------------------------------------------------------

def test_urdu_mein():
    assert urdu_to_roman("میں") == "mein"


def test_urdu_gaye():
    assert urdu_to_roman("گئے") == "gaye"


def test_urdu_sentence():
    assert urdu_to_roman("میں جا رہا ہوں") == "mein ja raha hoon"


# --- ROMAN_OVERRIDES -------------------------------------------------------

def test_roman_override_police():
    # पुलिस -> mechanical "pulis" -> override "police"
    assert devanagari_to_roman("पुलिस") == "police"
    assert ROMAN_OVERRIDES["pulis"] == "police"


def test_roman_override_karna():
    assert ROMAN_OVERRIDES["karana"] == "karna"


# --- _natural_word_dur -----------------------------------------------------

def test_natural_word_dur_floor():
    # 1 char -> 0.055 + 0.08 = 0.135, clamped up to the 0.18 floor
    assert _natural_word_dur("a") == pytest.approx(0.18)
    assert _natural_word_dur("hi") == pytest.approx(0.19)


def test_natural_word_dur_mid():
    # "hello" (5 chars) -> 0.055*5 + 0.08 = 0.355
    assert _natural_word_dur("hello") == pytest.approx(0.355)


def test_natural_word_dur_cap():
    # 20 chars -> 1.18, clamped down to the 0.55 cap
    assert _natural_word_dur("x" * 20) == pytest.approx(0.55)
    assert _natural_word_dur("x" * 9) == pytest.approx(0.55)  # 0.575 -> cap


def test_natural_word_dur_strips_punctuation():
    assert _natural_word_dur("hello!") == _natural_word_dur("hello")


# --- _retime_words_natural -------------------------------------------------

def _cue_words(texts, cue_start=10.0, cue_end=13.0, cue=0):
    return [
        {"word": t, "display": t, "start": 0.0, "end": 0.0,
         "_cue": cue, "_cue_start": cue_start, "_cue_end": cue_end}
        for t in texts
    ]


def test_retime_no_overlaps_and_min_duration():
    words = _retime_words_natural(_cue_words(["mein", "ja", "raha", "hoon"]))
    assert len(words) == 4
    for i, w in enumerate(words):
        assert w["end"] - w["start"] >= 0.08
        if i:
            assert w["start"] >= words[i - 1]["end"]  # no overlap
    assert words[0]["start"] == pytest.approx(10.0)


def test_retime_durations_come_from_display_length():
    # Long display word gets a longer slot than a short one (same cue).
    words = _retime_words_natural(
        _cue_words(["a", "supercalifragilistic"], cue_start=0.0, cue_end=10.0))
    assert words[0]["end"] - words[0]["start"] == pytest.approx(0.18)
    assert words[1]["end"] - words[1]["start"] == pytest.approx(0.55)


def test_retime_compresses_to_fit_cue_span():
    # Natural total (4 x ~0.3s) exceeds a 0.5s cue: compressed, still >=0.08s.
    words = _retime_words_natural(
        _cue_words(["mein", "ja", "raha", "hoon"], cue_start=5.0, cue_end=5.5))
    assert words[-1]["end"] <= 5.5 + 0.001
    for w in words:
        assert w["end"] - w["start"] >= 0.08


# --- _parse_vtt + autosubs_to_transcript -----------------------------------

_VTT = """WEBVTT

00:00:01.000 --> 00:00:03.000
मैं जा रहा हूँ

00:00:02.000 --> 00:00:04.000
मैं जा रहा हूँ और

00:00:04.000 --> 00:00:06.000
और देखो ये क्या है
"""


def test_parse_vtt_merges_rolling_duplicates(tmp_path):
    p = tmp_path / "subs.hi.vtt"
    p.write_text(_VTT, encoding="utf-8")
    cues = _parse_vtt(str(p))
    # Donor merge rule: a cue whose text contains the previous cue's text
    # REPLACES it (cue 2 swallows cue 1), so only 2 cues survive.
    assert len(cues) == 2
    assert cues[0]["text"] == "मैं जा रहा हूँ और"
    assert cues[0]["start"] == pytest.approx(2.0)
    assert cues[1]["text"] == "और देखो ये क्या है"


def test_autosubs_suffix_prefix_dedup():
    cues = [
        {"start": 1.0, "end": 3.0, "text": "hello world"},
        {"start": 2.0, "end": 4.0, "text": "world this is"},
        {"start": 3.0, "end": 5.0, "text": "this is a test"},
    ]
    tr = autosubs_to_transcript(cues, duration=5.0, detected_lang="en")
    words = [w["word"] for w in tr["words"]]
    assert words == ["hello", "world", "this", "is", "a", "test"]
    assert tr["full_text"] == "hello world this is a test"
    assert tr["language"] == "en"
    # segments carry their words (CFB shape, for candidate_picker)
    assert tr["segments"][0]["words"][0]["word"] == "hello"
    assert all("_cue" not in w for w in tr["words"])


def test_autosubs_hindi_words_get_roman_display():
    cues = [{"start": 1.0, "end": 3.0, "text": "मैं नहीं जा रहा"}]
    tr = autosubs_to_transcript(cues, duration=3.0, detected_lang="hi")
    assert [w["display"] for w in tr["words"]] == ["mein", "nahin", "ja", "raha"]
    assert tr["caption_script"] == "latin"
    assert tr["translit_method"] == "local"
    # retimed: consecutive, non-overlapping, starting at the cue start
    assert tr["words"][0]["start"] == pytest.approx(1.0)
    for a, b in zip(tr["words"], tr["words"][1:]):
        assert b["start"] >= a["end"]


# --- apply_caption_script ---------------------------------------------------

def test_apply_caption_script_hindi_to_roman():
    words = [
        {"word": "मैं", "start": 0.0, "end": 0.5},
        {"word": "नहीं", "start": 0.5, "end": 1.0},
        {"word": "करना", "start": 1.0, "end": 1.5},
    ]
    out, script, method = apply_caption_script(words, "hi")
    assert [w["display"] for w in out] == ["mein", "nahin", "karna"]
    assert script == "latin"
    assert method == "local"


def test_apply_caption_script_urdu_to_roman():
    words = [{"word": "میں", "start": 0.0, "end": 0.5},
             {"word": "گئے", "start": 0.5, "end": 1.0}]
    out, script, method = apply_caption_script(words, "ur")
    assert [w["display"] for w in out] == ["mein", "gaye"]
    assert script == "latin"


def test_apply_caption_script_english_as_is():
    words = [{"word": "hello", "start": 0.0, "end": 0.5},
             {"word": "world!", "start": 0.5, "end": 1.0}]
    out, script, method = apply_caption_script(words, "en")
    assert [w["display"] for w in out] == ["hello", "world!"]
    assert script == "latin"
    assert method == "none"


def test_apply_caption_script_explicit_hi_roman_mode():
    words = [{"word": "पुलिस", "start": 0.0, "end": 0.5}]
    out, script, method = apply_caption_script(words, "en", caption_mode="hi-roman")
    assert out[0]["display"] == "police"
    assert script == "latin"


# --- renderer honours display ----------------------------------------------

def test_generate_ass_uses_display_text(tmp_path):
    ass = tmp_path / "caps.ass"
    words = [
        {"word": "मैं", "display": "mein", "start": 0.0, "end": 0.4},
        {"word": "नहीं", "display": "nahin", "start": 0.4, "end": 0.9},
    ]
    generate_ass_subtitles(words, ass)
    text = ass.read_text(encoding="utf-8")
    # The default "pop" style uppercases (text_transform=uppercase),
    # so the Roman display text appears uppercased — and no Devanagari.
    assert "MEIN" in text and "NAHIN" in text
    assert "मैं" not in text and "नहीं" not in text


def test_generate_ass_falls_back_to_word_without_display(tmp_path):
    ass = tmp_path / "caps.ass"
    words = [{"word": "hello", "start": 0.0, "end": 0.4}]
    generate_ass_subtitles(words, ass)
    # "pop" style uppercases the rendered word.
    assert "HELLO" in ass.read_text(encoding="utf-8")
