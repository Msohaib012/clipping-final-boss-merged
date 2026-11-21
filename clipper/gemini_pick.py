"""Gemini wording-based moment scoring (optional auto-shorts quality gate).

Port of the donor ``gta6-live-clipper/gemini_pick.py`` into Clipping Final
Boss. The judge is the WORDING (what was actually said): a funny moment,
an emotional moment, or an incident people want to watch — NOT just
loudness. A loud-but-boring segment must NOT become a clip.

Differences from the donor:
- Auth is direct REST with ``httpx`` using CFB's ``settings.gemini_api_key``
  (same URL pattern as ``clipper.ai.generate_with_gemini``). It never
  shells out to a helper script and never reads a key from anywhere else.
- The transcript is passed in as a dict (``{"segments": [...]}``) or a
  plain segments list, not a file path.

Safety contract (the pipeline must never block on this):
- No key -> return None (caller keeps the candidate).
- <10 transcript words -> return None (too little speech to judge).
- Any API/network/parse failure -> return None (caller keeps the candidate).
- Never raise. Never print or log the API key.
- Model: gemini-flash-lite-latest (fast, ~1s; works with his key).
  gemini-2.5-flash/-lite 404 on his key, gemini-3.5-flash is severely
  slow — do not switch models without re-testing on his key.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

logger = logging.getLogger("clipper.gemini_pick")

MODEL = "gemini-flash-lite-latest"
GENERATE_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
                "%s:generateContent")
MAX_TRANSCRIPT_CHARS = 3500
CATEGORIES = ("funny", "emotional", "incident", "shocking", "viral", "none")

# Veto threshold: a candidate Gemini scores below this is dropped by the
# gate — unless it was already triggered by loudness/chat (see
# candidate_picker.py). Mirrors the donor's "only score >= 7 renders".
GEMINI_MIN_SCORE = 7.0

_PROMPT = """You are a Shorts editor for a GTA 6 gaming livestream. Below is
the transcript of one {dur:.0f}-second live segment (timestamps in seconds),
plus the loudest moment. Find the SINGLE best 30-45 second moment worth
turning into a vertical Short — something viewers would actually want to
watch: funny, emotional, shocking, or an incident (a fail, a clutch, a
rage moment, a surprise, a story being told).

Score it 0-10. Be strict: 0-3 boring / routine talk, 4-6 decent but not
special, 7-8 genuinely good Short material, 9-10 exceptional /
viral-worthy. Only score 7+ when it is truly funny, emotional, or an
incident. If nothing is worth a Short, score it below 7 and say so.

Reply with JSON ONLY, no other text:
{{"score": <0-10>, "category": "funny|emotional|incident|shocking|viral|none", "reason": "<one short sentence>", "quote": "<the punchiest line>", "start": <seconds>, "end": <seconds>}}

Loudest moment: ~{loud_t:.0f}s ({loud_db} dB).
Transcript:
{transcript}"""


def _transcript_text(transcript: Union[Dict[str, Any], List[Dict[str, Any]], None],
                     max_chars: int = MAX_TRANSCRIPT_CHARS) -> Tuple[str, float]:
    """Timestamped lines: '[12.3-18.1s] text'. Truncated to max_chars.

    Accepts a transcript dict (``{"segments": [...], "duration": ...}``)
    or a bare segments list.
    """
    try:
        if isinstance(transcript, dict):
            segments = transcript.get("segments") or []
            duration = float(transcript.get("duration") or 0.0)
        elif isinstance(transcript, list):
            segments = transcript
            duration = 0.0
        else:
            return "", 0.0
        lines = []
        for s in segments:
            t = (s.get("text") or "").strip()
            if t:
                lines.append("[%.1f-%.1fs] %s"
                             % (s.get("start", 0), s.get("end", 0), t))
        text = "\n".join(lines)
        return text[:max_chars], duration
    except Exception:
        return "", 0.0


def _parse_json(text):
    """Parse Gemini's reply: strip code fences, take the first {...}."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _ask_gemini(prompt: str, api_key: str, model: str = MODEL,
                timeout: float = 30) -> Optional[str]:
    """Direct REST call to Gemini. Returns the reply text, or None on any
    failure. Never raises, never logs the key."""
    url = (GENERATE_URL % model) + "?key=" + api_key
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
            )
        if resp.status_code != 200:
            logger.warning("[gemini] score_moment: HTTP %s — gate fallback",
                           resp.status_code)
            return None
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        logger.warning("[gemini] score_moment failed (%s) — gate fallback",
                       type(e).__name__)
        return None


def score_moment(transcript: Union[Dict[str, Any], List[Dict[str, Any]], None],
                 duration: float = 0.0,
                 loud_t: float = 0.0,
                 loud_db: Optional[float] = None,
                 api_key: Optional[str] = None,
                 model: str = MODEL,
                 timeout: float = 30) -> Optional[Dict[str, Any]]:
    """Score one transcribed segment with Gemini. Returns a dict:
    {"score": float 0-10, "category": str, "reason": str, "quote": str,
     "start": float|None, "end": float|None}
    or None when there is no key, too little speech, or any failure
    (the caller then keeps the candidate / falls back to loudness).
    Never raises, never logs the key."""
    try:
        if not api_key:
            return None  # no key — gate cannot judge, caller keeps candidate
        text, tr_dur = _transcript_text(transcript)
        if not text or len(text.split()) < 10:
            return None  # too little speech to judge wording
        dur = duration or tr_dur or 180.0
        prompt = _PROMPT.format(
            dur=dur, loud_t=loud_t,
            loud_db=loud_db if loud_db is not None else "?",
            transcript=text)
        reply = _ask_gemini(prompt, api_key, model=model, timeout=timeout)
        if not reply:
            return None
        out = _parse_json(reply)
        if not isinstance(out, dict):
            logger.warning("[gemini] score_moment: unparsable reply — "
                           "gate fallback")
            return None
        try:
            score = float(out.get("score"))
        except Exception:
            logger.warning("[gemini] score_moment: no numeric score — "
                           "gate fallback")
            return None
        score = max(0.0, min(10.0, score))
        cat = str(out.get("category") or "none").lower().strip()
        if cat not in CATEGORIES:
            cat = "none"
        start = end = None
        try:
            s0, e0 = float(out.get("start")), float(out.get("end"))
            if 0 <= s0 < e0 and (dur <= 0 or s0 < dur):
                start, end = s0, min(e0, dur) if dur > 0 else e0
        except Exception:
            start = end = None
        return {"score": round(score, 1), "category": cat,
                "reason": str(out.get("reason") or "")[:160],
                "quote": str(out.get("quote") or "")[:160],
                "start": start, "end": end}
    except Exception as e:
        logger.warning("[gemini] score_moment failed (%s) — gate fallback",
                       type(e).__name__)
        return None
