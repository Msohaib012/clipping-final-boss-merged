"""Session orchestration: manages pipeline sessions in data/out/session_N/."""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from .config import DATA_OUT_DIR, DATA_IN_DIR
from .downloader import download_or_load
from .transcriber import transcribe
from .candidate_picker import find_candidate_moments
from .cutter import cut_clip
from .ranker import rank_and_rename_clips
from .renderer import render_clip
from .ai import generate_clip_metadata
from .thumbnail import create_thumbnail, extract_poster_frame
from .games import detect as detect_game

def get_next_session_dir() -> Path:
    """Finds next available session directory e.g. data/out/session_1."""
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    idx = 1
    while (DATA_OUT_DIR / f"session_{idx}").exists():
        idx += 1
    s_dir = DATA_OUT_DIR / f"session_{idx}"
    s_dir.mkdir(parents=True, exist_ok=True)
    return s_dir

def save_session_data(session_dir: Path, data: Dict[str, Any]) -> None:
    session_file = session_dir / "session.json"
    session_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

def load_session_data(session_dir: Path) -> Optional[Dict[str, Any]]:
    session_file = session_dir / "session.json"
    if not session_file.exists():
        return None
    try:
        return json.loads(session_file.read_text(encoding="utf-8"))
    except Exception:
        return None

def process_video_pipeline(
    source: str,
    top_n: int = 5,
    min_len: Optional[float] = None,
    max_len: Optional[float] = None,
    length_range: Optional[str] = None,
    chat_events: Optional[List[Dict[str, Any]]] = None,
    progress_callback: Optional[Any] = None,
    preset: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes full pipeline:
    1. Download or load media
    2. Transcribe with word timestamps
    3. Find candidate moments respecting length constraints
    4. Cut candidate clips
    5. Generate unique AI metadata (title, caption, hashtags)
    6. Rank and rename clips (01_*.mp4)
    7. Generate poster frames & thumbnails
    """
    session_dir = get_next_session_dir()
    session_id = session_dir.name

    # Determine default length constraints
    if min_len is None or max_len is None:
        min_len = 20.0
        max_len = 90.0
        if not length_range:
            length_range = "Auto (< 90s)"
    elif not length_range:
        length_range = f"{int(min_len)}-{int(max_len)}s"

    def report_progress(stage: str, percent: int, msg: str):
        if progress_callback:
            try:
                # Primary consistent signature: def on_progress(pct: float, msg: str = "", **kw)
                progress_callback(float(percent), msg=msg, stage=stage)
            except TypeError:
                try:
                    progress_callback(stage, percent, msg)
                except TypeError:
                    try:
                        progress_callback(msg, percent)
                    except Exception:
                        pass
            except Exception:
                pass

    report_progress("download", 10, "Downloading or loading video source...")

    # 1. Download/Load
    media = download_or_load(source, session_id)
    source_video = media["file_path"]
    duration = media["duration"]

    # Detect game if present and set preset
    detected_game, _ = detect_game(media.get("title", ""), source)
    if preset is None and detected_game:
        preset = "gta6" if detected_game == "Grand Theft Auto VI" else "gaming"

    report_progress("transcribe", 25, "Transcribing audio locally with faster-whisper...")

    # 2. Transcribe
    transcript = transcribe(source_video)

    # Hindi/Urdu on local files: tiny garbles noisy game audio — re-
    # transcribe with the SMALL model (his clipper's ROMAN_MODEL) before
    # captioning. Remote sources skip this: autosubs (step 2b) are the
    # primary path there and replace the Whisper transcript anyway.
    if media.get("is_local", True) and transcript.get("language") in ("hi", "ur"):
        print("[transcribe] Hindi/Urdu detected — re-transcribing with "
              "small model", flush=True)
        transcript = transcribe(source_video, model_size="small")

    # 2b. Roman captions + YouTube auto-subs (merged Port 3)
    # Autosubs (when available) are cleaner than local Whisper for Hindi/
    # Urdu and already carry Roman display words + natural retiming;
    # Whisper words keep their real timings and are NEVER retimed.
    from .roman_captions import (
        apply_caption_script,
        autosubs_to_transcript,
        fetch_autosubs,
        load_gemini_key,
    )

    transcript_source = "whisper"
    caption_script = "latin"
    translit_method = "none"
    gemini_key = load_gemini_key()  # settings.gemini_api_key (.env), or None
    roman_mode = "gemini" if gemini_key else "local"

    if not media.get("is_local", True):
        try:
            subs_dir = session_dir / "subs"
            subs_dir.mkdir(parents=True, exist_ok=True)
            cues, subs_lang = fetch_autosubs(source, subs_dir)
        except Exception as _subs_exc:
            print(f"[roman] autosubs fetch failed ({_subs_exc}) — Whisper fallback")
            cues, subs_lang = None, None
        if cues:
            transcript = autosubs_to_transcript(
                cues, duration,
                caption_mode="auto",
                roman_mode=roman_mode,
                detected_lang=subs_lang or "hi",
                gemini_key=gemini_key,
            )
            transcript_source = "youtube-autosubs"
            caption_script = transcript.get("caption_script", "latin")
            translit_method = transcript.get("translit_method", "local")

    if transcript_source == "whisper":
        flat_words = [w for seg in transcript.get("segments", [])
                      for w in seg.get("words", [])]
        _, caption_script, translit_method = apply_caption_script(
            flat_words,
            transcript.get("language", "en"),
            caption_mode="auto",
            roman_mode=roman_mode,
            gemini_key=gemini_key,
        )
        # flat_words are the SAME dict objects as the segment words
        # (apply_caption_script mutates in place), so transcript segments
        # and candidate_picker both see the added "display" field.
        if flat_words:
            from .roman_captions import (
                gemini_romanize_words,
                retime_whisper_words,
            )
            # Hindi/Urdu: Gemini natural-Roman rewrite via the connected
            # helper (sentence-level, same pattern as his clipper). Sets
            # the final display words; segments that fail or mismatch
            # keep the local transliteration.
            if (transcript.get("language") in ("hi", "ur")
                    and caption_script == "latin"):
                _, n_rw, n_fb = gemini_romanize_words(
                    flat_words, transcript.get("segments", []),
                    transcript.get("language", "hi"))
                if n_rw:
                    translit_method = "gemini"
                print(f"[roman] gemini rewrite: {n_rw} segments rewritten, "
                      f"{n_fb} local fallback", flush=True)
            # Whisper words keep real timings — re-fit them to natural
            # pace ([0.18, 0.55]s) per segment, same clamp as autosubs
            # words (fixes multi-second highlights like UCHVA 2.84s).
            retime_whisper_words(flat_words, transcript.get("segments", []))

    report_progress("moments", 50, "Detecting hooks, loudness, and chat spikes...")

    # 3. Find Candidate Moments
    candidates = find_candidate_moments(
        video_path=source_video,
        duration=duration,
        transcript=transcript,
        chat_events=chat_events,
        target_count=top_n,
        min_clip_len=min_len,
        max_clip_len=max_len,
        preset=preset,
    )

    # A/V sync (merged Port 4): measure the video-behind-audio offset once
    # per source video (game-subtitle OCR vs Whisper word times). Falls
    # back to the per-channel carried-forward clock fix (seed) on failure.
    av_offset_s = 0.0
    try:
        from .av_align import (
            channel_key_for,
            load_clock_fix,
            measure_av_offset_from_transcript,
            save_clock_fix,
        )
        _av_key = channel_key_for(str(source_video))
        _av_seed = load_clock_fix(_av_key, 0.0)
        if candidates:
            _av_start, _av_end = float(candidates[0].start), float(candidates[0].end)
        else:
            _av_start, _av_end = 0.0, min(float(duration or 0.0), 120.0)
        av_offset_s, _av_n, _av_details = measure_av_offset_from_transcript(
            str(source_video), transcript, _av_start, _av_end, seed=_av_seed
        )
        if _av_n >= 3:
            save_clock_fix(_av_key, av_offset_s)
    except Exception as _av_exc:
        print(f"[av-align] measurement failed, using offset 0.0 ({_av_exc})")
        av_offset_s = 0.0

    report_progress("rank", 70, "Cutting candidate clips with FFmpeg...")

    # 4. Cut raw clips
    raw_clips = []
    used_titles = set()
    total_cands = max(1, len(candidates))
    for idx, cand in enumerate(candidates, start=1):
        # Skip candidates with no spoken words: metadata generation raises
        # on an empty transcript, and a silent segment must not kill the
        # whole session (e.g. a transient empty Whisper result under load).
        if not (cand.transcript_text or "").strip():
            print(f"[pipeline] skipping candidate #{idx} — no spoken words", flush=True)
            continue
        temp_name = f"raw_cand_{idx}.mp4"
        out_path = session_dir / temp_name
        cut_pct = 70 + int((idx / total_cands) * 12)
        report_progress("rank", cut_pct, f"Cutting clip #{idx} of {total_cands} and generating viral metadata...")
        success = cut_clip(source_video, out_path, cand.start, cand.end)
        if success:
            # Generate unique AI metadata (with safe timeout and fallback)
            meta = generate_clip_metadata(cand.transcript_text, idx, used_titles=used_titles, game=detected_game)
            # Roman-caption clip: the hook/title must not be Devanagari
            # (the 0-5s hook shows the title). Local transliteration —
            # a title is short, the mechanical path is fine here.
            if caption_script == "latin":
                from .roman_captions import to_roman_text
                meta["title"] = to_roman_text(
                    meta["title"], transcript.get("language", "en"))
            raw_clips.append({
                "id": f"clip_{idx}",
                "file_path": str(out_path),
                "start": cand.start,
                "end": cand.end,
                "duration": round(cand.duration, 2),
                "length_range": length_range,
                "hook_score": cand.hook_score,
                "chat_score": cand.chat_score,
                "emotion_score": cand.emotion_score,
                "story_score": cand.story_score,
                "trend_score": cand.trend_score,
                "loudness_score": cand.loudness_score,
                "title": meta["title"],
                "caption": meta["caption"],
                "hashtags": meta["hashtags"],
                "transcript_text": cand.transcript_text,
                # Words are re-based to clip-relative time: candidate words
                # carry ABSOLUTE source times, but the cut clip starts at
                # 0 — without this, ASS captions (trim_offset=0) and the
                # editor live preview never show. New dicts ({**w}) so the
                # transcript's word objects are not mutated.
                "words": [
                    {**w,
                     "start": round(w["start"] - cand.start, 3),
                     "end": round(w["end"] - cand.start, 3)}
                    for w in cand.words
                ],
                "av_offset_s": av_offset_s,
                "aspect": "9:16",
                "framing": "blur_fit",
                "trim_start": 0.0,
                "trim_end": round(cand.duration, 2),
                "captions_on": True,
                "caption_style": "pop",
                "hook_text": meta["title"],
            })

    report_progress("rank", 84, "Scoring and ranking clips...")

    # 5. Rank and rename
    has_chat = bool(chat_events and len(chat_events) > 0)
    ranked_clips = rank_and_rename_clips(raw_clips, session_dir, has_chat=has_chat)

    report_progress("render", 88, "Generating 9:16 poster frames and thumbnails...")

    # 6. Render preview thumbnails and poster frames
    total_ranked = max(1, len(ranked_clips))
    for r_idx, clip in enumerate(ranked_clips, start=1):
        thumb_pct = 88 + int((r_idx / total_ranked) * 10)
        report_progress("render", thumb_pct, f"Rendering poster frame for clip #{clip['rank']}...")
        stem = Path(clip["file_path"]).stem
        poster_name = f"{stem}_poster.jpg"
        poster_path = session_dir / poster_name
        # Extract clean 9:16 vertical poster frame from middle of clip
        extract_poster_frame(video_path=Path(clip["file_path"]), output_path=poster_path)

        # Standard thumbnail overlay as backup
        thumb_name = f"{stem}_thumb.jpg"
        thumb_path = session_dir / thumb_name
        create_thumbnail(
            video_path=Path(clip["file_path"]),
            title=clip["title"],
            output_path=thumb_path,
            aspect_ratio="9:16",
            use_ai_image=False,
        )

        clip["poster_url"] = f"/api/media/{session_id}/{poster_name}"
        clip["thumbnail_url"] = f"/api/media/{session_id}/{poster_name}" if poster_path.exists() else f"/api/media/{session_id}/{thumb_name}"
        clip["video_url"] = f"/api/media/{session_id}/{clip['filename']}"

    report_progress("complete", 100, "Complete! Ranked short clips ready.")

    session_data = {
        "id": session_id,
        "title": media["title"],
        "duration": duration,
        "uploader": media["uploader"],
        "source": source,
        "length_range": length_range,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "has_chat": has_chat,
        "caption_script": caption_script,
        "translit_method": translit_method,
        "transcript_source": transcript_source,
        "language": transcript.get("language", "en"),
        "av_offset_s": av_offset_s,
        "clips": ranked_clips,
    }

    save_session_data(session_dir, session_data)

    if progress_callback:
        report_progress("complete", 100, "Complete! Ranked short clips ready.")

    return session_data
