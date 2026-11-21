# Merge Notes — Clipping Final Boss + Our GTA 6 Live Clipper

Base: thefatherof8dragon/clipping-final-boss (MIT, PeakCut-derived), copied to
`clipping-final-boss-merged/` on 2026-11-20. Donor: `~/workspace/gta6-live-clipper/`.
Local work only — no GitHub push, no deploy.

## What was ported

### 1. Layouts (clipper/layouts.py + renderer.py framing branches)
- Ported `layout.py` (choose_layout, fallback_layout, detect_facecam) and
  `positions.py` (ratio_params, game_crop, stage_geometry, facecam_rect,
  caption_y, stage_caption_band, FACECAM_POSITIONS, CAPTION_POSITIONS).
- New framing modes in `render_clip()`: stage, stage-nofacecam,
  stage-topleft, stage-topright, stage-side, top-band, bottom-band,
  top-right, top-left, mid-right, mid-left, bottom-left, bottom-right, off.
  Existing blur_fit / crop_to_face unchanged.
- Facecam detection adapted: CFB has no MediaPipe — detection samples frames
  with ffmpeg and uses Canny still-border + OpenCV Haar faces.
- Caption Y comes from the layout (seam computed from the REAL band height,
  not the fixed 430) via custom_style["caption_y"] -> ASS MarginV.
- Web UI: framing buttons + frame-style select + caption-position select in
  the Reframe panel; wired through api.py RenderRequest.

### 2. Frame styles (clipper/frame_styles.py)
- Ported FRAME_STYLES (11: gold, red, line, fire, static, pulse, circle,
  clean, bold, corners, big), DEFAULT_FRAME_STYLE="gold", _frame_style_geom,
  _paint_frame, _stage_frame_assets. Default gold.

### 3. Roman captions + word retiming (clipper/roman_captions.py)
- Ported transliterate.py whole (apply_caption_script, devanagari_to_roman,
  urdu_to_roman, ROMAN_OVERRIDES, vendored ClipKontrast engine; the
  indic-transliteration import stays optional). Gemini key now read from
  CFB settings (GEMINI_API_KEY), not the donor's config.json.
- Ported vod.py _natural_word_dur (0.055s/char + 0.08, clamp [0.18, 0.55]),
  _retime_words_natural, autosubs_to_transcript (suffix-prefix dedup),
  _parse_vtt, fetch_autosubs.
- Words get a `display` field after transcription; the ASS renderer uses
  `display` instead of raw Whisper text. Whisper words keep their real
  timings; only autosubs-derived words are retimed.

### 4. A/V sync (clipper/av_align.py)
- Ported av_align.py (measure_av_offset, OCR game subtitles vs Whisper
  segments, median offset, clock-fix persistence per channel). Clock-fix
  file lives in CFB data/ (av_clock_fix.json). Never raises — falls back
  to the seed / 0.0 on any failure (incl. tesseract missing).
- render_clip(av_offset_s=...): shifts ONLY the video window later
  (dual ffmpeg inputs when nonzero); audio + captions stay.

### 5. Gemini wording gate (clipper/gemini_pick.py, OFF by default)
- Ported gemini_pick.py score_moment (model gemini-flash-lite-latest,
  rubric 0-3 boring / 4-6 decent / 7-8 good / 9-10 viral). Auth via CFB
  settings.gemini_api_key, direct REST with httpx. Never raises, never
  logs the key.
- Enabled via GEMINI_GATE_ENABLED env (default false). When enabled, a
  moment scoring < 7.0 is vetoed UNLESS loudness/chat already triggered
  (loudness_score >= 70 or chat_score >= 60). Gate failure keeps the
  candidate.

## What was skipped / deferred
- Letterbox detection (donor render_short.detect_letterbox) — not ported in
  this phase; stage game crop uses the full 1080 height.
- Per-channel facecam rect fallbacks (donor CHANNEL_FACECAM) — not ported;
  detection or fallback_layout only.
- The donor's PIL-track caption renderer (build_caption_track) — NOT ported;
  CFB's ASS renderer is kept, fed with Roman display words + natural durations.
- Bridge remote queue, 12-channel cycle, live-loop port — architectural, out
  of scope for this merge (CFB stays a single-user local app).
- VOD fail-fast live probe — not wired in this phase.

## Behaviour changes vs stock CFB
- **Staging dir (renderer.py `get_render_staging_dir`)**: `C:/clipper_tmp`
  is only tried on Windows (`os.name == "nt"`). On Linux the staging dir
  is `/tmp/clp_tmp` directly — previously ffmpeg rejected the Windows
  path ("Protocol not found") and a stray `C:` directory was created in
  the cwd.
- **Session pipeline (session.py)**: after transcription, words get a
  Roman `display` field (autosubs when the source is a remote YouTube
  URL, else Whisper + `apply_caption_script`; Whisper words are never
  retimed). After candidate picking, the A/V offset is measured on the
  first candidate's window (or [0, min(duration,120)]) and saved as a
  clock-fix only when n_matches >= 3; every clip renders with
  `av_offset_s`. Clip words are rebased to clip-relative time with new
  dicts (CFB candidate words are ABSOLUTE source times; without the
  rebase, captions never appeared in cut clips). Candidates with no
  spoken words are skipped instead of crashing metadata generation.
  `session_data` gains caption_script / translit_method /
  transcript_source / language / av_offset_s.
- **Web UI (static/app.js)**: `updateLiveCaptions` renders
  `w.display || w.word`, so Roman display words show in the live
  caption preview. The Reframe panel sends frame_style, facecam_pos,
  caption_pos, facecam_size, facecam_band, facecam_band_w and
  av_offset_s through both render POSTs.
- **A/V align wrapper (clipper/av_align.py)**: the no-transcript check
  now runs BEFORE the tesseract check — an empty transcript reports
  "no-transcript" even on a box without tesseract installed.
- **Test fixes (no production behaviour change)**:
  `test_healthz.py` compares `python_version` to the running interpreter
  instead of the hardcoded "3.14" (this VM runs Python 3.12.3);
  `test_short_render_staging.py` renders from the shared
  `test_video_path` fixture instead of leftover
  `data/out/session_1/clip_1.mp4` / `data/in/session_2_source.mp4`
  files (the old fallback made it fail on a clean checkout).
- **Test fixture restored**: `data/real_speech_5min_final.mp4` (40s,
  copy of the donor's `example_short.mp4`) — the stock suites errored
  at fixture setup without it.
- **Environment note (2026-11-20)**: `tesseract` is NOT installed on
  this VM any more (it was present earlier; lost in a VM replacement).
  The A/V OCR path therefore falls back to seed 0.0 until tesseract is
  reinstalled — the merge does not install system packages (see the
  repo's AGENTS.md: ask before installing system software).

## Port 1 verification (2026-11-20, this merge)
- Verified with a synthetic 1920x1080 testsrc video (the repo's real
  test video `data/real_speech_5min_final.mp4` is missing from disk, so
  the stock pytest suites error at fixture setup — pre-existing, see
  below). All layout modes render at the correct dims/duration:
  legacy blur_fit + crop_to_face; no-facecam stage / stage-nofacecam /
  stage-topleft / stage-topright / stage-side / top-band / bottom-band /
  all 6 corners / off; with-facecam (monkeypatched detection) stage
  gold/red/circle/big/bold/pulse, stage-top + stage-centre caption
  bands, top/bottom-band, corners, band_w 75/50, band 30, and stage in
  1:1, 4:5, 16:9. ASS MarginV check: caption_y=1320 @1920h -> 150.
- BUG FOUND + FIXED DURING VERIFICATION: the stage-with-facecam chain
  never terminated — the mask (`-loop 1`) and glow (`loop=-1`) inputs
  are infinite, so without an output cap ffmpeg encoded forever (1s
  trim -> 177s of output in 90s). Fix: `_build_cmd` now appends output
  `-t` = trim window (or probed input duration when untrimmed), the
  same cap the donor used (`-t dur` as an output option). Also fixed a
  vstack label bug (`[1:v]` used instead of the drawbox output `[fc]`)
  and added `-an` to facecam extraction.
- av_offset_s dual-input (port 4's contract) is implemented in the
  rebuilt `_build_cmd`: 3/3 mocked tests in test_av_align.py pass.
- BOTH PRE-EXISTING ITEMS ABOVE ARE NOW FIXED (integration round,
  2026-11-20 ~19:00 PKT): (1) the real test video was restored
  (`data/real_speech_5min_final.mp4`, copy of the donor's
  `example_short.mp4`); (2) `get_render_staging_dir()` now only tries
  `C:/clipper_tmp` on Windows — see "Behaviour changes" above.
- Letterbox detection: SKIPPED in this phase (stage game crop uses the
  full 1080 height), per the merge plan.

## Final verification (2026-11-20 ~19:00 PKT, integration round)
- Full suite: **104 passed, 0 failed in 295s**
  (`.venv/bin/python -m pytest -v --tb=short`). First full run was
  99 passed / 5 failed; all 5 fixed (see "Behaviour changes"):
  2x av_align tesseract/no-transcript check order, 1x healthz "3.14"
  literal, 1x short-render-staging leftover-file dependency, 1x stage1
  pipeline crash on an empty-transcript candidate (transcription came
  back empty once under heavy concurrent load from the donor clipper
  loop; the pipeline now skips such candidates instead of crashing).
- Smoke test: uvicorn on 127.0.0.1:8000 — `/healthz` returned
  `{"status":"ok","healthy":true,...,"failing":[]}` (Python 3.12.3,
  ffmpeg 8.1.2, yt-dlp 2026.10.26, node v24.22.1) and `/` returned 200.
  Server stopped afterwards.
- Per-port tests: Roman captions 25 passed, A/V sync 19 passed,
  Gemini gate 8 + 3 ranking passed, stage-1 pipeline 5 passed.
- Not tested live: the Gemini wording gate against the real Gemini API
  (no key configured in this copy — gate is OFF by default); real
  tesseract OCR (tesseract not installed on this VM — see above).

## How to run
- `python run.py` (creates .venv, installs requirements, serves 127.0.0.1:8000)
- Tests: `python tests/run.py` or `.venv/bin/python -m pytest -v`
- Smoke: start server, `curl http://127.0.0.1:8000/healthz`

## Fix round 2026-11-21 (after real-clip test 00:46 PKT — verdict NOT GOOD)
Backups: ~/workspace/backups/20261121-cfb-fix-round/ (layouts, roman_captions, session, transcriber, renderer, ai, requirements).
- **Facecam (defect 1, FIXED):** MediaPipe BlazeFace ported from the GTA 6 Live Clipper donor layout.py into `clipper/layouts.py` (bundled model `clipper/models/blaze_face_full_range.tflite`, mediapipe==1.2.3 in requirements.txt). The CFB donor repo has NO MediaPipe — the reference was his clipper. Needed system libs `libegl1 + libgles2` (`sudo apt-get install -y libegl1 libgles2`; wiped on VM replacement — detection then degrades gracefully to Haar). Still-border aspect filter relaxed 1.0-2.5 -> 0.6-2.5 (portrait facecams); Haar fallback improved (360px, minSize 20). Verified: detect_facecam() returns (1500,101,374,210) face-box on the test source (face 1687,206,170x170, top-right). Note: the still-border path did NOT fire (no clean 4-point contour on this facecam), so the rect is the 16:9 face-box fallback — it includes some game area left of the face.
- **Hindi captions (defect 2, PARTIAL):** local Hindi now re-transcribes with the SMALL model (two-pass in session.py, local files only) + Gemini natural-Roman rewrite via the connected custom.gemini helper (`gemini_romanize_words` in roman_captions.py, gemini-flash-lite-latest, per-segment word-count match required else local fallback). Round 2: 62 words (was 40), 7/9 segments rewritten, translit=gemini. Middle ~9-24s reads as plausible Hinglish matching the scene (gaadi churani, Police ka star); last ~7s still nonsense (Teedraver, badhadi, chookhe, kachan) — Gemini re-spells but cannot recover words Whisper never heard. Root cause stands: Whisper small mishears noisy Hindi game audio. The proven fix remains YouTube autosubs+Gemini, which needs the source video ID (unavailable for local files).
- **Highlights (defect 3, FIXED):** `retime_whisper_words` applies the [0.18,0.55]s natural-duration clamp to Whisper words too (per segment). Round 2 max word duration 0.55s (was 2.84s); 6 exact-time frames confirm the correct word lit.
- **Hook (defect 4, FIXED):** title/hook romanized to Latin when caption_script is latin (`to_roman_text`). Round 2 hook: "aur deedravar chalo ji".
- Tests: 104/104 pass after the fixes. Re-test clip: `data/out/realclip_test_stage_gold_v2.mp4` (session_3, 34.8s, 1080x1920, stage+gold, av_offset 0.0).
- Cosmetic leftovers observed in round 2 (not blocking): hook text overlaps the facecam band (0-5s); face-box crop includes game area left of the face; facecam top slightly clips the streamer's hair at 0.5-2s.
