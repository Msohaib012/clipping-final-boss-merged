"""Video rendering and editing engine supporting aspect ratios, framing modes, word-by-word captions, and overlays."""

import os
import re
import shutil
import tempfile
import uuid
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from .config import find_ffmpeg, find_ffprobe
from .caption_styles import get_caption_style
from .layouts import (BLUR_OVERSCAN, choose_layout, detect_facecam,
                      fallback_layout, game_crop, ratio_params)
from .frame_styles import _frame_style_geom, _stage_frame_assets

# Target aspect ratio definitions (width, height)
ASPECT_RATIOS = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
    "16:9": (1920, 1080),
}

# Framing values that are layout names (facecam_pos values from
# clipper.layouts) rather than the legacy blur_fit / crop_to_face modes.
LAYOUT_FRAMINGS = frozenset({
    "stage", "stage-nofacecam", "stage-topleft", "stage-topright",
    "stage-side", "top-band", "bottom-band", "top-right", "top-left",
    "mid-right", "mid-left", "bottom-left", "bottom-right", "off",
})

def detect_face_centers(video_path: Path, max_samples: int = 12) -> Tuple[float, float]:
    """
    Detects face center position normalized (0.0 to 1.0) using OpenCV Haar Cascade.
    Samples frames sequentially across video for maximum speed and smoothness.
    """
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(face_cascade_path):
        return (0.5, 0.4)

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return (0.5, 0.4)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    if total_frames <= 0 or frame_w <= 0 or frame_h <= 0:
        cap.release()
        return (0.5, 0.4)

    step = max(1, total_frames // max_samples)
    x_centers = []
    y_centers = []
    curr_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if curr_frame % step == 0:
            small_h = 240
            small_w = int(frame_w * (small_h / frame_h))
            small = cv2.resize(frame, (small_w, small_h))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=3,
                minSize=(25, 25),
            )

            if len(faces) > 0:
                faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
                fx, fy, fw, fh = faces[0]
                x_centers.append((fx + fw / 2.0) / small_w)
                y_centers.append((fy + fh / 2.0) / small_h)

            if len(x_centers) >= max_samples:
                break

        curr_frame += 1

    cap.release()

    if x_centers and y_centers:
        avg_x = float(np.median(x_centers))
        avg_y = float(np.median(y_centers))
        return (max(0.1, min(0.9, avg_x)), max(0.1, min(0.9, avg_y)))

    return (0.5, 0.4)

def format_ass_time(seconds: float) -> str:
    """Format seconds into ASS timestamp H:MM:SS.cc"""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def hex_to_ass_color(hex_str: str, alpha: str = "00") -> str:
    """Converts CSS hex color #RRGGBB to ASS &HAABBGGRR format."""
    h = hex_str.strip().lstrip("#")
    if len(h) == 3:
        h = f"{h[0]}{h[0]}{h[1]}{h[1]}{h[2]}{h[2]}"
    if len(h) >= 6:
        r, g, b = h[0:2], h[2:4], h[4:6]
        return f"&H{alpha}{b}{g}{r}".upper()
    return f"&H{alpha}FFFFFF"

def generate_ass_subtitles(
    words: List[Dict[str, Any]],
    output_ass: Path,
    style: str = "pop",
    target_w: int = 1080,
    target_h: int = 1920,
    hook_text: Optional[str] = None,
    trim_offset: float = 0.0,
    custom_style: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Generates word-by-word animated ASS subtitle file matching the visual templates.
    Supports custom style overrides (font, size, colors, position).
    """
    # Normalize aliases
    clean_style = (style or "pop").lower().strip()
    if clean_style == "karaoke":
        clean_style = "pop"
    elif clean_style == "tiktok":
        clean_style = "box"

    tmpl = get_caption_style(clean_style)
    font_name = tmpl.get("ass_font", "Arial")
    primary_color = tmpl.get("ass_primary", "&H00FFFFFF")
    active_color = tmpl.get("ass_active", "&H0000F2FF")
    outline_color = tmpl.get("ass_outline", "&H00000000")
    back_color = tmpl.get("ass_back", "&H80000000")
    bold = tmpl.get("bold", 1)
    italic = tmpl.get("italic", 0)
    border_style = tmpl.get("border_style", 1)
    outline = tmpl.get("outline", 4)
    shadow = tmpl.get("shadow", 2)
    scale_pct = int(tmpl.get("active_scale", 1.1) * 100)
    is_uppercase = tmpl.get("text_transform") == "uppercase"

    font_size = int(target_h * 0.038)
    margin_v = int(target_h * 0.28)  # lower-middle region

    # Apply custom style overrides if provided
    if custom_style:
        if custom_style.get("font"):
            font_name = custom_style["font"]
        if custom_style.get("size"):
            # Size scale: 24 -> normal
            scale_fac = float(custom_style["size"]) / 28.0
            font_size = int(target_h * 0.038 * scale_fac)
        if custom_style.get("primary_color"):
            primary_color = hex_to_ass_color(custom_style["primary_color"])
        if custom_style.get("highlight_color"):
            active_color = hex_to_ass_color(custom_style["highlight_color"])
        if custom_style.get("position"):
            pos = str(custom_style["position"]).lower()
            if pos == "bottom":
                margin_v = int(target_h * 0.16)
            elif pos == "middle":
                margin_v = int(target_h * 0.44)
            elif pos == "top":
                margin_v = int(target_h * 0.76)
        if custom_style.get("caption_y") is not None:
            # Layout-driven caption position: absolute strip-top Y in
            # output coords (from clipper.layouts.choose_layout).
            # margin_v = distance from the strip bottom to the frame
            # bottom, so the ASS text sits exactly on the layout's strip.
            try:
                cap_y = int(custom_style["caption_y"])
                strip_h = int(300 * target_h / 1280)
                margin_v = max(0, target_h - cap_y - strip_h)
            except (TypeError, ValueError):
                pass
        if custom_style.get("boxed"):
            border_style = 3  # opaque box
            back_color = hex_to_ass_color(custom_style.get("box_color", "#111827"), alpha="40")

    hook_style_size = int(target_h * 0.042)
    hook_margin_v = int(target_h * 0.10)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {target_w}
PlayResY: {target_h}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_color},&H000000FF,{outline_color},{back_color},{bold},{italic},0,0,100,100,1,0,{border_style},{outline},{shadow},2,40,40,{margin_v},1
Style: Hook,Arial,{hook_style_size},&H00FFFFFF,&H000000FF,&H00000000,&H90000000,1,0,0,0,100,100,1,0,1,5,3,8,40,40,{hook_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []

    # Optional top hook banner for the first 5 seconds
    if hook_text and hook_text.strip():
        clean_hook = hook_text.strip().replace("\n", " ")
        events.append(f"Dialogue: 1,0:00:00.00,0:00:05.00,Hook,,0,0,0,,{clean_hook}")

    # Process words into small 3-5 word chunks
    if words:
        chunks = []
        current_chunk = []
        for w in words:
            # Adjust word start/end relative to trim offset
            w_start = w.get("start", 0.0) - trim_offset
            w_end = w.get("end", 0.0) - trim_offset
            if w_end < 0:
                continue
            # Roman captions: render the transliterated "display" text when
            # present (clipper.roman_captions.apply_caption_script), else
            # the raw transcribed word. Both the chunk text and the
            # sentence-end check use the display form.
            disp = (w.get("display") or w.get("word") or "").strip()
            current_chunk.append({
                "word": disp,
                "start": max(0.0, w_start),
                "end": max(0.0, w_end),
            })
            if len(current_chunk) >= 4 or disp.endswith((".", "!", "?")):
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
        if current_chunk:
            chunks.append(current_chunk)

        # For each chunk, highlight active word as it plays
        for chunk in chunks:
            chunk_start = chunk[0]["start"]
            chunk_end = chunk[-1]["end"]
            if chunk_end <= chunk_start:
                chunk_end = chunk_start + 1.5

            for i, active_w in enumerate(chunk):
                w_start = active_w["start"]
                w_end = max(w_start + 0.1, active_w["end"])

                # Assemble line with active word styled
                line_parts = []
                for j, item in enumerate(chunk):
                    word_str = item["word"]
                    if is_uppercase:
                        word_str = word_str.upper()

                    if i == j:
                        # Active word highlighted
                        if scale_pct != 100:
                            line_parts.append(r"{\c" + active_color + f"\\fscx{scale_pct}\\fscy{scale_pct}" + r"}" + word_str + r"{\r}")
                        else:
                            line_parts.append(r"{\c" + active_color + r"}" + word_str + r"{\r}")
                    else:
                        line_parts.append(r"{\c" + primary_color + r"}" + word_str)

                text_line = " ".join(line_parts)
                start_str = format_ass_time(w_start)
                end_str = format_ass_time(w_end)
                events.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text_line}")

    output_ass.write_text(header + "\n".join(events) + "\n", encoding="utf-8")

def get_render_staging_dir() -> Path:
    """Returns a short, writable staging directory for FFmpeg rendering."""
    # MERGE FIX: the Windows short path (C:/clipper_tmp, the MAX_PATH
    # workaround) must only be tried on Windows. On Linux, Path("C:/...")
    # is a RELATIVE path, mkdir creates a stray ./C:/ dir, and ffmpeg
    # then fails with "Protocol not found" on the C: output URL.
    candidates = [Path(tempfile.gettempdir()) / "clp_tmp"]
    if os.name == "nt":
        candidates.insert(0, Path("C:/clipper_tmp"))
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            test_file = candidate / f".probe_{os.getpid()}"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
            return candidate
        except Exception:
            continue
    fallback = Path(tempfile.gettempdir())
    return fallback


def _ev(n: int) -> int:
    """Even int (libx264 needs even dims); at least 0."""
    return (int(n) // 2) * 2


def _input_dims(video_path: Path) -> Tuple[int, int]:
    """Input video dimensions via OpenCV; falls back to the donor's
    assumed 1920x1080 source when probing fails."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            if w > 0 and h > 0:
                return (w, h)
    except Exception:
        pass
    return (1920, 1080)


def _input_duration(video_path: Path) -> Optional[float]:
    """Input video duration in seconds (cv2, ffprobe fallback); None
    when it cannot be probed (e.g. mocked / missing file in tests)."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            cap.release()
            if total > 0 and fps > 0:
                return total / fps
    except Exception:
        pass
    try:
        ffprobe = find_ffprobe()
        res = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=15)
        dur = float(res.stdout.strip())
        if dur > 0:
            return dur
    except Exception:
        pass
    return None


def _scale_src_crop(crop: Tuple[int, int, int, int],
                    iw: int, ih: int) -> Tuple[int, int, int, int]:
    """Scale a donor crop rect (1920x1080 source coords) to the actual
    input dimensions, keeping even dims and clamping inside the frame."""
    cx, cy, cw, ch = crop
    x = _ev(cx * iw / 1920)
    y = _ev(cy * ih / 1080)
    w = max(2, _ev(cw * iw / 1920))
    h = max(2, _ev(ch * ih / 1080))
    if x + w > iw:
        w = max(2, _ev(iw - x))
    if y + h > ih:
        h = max(2, _ev(ih - y))
    return (x, y, w, h)


def _scale_src_rect(rect: Tuple[int, int, int, int],
                    iw: int, ih: int) -> Tuple[int, int, int, int]:
    """Scale a donor facecam source rect (1920x1080 coords) to the
    actual input dimensions."""
    return _scale_src_crop(rect, iw, ih)


def _stage_game_crop(sg: Dict[str, Any], out_w: int, game_h: int) -> Tuple[int, int, int, int]:
    """Source crop (x, y, w, h) in the 1920x1080 source for the stage
    layout's gameplay area (full height, side-crop only). The donor's
    formula crop_w = out_w * 1080 / game_h overflows 1920 for wide
    outputs (16:9); there the crop is height-limited instead so the
    game area aspect is honoured without distortion."""
    ar = out_w / max(game_h, 1)
    if ar >= 16 / 9:
        ch = max(2, _ev(1920 / ar))
        return (0, _ev((1080 - ch) // 2), 1920, ch)
    return sg["crop"]


def _extract_facecam_video(input_video: Path, src_rect: Tuple[int, int, int, int],
                           out_w: int, out_h: int, ffmpeg: str,
                           trim_start: float, trim_duration: Optional[float],
                           out_path: Path,
                           target_aspect: Optional[float] = None) -> Optional[Path]:
    """Crop the facecam region out of the input video and encode it as
    a small video (same trim window as the main render). target_aspect:
    when given, trim the source rect (centred) to that aspect first —
    used by the band layout (band is wider than 16:9) and the stage
    layout (facecam aspect). Returns out_path on success, else None.
    Never raises."""
    try:
        x, y, w, h = [int(v) for v in src_rect]
        if target_aspect:
            ar = w / max(h, 1)
            if ar > target_aspect:  # too wide -> trim width
                nw = int(h * target_aspect)
                x += (w - nw) // 2
                w = nw
            elif ar < target_aspect:  # too tall -> trim height
                nh = int(w / target_aspect)
                y += (h - nh) // 2
                h = nh
        x, y = _ev(x), _ev(y)
        w, h = max(2, _ev(w)), max(2, _ev(h))
        out_w, out_h = max(2, _ev(out_w)), max(2, _ev(out_h))
        cmd = [ffmpeg, "-y", "-v", "error"]
        if trim_start > 0:
            cmd += ["-ss", str(trim_start)]
        if trim_duration is not None:
            cmd += ["-t", str(trim_duration)]
        cmd += [
            "-i", str(input_video),
            "-vf", f"crop={w}:{h}:{x}:{y},scale={out_w}:{out_h}:flags=lanczos",
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", str(out_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        if out_path.exists() and out_path.stat().st_size > 1000:
            return out_path
    except Exception as e:
        print(f"[render] facecam extraction failed ({e})")
    return None


def render_clip(
    input_video: Path,
    output_video: Path,
    aspect_ratio: str = "9:16",
    framing: str = "blur_fit",
    words: Optional[List[Dict[str, Any]]] = None,
    caption_style: str = "karaoke",
    captions_on: bool = True,
    hook_text: Optional[str] = None,
    trim_start: float = 0.0,
    trim_end: Optional[float] = None,
    custom_style: Optional[Dict[str, Any]] = None,
    facecam_pos: Optional[str] = None,
    caption_pos: str = "auto",
    facecam_size: str = "medium",
    facecam_band: int = 50,
    facecam_band_w: int = 100,
    frame_style: str = "gold",
    av_offset_s: float = 0.0,
) -> bool:
    """
    Renders clip with chosen aspect ratio, framing mode, captions, and overlays.
    Uses short staging folder for FFmpeg execution to prevent Windows MAX_PATH limits,
    then atomically moves completed video to output_video.

    framing may be a legacy mode ("blur_fit", "crop_to_face") or a layout
    name ("stage", "stage-nofacecam", "stage-topleft", "stage-topright",
    "stage-side", "top-band", "bottom-band", "top-right", "top-left",
    "mid-right", "mid-left", "bottom-left", "bottom-right", "off").
    facecam_pos defaults to framing when framing is a layout name.
    Layout geometry comes from clipper.layouts (ported donor
    layout.py/positions.py); the donor's frame-sequence pipeline is
    re-implemented here as pure-ffmpeg filtergraphs on the single input
    video. Letterbox detection is SKIPPED in this phase (see
    MERGE_NOTES.md) — the stage game crop uses the full 1080 height.

    av_offset_s: A/V sync offset in seconds (video lags audio by this
    much). Nonzero => dual ffmpeg inputs of the same file: input 0 is
    the video window shifted later by the offset, input 1 is the
    unshifted audio source; the filtergraph still reads [0:v] and audio
    is mapped from input 1. Zero keeps the legacy single input.
    """
    ffmpeg = find_ffmpeg()
    target_w, target_h = ASPECT_RATIOS.get(aspect_ratio, (1080, 1920))
    output_video.parent.mkdir(parents=True, exist_ok=True)

    staging_dir = get_render_staging_dir()
    staging_id = uuid.uuid4().hex[:8]
    staging_video = staging_dir / f"clp_{staging_id}.mp4"
    staging_ass = staging_dir / f"clp_{staging_id}.ass"
    stage_work_dir = staging_dir / f"clp_{staging_id}_stage"

    # Determine trim duration
    trim_duration = None
    if trim_end is not None and trim_end > trim_start:
        trim_duration = trim_end - trim_start

    # --- A/V offset: dual inputs when nonzero (see docstring) ---
    try:
        av_offset = float(av_offset_s or 0.0)
    except (TypeError, ValueError):
        av_offset = 0.0
    use_dual = av_offset != 0.0
    audio_idx = 1 if use_dual else 0
    # Extra ffmpeg inputs (facecam video, stage mask, stage glow) start
    # after the main input(s): index 1 single-input, 2 dual-input.
    extra_base_idx = 2 if use_dual else 1

    # --- Layout resolution (only for layout framing modes) ---
    layout_name = None
    if framing in LAYOUT_FRAMINGS:
        layout_name = framing
    elif facecam_pos in LAYOUT_FRAMINGS:
        layout_name = facecam_pos
    eff_facecam_pos = facecam_pos or layout_name or "auto"
    eff_caption_pos = caption_pos or "auto"
    layout = None
    layout_cap_y = None
    if layout_name:
        detection = None
        if layout_name not in ("stage-nofacecam", "off"):
            try:
                detection = detect_facecam(input_video)
            except Exception as e:
                print(f"[render] facecam detection failed ({e})")
                detection = None
        try:
            layout = choose_layout(
                detection, 1920, 1080,
                facecam_pos=layout_name, caption_pos=eff_caption_pos,
                ratio=aspect_ratio, res=1080,
                facecam_size=facecam_size, facecam_band=facecam_band,
                facecam_band_w=facecam_band_w)
        except Exception as e:
            print(f"[render] layout failed ({e}) — using fallback")
            try:
                layout = fallback_layout(
                    aspect_ratio, 1080, facecam_size=facecam_size,
                    facecam_band=facecam_band, facecam_band_w=facecam_band_w,
                    facecam_pos=layout_name, caption_pos=eff_caption_pos)
            except Exception:
                layout = None
        if layout is not None:
            layout_cap_y = layout.get("caption_y")

    color_fix = ("scale=out_color_matrix=bt709:out_range=tv,"
                 "format=yuv420p,setsar=1")
    extra_inputs: List[str] = []
    facecam_path: Optional[Path] = None
    filter_base: Optional[str] = None

    if layout is not None:
        rp = ratio_params(aspect_ratio, 1080, facecam_size=facecam_size,
                          facecam_band=facecam_band,
                          facecam_band_w=facecam_band_w)
        iw, ih = _input_dims(input_video)
        fc_idx = extra_base_idx
        mask_idx = extra_base_idx + 1
        glow_idx = extra_base_idx + 2
        src_rect = layout.get("facecam_src_rect")
        src_rect_scaled = _scale_src_rect(src_rect, iw, ih) if src_rect else None

        if layout.get("stage"):
            # --- stage family: sharp game over blurred bg, facecam on
            # top with rounded corners + animated frame, captions over
            # the blurred zone at the bottom (or a blurred caption band
            # for stage-top/centre/upper/lower).
            sg = dict(layout["stage_geom"])
            cap_band = layout.get("caption_band")
            has_fc = src_rect_scaled is not None and layout_name != "stage-nofacecam"
            glow_n = 0
            if has_fc:
                fc_rect, frame_rect, use_border = _frame_style_geom(
                    sg, frame_style, layout_name, target_w, target_h)
                sg["fc"] = fc_rect
                sg["frame"] = frame_rect
                sg["border"] = use_border
                fx, fy, fcw, fch = fc_rect
                facecam_path = _extract_facecam_video(
                    input_video, src_rect_scaled, fcw, fch, ffmpeg,
                    trim_start, trim_duration,
                    staging_dir / f"clp_{staging_id}_fc.mp4",
                    target_aspect=fcw / max(fch, 1))
                if facecam_path is None:
                    has_fc = False
                else:
                    stage_work_dir.mkdir(parents=True, exist_ok=True)
                    mask_path, glow_pattern, glow_n = _stage_frame_assets(
                        fcw, fch, use_border, sg["radius"], sg["glow_pad"],
                        str(stage_work_dir), frame_style=frame_style)
                    extra_inputs += ["-i", str(facecam_path),
                                     "-loop", "1", "-i", str(mask_path),
                                     "-framerate", "30", "-i", glow_pattern]
            if has_fc:
                fx, fy, fcw, fch = sg["fc"]
                frx, fry, _frw, _frh = sg["frame"]
                cx, cy, cw, ch = _scale_src_crop(
                    _stage_game_crop(sg, target_w, sg["game_h"]), iw, ih)
                if cap_band:
                    band_y, band_h_cap = cap_band
                    blur_src = (f"gblur=sigma={sg['blur_sigma']},"
                                f"crop={target_w}:{target_h}:0:{BLUR_OVERSCAN},"
                                f"split=2[vblur][vblur2]")
                    band_filt = (f"[vblur2]crop={target_w}:{band_h_cap}:0:{band_y}"
                                 f"[cband];"
                                 f"[v1][cband]overlay=0:{band_y}[v1b];")
                    cap_src = "[v1b]"
                else:
                    blur_src = (f"gblur=sigma={sg['blur_sigma']},"
                                f"crop={target_w}:{target_h}:0:{BLUR_OVERSCAN}[vblur]")
                    band_filt = ""
                    cap_src = "[v1]"
                filter_base = (
                    f"[0:v]crop={cw}:{ch}:{cx}:{cy},split=2[gc1][gc2];"
                    f"[gc1]scale={target_w}:{sg['game_h']}:flags=lanczos,"
                    f"unsharp=5:5:0.8:5:5:0.0[vsharp];"
                    f"[gc2]scale={target_w}:{target_h + 2 * BLUR_OVERSCAN}:flags=lanczos,"
                    f"{blur_src};"
                    f"[vblur][vsharp]overlay=0:{sg['band_h']}[v0];"
                    f"[{fc_idx}:v]format=rgba[fc0];"
                    f"[{mask_idx}:v]format=rgba[mk];[fc0][mk]alphamerge[fcr];"
                    f"[{glow_idx}:v]format=rgba,loop=loop=-1:size={glow_n}[glow];"
                    f"[v0][glow]overlay={frx}:{fry}[v1a];"
                    f"[v1a][fcr]overlay={fx}:{fy}[v1];"
                    f"{band_filt}"
                    f"{cap_src}{color_fix}[base]"
                )
            else:
                # stage WITHOUT facecam: sharp game fills the FULL frame
                # height (the freed top band goes to gameplay) over a
                # blurred copy; captions over the blurred zone at the
                # bottom. No facecam, no frame, no mask.
                game_h_nofc = sg["game_bottom"]
                cx, cy, cw, ch = _scale_src_crop(
                    _stage_game_crop(sg, target_w, game_h_nofc), iw, ih)
                if cap_band:
                    band_y, band_h_cap = cap_band
                    blur_src = (f"gblur=sigma={sg['blur_sigma']},"
                                f"crop={target_w}:{target_h}:0:{BLUR_OVERSCAN},"
                                f"split=2[vblur][vblur2]")
                    band_filt = (f"[vblur2]crop={target_w}:{band_h_cap}:0:{band_y}"
                                 f"[cband];"
                                 f"[v0][cband]overlay=0:{band_y}[v0b];")
                    cap_src = "[v0b]"
                else:
                    blur_src = (f"gblur=sigma={sg['blur_sigma']},"
                                f"crop={target_w}:{target_h}:0:{BLUR_OVERSCAN}[vblur]")
                    band_filt = ""
                    cap_src = "[v0]"
                filter_base = (
                    f"[0:v]crop={cw}:{ch}:{cx}:{cy},split=2[gc1][gc2];"
                    f"[gc1]scale={target_w}:{game_h_nofc}:flags=lanczos,"
                    f"unsharp=5:5:0.8:5:5:0.0[vsharp];"
                    f"[gc2]scale={target_w}:{target_h + 2 * BLUR_OVERSCAN}:flags=lanczos,"
                    f"{blur_src};"
                    f"[vblur][vsharp]overlay=0:0[v0];"
                    f"{band_filt}"
                    f"{cap_src}{color_fix}[base]"
                )
        elif layout.get("band"):
            # --- band layouts: facecam as a horizontal strip. No
            # facecam found -> fall back to plain game (blur_fit).
            band_h = layout["band_h"]
            band_w = layout["band_w"]
            band_pos = layout.get("band_pos", "top")
            if src_rect_scaled is not None:
                facecam_path = _extract_facecam_video(
                    input_video, src_rect_scaled, band_w, band_h, ffmpeg,
                    trim_start, trim_duration,
                    staging_dir / f"clp_{staging_id}_fc.mp4",
                    target_aspect=band_w / max(band_h, 1))
            if facecam_path is not None:
                extra_inputs += ["-i", str(facecam_path)]
                if band_w >= target_w:
                    # full-width band: split-screen — facecam + game stacked
                    cx, cy, cw, ch = _scale_src_crop(
                        game_crop(rp, band=True), iw, ih)
                    game_h = target_h - band_h
                    stack = ("[fc][v0]vstack[v1]" if band_pos == "top"
                             else "[v0][fc]vstack[v1]")
                    filter_base = (
                        f"[0:v]crop={cw}:{ch}:{cx}:{cy},"
                        f"scale={target_w}:{game_h}:flags=lanczos,"
                        f"unsharp=5:5:0.8:5:5:0.0[v0];"
                        f"[{fc_idx}:v]drawbox=x=0:y=0:w=iw:h=ih:color=0x00E5FF:t=3[fc];"
                        f"{stack};"
                        f"[v1]{color_fix}[base]"
                    )
                else:
                    # narrower band: game stays full-frame, band overlaid
                    # centred (top or bottom) — no black bars.
                    cx, cy, cw, ch = _scale_src_crop(
                        game_crop(rp, band=False), iw, ih)
                    band_x = (target_w - band_w) // 2
                    band_y = 0 if band_pos == "top" else target_h - band_h
                    filter_base = (
                        f"[0:v]crop={cw}:{ch}:{cx}:{cy},"
                        f"scale={target_w}:{target_h}:flags=lanczos,"
                        f"unsharp=5:5:0.8:5:5:0.0[v0];"
                        f"[{fc_idx}:v]drawbox=x=0:y=0:w=iw:h=ih:color=0x00E5FF:t=3[fc];"
                        f"[v0][fc]overlay={band_x}:{band_y}[v1];"
                        f"[v1]{color_fix}[base]"
                    )
            else:
                print(f"[render] {band_pos}-band chosen but no facecam "
                      f"found — falling back to blur_fit", flush=True)
        elif layout.get("overlay"):
            # --- corner / mid modes: game fills the frame, facecam in a
            # corner (cyan border). No facecam found -> blur_fit.
            ov = layout["overlay"]
            if src_rect_scaled is not None:
                facecam_path = _extract_facecam_video(
                    input_video, src_rect_scaled, ov["w"], ov["h"], ffmpeg,
                    trim_start, trim_duration,
                    staging_dir / f"clp_{staging_id}_fc.mp4")
            if facecam_path is not None:
                extra_inputs += ["-i", str(facecam_path)]
                cx, cy, cw, ch = _scale_src_crop(
                    game_crop(rp, band=False), iw, ih)
                filter_base = (
                    f"[0:v]crop={cw}:{ch}:{cx}:{cy},"
                    f"scale={target_w}:{target_h}:flags=lanczos,"
                    f"unsharp=5:5:0.8:5:5:0.0[v0];"
                    f"[{fc_idx}:v]drawbox=x=0:y=0:w=iw:h=ih:color=0x00E5FF:t=3[fc];"
                    f"[v0][fc]overlay={ov['x']}:{ov['y']}[v1];"
                    f"[v1]{color_fix}[base]"
                )
            else:
                print("[render] no facecam found — falling back to blur_fit",
                      flush=True)
        else:
            # --- facecam off: plain game visuals, centre crop + scale.
            cx, cy, cw, ch = _scale_src_crop(
                game_crop(rp, band=False), iw, ih)
            filter_base = (
                f"[0:v]crop={cw}:{ch}:{cx}:{cy},"
                f"scale={target_w}:{target_h}:flags=lanczos,"
                f"unsharp=5:5:0.8:5:5:0.0[v0];"
                f"[v0]{color_fix}[base]"
            )

    # Build video filter graph based on framing mode (legacy branches
    # unchanged — blur_fit / crop_to_face — plus the layout branches
    # above, which leave filter_base set; a layout that fell back to
    # plain game (band/corner with no facecam) gets blur_fit here).
    if filter_base is None:
        if framing == "crop_to_face":
            center_x, center_y = detect_face_centers(input_video)
            target_ratio = target_w / target_h
            filter_base = (
                f"[0:v]split[v1][v2];"
                f"[v1]scale=trunc(ih*{target_ratio}/2)*2:ih[face_crop];"
                f"[v2]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{target_h},boxblur=20:5[bg];"
                f"[face_crop]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base]"
            )
        else:  # blur_fit (default)
            filter_base = (
                f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
                f"crop={target_w}:{target_h},"
                f"boxblur=20:5,eq=brightness=-0.10[bg];"
                f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base]"
            )

    # Subtitles handling in short staging directory. The layout's
    # caption Y (if any) is passed through custom_style so the ASS
    # MarginV lands the text on the layout's strip.
    has_subtitles = False
    if captions_on and (words or (hook_text and hook_text.strip())):
        ass_custom = dict(custom_style) if custom_style else {}
        if layout_cap_y is not None:
            ass_custom["caption_y"] = layout_cap_y
        generate_ass_subtitles(
            words=words or [],
            output_ass=staging_ass,
            style=caption_style,
            target_w=target_w,
            target_h=target_h,
            hook_text=hook_text,
            trim_offset=trim_start,
            custom_style=ass_custom,
        )
        has_subtitles = staging_ass.exists()

    # Format subtitles filter safely for Windows path escaping
    if has_subtitles:
        ass_str = str(staging_ass).replace("\\", "/").replace(":", "\\:")
        video_filter = f"{filter_base};[base]ass='{ass_str}'[outv]"
    else:
        video_filter = f"{filter_base};[base]null[outv]"

    def _build_cmd(vf: str) -> List[str]:
        # Input 0 = video window (shifted later by av_offset when set);
        # input 1 (dual only) = the same file, unshifted, for audio.
        # Output -t caps the encode at the trim window (or the full
        # input duration when untrimmed). This is REQUIRED, not just a
        # convenience: the stage mask (-loop 1) and glow (loop=-1)
        # inputs are infinite, so without an output cap the filtergraph
        # never EOFs and ffmpeg encodes forever (the donor capped its
        # output the same way, with -t dur as an output option).
        c = [ffmpeg, "-y"]
        video_ss = trim_start + av_offset if use_dual else trim_start
        if video_ss > 0:
            c += ["-ss", str(video_ss)]
        if trim_duration is not None:
            c += ["-t", str(trim_duration)]
        c += ["-i", str(input_video)]
        if use_dual:
            if trim_start > 0:
                c += ["-ss", str(trim_start)]
            if trim_duration is not None:
                c += ["-t", str(trim_duration)]
            c += ["-i", str(input_video)]
        c += extra_inputs
        c += [
            "-filter_complex", vf,
            "-map", "[outv]",
            "-map", f"{audio_idx}:a?",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-c:a", "aac",
            "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
        ]
        out_dur = trim_duration
        if out_dur is None:
            out_dur = _input_duration(input_video)
        if out_dur is not None and out_dur > 0:
            c += ["-t", str(out_dur)]
        c += [str(staging_video)]
        return c

    cmd = _build_cmd(video_filter)

    success = False
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if staging_video.exists() and staging_video.stat().st_size > 1000:
            shutil.move(str(staging_video), str(output_video))
            success = output_video.exists() and output_video.stat().st_size > 1000
    except subprocess.CalledProcessError as e:
        print(f"Render failed with error: {e.stderr}")
        # If ASS filter failed (e.g. fontconfig), retry without ASS
        if has_subtitles:
            try:
                cmd_retry = _build_cmd(f"{filter_base};[base]null[outv]")
                subprocess.run(cmd_retry, capture_output=True, check=True)
                if staging_video.exists() and staging_video.stat().st_size > 1000:
                    shutil.move(str(staging_video), str(output_video))
                    success = output_video.exists() and output_video.stat().st_size > 1000
            except Exception as e2:
                print(f"Render retry failed: {e2}")
    finally:
        if staging_ass.exists():
            try:
                staging_ass.unlink(missing_ok=True)
            except Exception:
                pass
        if staging_video.exists():
            try:
                staging_video.unlink(missing_ok=True)
            except Exception:
                pass
        if facecam_path is not None and facecam_path.exists():
            try:
                facecam_path.unlink(missing_ok=True)
            except Exception:
                pass
        if stage_work_dir.exists():
            shutil.rmtree(stage_work_dir, ignore_errors=True)

    return success
