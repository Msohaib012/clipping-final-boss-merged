"""Auto layout for the PeakCut-style 'gameplay' mode — ported from the
GTA 6 Live Clipper donor (layout.py + positions.py) into Clipping Final
Boss.

Adaptations vs the donor:
- Face detection: MediaPipe BlazeFace (FaceDetector, full-range model
  bundled in clipper/models/) is the primary face finder — ported from
  the GTA 6 Live Clipper donor layout.py; OpenCV Haar cascades (the
  same cascade CFB's renderer already uses) are the fallback when
  MediaPipe or the model is unavailable. MediaPipe's native loader
  links libEGL + libGLESv2 — on a fresh VM, `sudo apt-get install -y
  libegl1 libgles2` first, else detection degrades gracefully to Haar.
- Still-border aspect filter is 0.6-2.5 (not the donor's 1.0-2.5):
  real streamer facecams are often portrait (~0.67).
- No per-channel facecam rect fallbacks (no hardcoded channel ids).
- No preview-drawing helpers (draw_layout_preview /
  build_position_previews) — CFB renders captions with ASS, not the
  donor's PIL caption track.
- Letterbox detection is NOT ported in this phase (see MERGE_NOTES.md).
"""

import os
import statistics
import subprocess
import tempfile

from .config import find_ffmpeg

BLUR_OVERSCAN = 64

OUT_W, OUT_H = 720, 1280
# Facecam overlay: 40% of output width (PeakCut: 30%). 16:9.
FC_W, FC_H = 288, 162
MARGIN = 28
CAP_Y_DEFAULT = 880
CAP_Y_CANDIDATES = (880, 650, 450)  # strip-top Y, in order of preference
STRIP_H = 300
TEXT_TOP, TEXT_BOTTOM = 100, 200    # text band inside the strip

# --- facecam positions -------------------------------------------------
FACECAM_POSITIONS = ("auto", "top-right", "top-left", "mid-right",
                     "mid-left", "bottom-left", "bottom-right",
                     "top-band", "bottom-band", "stage", "stage-nofacecam",
                     "stage-topleft", "stage-topright", "stage-side",
                     "off")
FACECAM_META = {
    "auto": {"name": "Auto (per stream)",
             "desc": "The tool checks this stream's own frames and picks the freest corner — avoids the HUD and the streamer's own facecam. Default.",
             "rect": None},
    "top-right": {"name": "Top-right corner",
                  "desc": "PeakCut-like corner — the streamer's face, bigger than PeakCut (40% of width).",
                  "rect": (416, 16, 288, 162)},
    "top-left": {"name": "Top-left corner",
                 "desc": "Mirror of top-right — for streams whose top-right has a HUD or logo.",
                 "rect": (16, 16, 288, 162)},
    "mid-right": {"name": "Mid-right",
                  "desc": "Right side, vertically centred — below the top HUD, above the caption band.",
                  "rect": (416, 420, 288, 162)},
    "mid-left": {"name": "Mid-left",
                 "desc": "Left side, vertically centred — same as mid-right, other side.",
                 "rect": (16, 420, 288, 162)},
    "bottom-left": {"name": "Bottom-left",
                    "desc": "Bottom-left corner, above the platform UI strip — captions move up to clear it.",
                    "rect": (28, 1090, 288, 162)},
    "bottom-right": {"name": "Bottom-right",
                     "desc": "Bottom-right corner, above the platform UI strip — captions move up to clear it.",
                     "rect": (404, 1090, 288, 162)},
    "top-band": {"name": "Top band (full width)",
                 "desc": "Clipie-style: the facecam stretches across the whole top as a band (50% by default, 30% option), gameplay fills below it.",
                 "rect": "band"},
    "bottom-band": {"name": "Bottom band (full width)",
                    "desc": "Mirror of top-band: the facecam stretches across the whole bottom as a band, gameplay fills above it.",
                    "rect": "band"},
    "stage": {"name": "Stage — facecam top, big game, captions below",
              "desc": "Compact facecam band on top (large facecam, centred), gameplay AS LARGE AS POSSIBLE in the middle — full height, side-crop only — captions in a tight empty zone at the bottom.",
              "rect": "stage"},
    "stage-nofacecam": {"name": "Stage — no facecam, full-frame game",
                        "desc": "The stage look WITHOUT a facecam — gameplay fills the FULL frame height, captions over the same blurred-gameplay zone at the bottom.",
                        "rect": "stage-nofacecam"},
    "stage-topleft": {"name": "Stage — facecam top-left",
                      "desc": "Stage structure, facecam in the top-LEFT of the top band (gold-framed, rounded corners) — big gameplay below, captions over the blurred zone at the bottom.",
                      "rect": "stage-topleft"},
    "stage-topright": {"name": "Stage — facecam top-right",
                       "desc": "Stage structure, facecam in the top-RIGHT of the top band (gold-framed, rounded corners) — big gameplay below, captions over the blurred zone at the bottom.",
                       "rect": "stage-topright"},
    "stage-side": {"name": "Stage — facecam mid-left",
                   "desc": "Stage structure, facecam on the MID-LEFT side (gold-framed, rounded corners), gameplay fills the full frame behind it — captions over the blurred zone at the bottom.",
                   "rect": "stage-side"},
    "off": {"name": "Off (no facecam)",
            "desc": "No facecam overlay at all — just the game, full frame.",
            "rect": None},
}

# --- caption positions -------------------------------------------------
CAPTION_POSITIONS = ("auto", "bottom", "seam", "top",
                     "stage-bottom", "stage-top", "stage-centre",
                     "stage-upper", "stage-lower")
CAPTION_META = {
    "auto": {"name": "Auto (first free line)",
             "desc": "The tool checks the facecam + HUD and picks the first free line: bottom, then higher. Default.",
             "y": None},
    "bottom": {"name": "Bottom band (y~995-1065)",
               "desc": "Text at y~995-1065 — below the action centre, above the platforms' own UI.",
               "y": 880},
    "seam": {"name": "Seam (below the facecam band)",
             "desc": "Text just below the top-band facecam (Clipie-style seam). Best with the top-band facecam.",
             "y": 430},
    "top": {"name": "Top",
            "desc": "Text at the very top of the frame — for when the bottom is busy (HUD, subtitles in the game).",
            "y": 16},
    "stage-bottom": {"name": "Stage — bottom",
                     "desc": "Stage captions in the blurred zone at the bottom — text y~995-1065.",
                     "y": 880},
    "stage-top": {"name": "Stage — top",
                  "desc": "Stage captions just below the facecam band — text y~245-315 over a blurred gameplay strip, game fills below.",
                  "y": 130},
    "stage-centre": {"name": "Stage — centre",
                     "desc": "Stage captions in the middle of the gameplay — text y~635-705 over a blurred gameplay strip.",
                     "y": 520},
    "stage-upper": {"name": "Stage — upper",
                    "desc": "Stage captions in the upper-middle of the gameplay — text y~450-520 over a blurred gameplay strip.",
                    "y": 335},
    "stage-lower": {"name": "Stage — lower",
                    "desc": "Stage captions in the lower third of the gameplay — text y~850-920 over a blurred gameplay strip, above the bottom zone.",
                    "y": 735},
}

# --- output ratios -----------------------------------------------------
RATIOS = ("9:16", "1:1", "4:5", "16:9")
RATIO_META = {
    "9:16": {"name": "9:16 vertical",
             "desc": "Shorts / Reels / TikTok. Default.",
             "size_720": (720, 1280), "size_1080": (1080, 1920)},
    "1:1": {"name": "1:1 square",
            "desc": "Square feeds (Instagram grid, X).",
            "size_720": (720, 720), "size_1080": (1080, 1080)},
    "4:5": {"name": "4:5 portrait",
            "desc": "Portrait feeds (Instagram, Facebook).",
            "size_720": (720, 900), "size_1080": (1080, 1350)},
    "16:9": {"name": "16:9 landscape",
             "desc": "Landscape (YouTube, X, Facebook).",
             "size_720": (1280, 720), "size_1080": (1920, 1080)},
}

# facecam size per ratio at 720 (width, height); scaled by res/720.
# 40% of width for portrait/square; 25% for landscape. These are the
# MEDIUM sizes; facecam_size scales them (small 0.75x, large 1.25x).
FACECAM_SIZE_720 = {
    "9:16": (288, 162),
    "1:1": (288, 162),
    "4:5": (288, 162),
    "16:9": (320, 180),
}
FACECAM_SIZE_SCALE = {"small": 0.75, "medium": 1.0, "large": 1.25}


def ratio_params(ratio="9:16", res=720, facecam_size="medium",
                 facecam_band=50, facecam_band_w=100):
    """Everything the renderer needs for one ratio+res.

    facecam_size: small/medium/large (scales the facecam overlay).
    facecam_band: 50 or 30 — the top/bottom-band facecam height as % of
    out_h (50 = half-half, 30 = the smaller band).
    facecam_band_w: 100, 75 or 50 — the band's width as % of out_w
    (100 = full width, narrower bands are centred over the game).
    The source is 1920x1080; the crop is the largest window at the target
    aspect (9:16 keeps the proven 608x1080 centre crop = 1.18x upscale).
    """
    if ratio not in RATIOS:
        raise ValueError(f"ratio must be one of {RATIOS}, got {ratio!r}")
    if res not in (720, 1080):
        raise ValueError(f"res must be 720 or 1080, got {res!r}")
    if facecam_size not in FACECAM_SIZE_SCALE:
        facecam_size = "medium"
    if facecam_band not in (30, 50):
        facecam_band = 50
    if facecam_band_w not in (100, 75, 50):
        facecam_band_w = 100
    out_w, out_h = RATIO_META[ratio][f"size_{res}"]
    s = res / 720.0  # resolution scale (1.0 at 720, 1.5 at 1080)
    # source crop at the target aspect, largest possible from 1920x1080
    ar = out_w / out_h
    if ar >= 16 / 9:
        crop_w, crop_h = 1920, 1080
    else:
        crop_h = 1080
        crop_w = int(1080 * ar)
        if ratio == "9:16":
            crop_w = 608 if res == 720 else 912  # the proven centre crop
    crop_x = (1920 - crop_w) // 2
    fc_w, fc_h = FACECAM_SIZE_720[ratio]
    fc_w, fc_h = int(fc_w * s * FACECAM_SIZE_SCALE[facecam_size]), \
        int(fc_h * s * FACECAM_SIZE_SCALE[facecam_size])
    fc_w, fc_h = (fc_w // 2) * 2, (fc_h // 2) * 2  # libx264 needs even dims
    band_h = int(out_h * facecam_band / 100)
    band_w = (int(out_w * facecam_band_w / 100) // 2) * 2
    return {
        "ratio": ratio, "res": res,
        "out_w": out_w, "out_h": out_h,
        "crop": (crop_x, 0, crop_w, crop_h),
        "fc_w": fc_w, "fc_h": fc_h,
        "band_h": band_h,
        "band_w": band_w,
        "strip_h": int(300 * out_h / 1280),  # caption strip height
        "cap_y_default": int(880 * out_h / 1280),
        "cap_y_seam": int(430 * out_h / 1280),
        "cap_y_top": int(16 * out_h / 1280),
        "cap_y_stage_bottom": int(880 * out_h / 1280),
        "cap_y_stage_top": int(130 * out_h / 1280),
        "cap_y_stage_centre": int(520 * out_h / 1280),
        "cap_y_stage_upper": int(335 * out_h / 1280),
        "cap_y_stage_lower": int(735 * out_h / 1280),
    }


def game_crop(rp, band=False):
    """Source crop (x, y, w, h) in the 1920x1080 source for the game area.

    band=False: the full output frame is gameplay (the normal crop).
    band=True: only the area beside the top/bottom-band facecam is
    gameplay, so the crop is wider (e.g. 910x1080 for 720x1280 — more of
    the game stays visible instead of stretching the 608px crop).
    """
    if not band:
        cx, cy, cw, ch = rp["crop"]
        return (cx, cy, cw, ch)
    out_w, out_h = rp["out_w"], rp["out_h"]
    game_h = out_h - rp["band_h"]
    ar = out_w / game_h
    if ar >= 16 / 9:
        cw, ch = 1920, int(1920 / ar)
        return ((1920 - cw) // 2, (1080 - ch) // 2, cw, ch)
    ch = 1080
    cw = int(1080 * ar)
    return ((1920 - cw) // 2, 0, cw, ch)


def stage_geometry(rp, facecam_pos="stage"):
    """Geometry for the 'stage' layout FAMILY (output coords).

    Base format (all variants): SMALL facecam (288x162 at 720p) with a
    rounded-corner GOLD frame + soft glow, gameplay AS LARGE AS
    POSSIBLE (full game height, side-crop only), captions over BLURRED
    gameplay in a tight zone at the bottom (strip-top 880 / text
    995-1064 at 720p, clear of the platform UI that starts ~1080).

    facecam_pos — where the facecam sits:
      "stage" (top-centre): facecam centred in the top band; gameplay
        y 188-980 (792px = 61.9% of the canvas).
      "stage-topleft" / "stage-topright": facecam in the top band,
        left/right; same gameplay as stage.
      "stage-side": facecam mid-left, vertically centred; gameplay
        fills the FULL frame (y 0-980, no top band) behind it.
      "stage-nofacecam": no facecam; gameplay fills the FULL frame
        (y 0-980). (fc is a dummy — the renderer never uses it.)
    """
    out_w, out_h = rp["out_w"], rp["out_h"]
    s = out_h / 1280.0
    # smaller facecam (288x162 at 720p), gold frame around it
    fc_w, fc_h = int(288 * s), int(162 * s)
    fc_w, fc_h = (fc_w // 2) * 2, (fc_h // 2) * 2
    border = max(3, int(4 * s))       # gold border thickness
    radius = max(10, int(18 * s))     # corner radius of facecam + frame
    glow_pad = max(6, int(8 * s))     # glow extends this far past border
    top_margin = max(12, int(14 * s))  # keeps frame/glow off the top edge
    ext = border + glow_pad            # frame extends this far past facecam
    # gameplay bottom: just above the caption text top (~995 at 720p)
    game_bottom = int(980 * s)
    if facecam_pos == "stage-side":
        # facecam mid-left, game fills the full frame (no top band)
        band_h = 0
        fc_x = top_margin
        fc_y = (out_h - fc_h) // 2
        game_y0 = 0
    elif facecam_pos == "stage-nofacecam":
        band_h = 0
        fc_x, fc_y = 0, 0  # no facecam (unused by the renderer)
        game_y0 = 0
    elif facecam_pos == "stage-topleft":
        band_h = top_margin + fc_h + ext
        fc_x = top_margin
        fc_y = top_margin
        game_y0 = band_h
    elif facecam_pos == "stage-topright":
        band_h = top_margin + fc_h + ext
        fc_x = out_w - fc_w - top_margin
        fc_y = top_margin
        game_y0 = band_h
    else:  # "stage" (top-centre, the original)
        band_h = top_margin + fc_h + ext
        fc_x = (out_w - fc_w) // 2
        fc_y = top_margin
        game_y0 = band_h
    frame_w, frame_h = fc_w + 2 * ext, fc_h + 2 * ext
    frame_x, frame_y = fc_x - ext, fc_y - ext
    cap_y = rp["cap_y_default"]
    game_h = game_bottom - game_y0
    # source crop: full height (1080), width set by the game-area aspect
    crop_w = int(out_w * 1080 / game_h)
    crop_w = (crop_w // 2) * 2
    crop_x = (1920 - crop_w) // 2
    return {
        "band_h": band_h,
        "fc": (fc_x, fc_y, fc_w, fc_h),
        "frame": (frame_x, frame_y, frame_w, frame_h),
        "border": border,
        "radius": radius,
        "glow_pad": glow_pad,
        "top_margin": top_margin,
        "blur_sigma": 14,
        "game_y0": game_y0,
        "game_h": game_h,
        "game_bottom": game_bottom,
        "cap_y": cap_y,
        "crop": (crop_x, 0, crop_w, 1080),
        "visible_w_frac": crop_w / 1920.0,
        "game_frac_h": game_h / out_h,
    }


def stage_game_crop(rp):
    """Source crop (x, y, w, h) in the 1920x1080 source for the stage
    layout's gameplay area (full height, side-crop only)."""
    return stage_geometry(rp)["crop"]


def facecam_rect(position, rp, auto_layout=None):
    """Overlay rect (x, y, w, h) in OUTPUT coords for a facecam position.

    position: one of FACECAM_POSITIONS. auto_layout: the dict returned by
    choose_layout() (used for "auto"). Returns:
      ("corner", (x, y, w, h)) — a corner/mid overlay,
      ("band", None)           — the full-width top band,
      (None, None)             — off / no facecam.
    """
    if position not in FACECAM_POSITIONS:
        raise ValueError(f"facecam position must be one of "
                         f"{FACECAM_POSITIONS}, got {position!r}")
    out_w, out_h = rp["out_w"], rp["out_h"]
    fc_w, fc_h = rp["fc_w"], rp["fc_h"]
    sx, sy = out_w / 720.0, out_h / 1280.0
    if position == "off":
        return None, None
    if position == "top-band":
        return "band", None
    if position == "bottom-band":
        return "band", None
    if position == "stage":
        return "stage", None
    if position == "auto":
        if auto_layout and auto_layout.get("overlay"):
            ov = auto_layout["overlay"]
            return "corner", (int(ov["x"] * sx), int(ov["y"] * sy),
                              int(ov["w"] * sx), int(ov["h"] * sy))
        position = "top-right"  # fallback when auto has nothing
    # fixed positions: rects are defined in 720x1280 space, scaled here
    rect = FACECAM_META[position]["rect"]
    x, y, w, h = rect
    # fixed rects are 288x162 at 720 — rescale to this ratio's facecam size
    # (keeps the same corner/margin, swaps in the right size)
    margin_x = int(28 * sx)
    margin_y = int(28 * sy)
    if "right" in position:
        x = out_w - fc_w - margin_x
    else:
        x = margin_x
    if position.startswith("mid"):
        y = int(420 * sy)
    elif position.startswith("bottom"):
        y = out_h - fc_h - margin_y
    else:
        y = margin_y
    return "corner", (x, y, fc_w, fc_h)


def caption_y(position, rp, auto_layout=None):
    """Strip-top Y (output coords) for a caption position."""
    if position not in CAPTION_POSITIONS:
        raise ValueError(f"caption position must be one of "
                         f"{CAPTION_POSITIONS}, got {position!r}")
    if position == "auto":
        if auto_layout and auto_layout.get("caption_y") is not None:
            return int(auto_layout["caption_y"] * rp["out_h"] / 1280)
        return rp["cap_y_default"]
    if position == "bottom":
        return rp["cap_y_default"]
    if position == "seam":
        return rp["cap_y_seam"]
    if position == "top":
        return rp["cap_y_top"]
    # stage caption positions (strip-top Y from ratio_params)
    return rp["cap_y_" + position.replace("-", "_")]


def stage_caption_band(position, rp):
    """Blurred caption band (y, h) in OUTPUT coords for a stage caption
    position — the strip of blurred gameplay the captions sit over.
    Returns None when no extra band is needed (stage-bottom: the text
    already sits in the bottom blurred zone). The band covers the
    caption text (cap_y + 115..184) with ~35px padding top/bottom.
    Only meaningful for the stage layout (the renderer draws the band
    in the stage filter branches)."""
    if position not in ("stage-top", "stage-centre", "stage-upper",
                        "stage-lower"):
        return None
    cap_y = caption_y(position, rp)  # already in OUTPUT coords
    s = rp["out_h"] / 1280.0
    return (int(cap_y + 80 * s), int(140 * s))


# --- facecam detection -------------------------------------------------
# CFB adaptation of the donor's detect_facecam: frames are sampled from
# the input video with ffmpeg into a temp dir, then the donor's Canny
# still-border + face detection runs on them. The donor used MediaPipe
# BlazeFace; CFB has no MediaPipe, so faces come from OpenCV Haar
# cascades (the same cascade CFB's renderer already uses). No
# per-channel rect fallbacks — a wrong rect is worse than no facecam.

# Bundled BlazeFace model (Google, ~1MB) for MediaPipe face detection.
_FACE_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "models", "blaze_face_full_range.tflite")


def _sample_frame_paths(frames_dir, n=8):
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    if not files:
        return []
    if len(files) <= n:
        return [os.path.join(frames_dir, f) for f in files]
    step = len(files) / n
    return [os.path.join(frames_dir, files[int(i * step)]) for i in range(n)]


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _detect_facecam_from_frames(frames_dir, src_w=1920, src_h=1080,
                                 model_path=None):
    """Find the streamer's facecam rectangle in source coords.

    Returns {"rect": (x, y, w, h), "method": "still-border"|"face-box",
    "face": (cx, cy, w, h) | None, "samples": n} or None when nothing
    usable was found (caller falls back to plain game mode).

    Faces come from MediaPipe BlazeFace (bundled full-range model) when
    available — ported from the GTA 6 Live Clipper donor layout.py —
    with OpenCV Haar as the fallback. The still-border aspect filter
    is 0.6-2.5: real streamer facecams are often portrait (~0.67).
    """
    try:
        import cv2
    except ImportError:
        print("[layout] opencv not installed — facecam detection unavailable")
        return None
    paths = _sample_frame_paths(frames_dir)
    if not paths:
        return None

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = None
    if os.path.exists(cascade_path):
        face_cascade = cv2.CascadeClassifier(cascade_path)

    # --- MediaPipe face boxes (BlazeFace full-range, same model + options
    # as the donor). Graceful fallback to Haar when unavailable.
    mp_det = None
    _model = model_path or _FACE_MODEL
    if _model and os.path.exists(_model):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            opts = vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=_model),
                min_detection_confidence=0.4)
            mp_det = vision.FaceDetector.create_from_options(opts)
        except Exception as e:
            print(f"[layout] mediapipe unavailable ({e}) — Haar-only detection")

    rect_votes = {}  # quantised rect -> count
    busy = None      # edge-density grid, accumulated
    face_boxes = []  # per-sample (cx, cy, w, h) in source coords
    for p in paths:
        frame = cv2.imread(p)
        if frame is None:
            continue
        frame = cv2.resize(frame, (src_w, src_h))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # busy grid (12 rows x 20 cols) for HUD detection
        gh, gw = 12, 20
        cell = edges.reshape(gh, gray.shape[0] // gh, gw, gray.shape[1] // gw)
        dens = (cell > 0).mean(axis=(1, 3))
        busy = dens if busy is None else busy + dens
        # still-border rectangles
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 0.01 * src_w * src_h or area > 0.30 * src_w * src_h:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            x, y, w, h = cv2.boundingRect(approx)
            if not (0.6 <= w / max(h, 1) <= 2.5):
                continue
            key = (round(x / 20) * 20, round(y / 20) * 20,
                   round(w / 20) * 20, round(h / 20) * 20)
            rect_votes[key] = rect_votes.get(key, 0) + 1
        # face boxes (MediaPipe BlazeFace — primary)
        if mp_det is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = mp_det.detect(img)
            except Exception:
                res = None
            if res:
                for d in res.detections:
                    bb = d.bounding_box
                    face_boxes.append((
                        int(bb.origin_x + bb.width // 2),
                        int(bb.origin_y + bb.height // 2),
                        int(bb.width), int(bb.height)))
        # face boxes (Haar, on a downscaled copy for speed — fallback)
        elif face_cascade is not None:
            small_h = 360
            small_w = int(src_w * (small_h / src_h))
            small = cv2.resize(gray, (small_w, small_h))
            faces = face_cascade.detectMultiScale(
                small, scaleFactor=1.2, minNeighbors=4, minSize=(20, 20))
            if len(faces) > 0:
                faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
                fx, fy, fw, fh = faces[0]
                face_boxes.append((
                    int((fx + fw / 2.0) / small_w * src_w),
                    int((fy + fh / 2.0) / small_h * src_h),
                    int(fw / small_w * src_w),
                    int(fh / small_h * src_h)))
    n = len(paths)
    busy = (busy / max(n, 1)) if busy is not None else None
    # close the detector now — otherwise its __del__ fires at interpreter
    # shutdown and prints "Exception ignored ..." noise into the logs.
    if mp_det is not None:
        try:
            mp_det.close()
        except Exception:
            pass

    # median face box (stable across the clip)
    face = None
    if face_boxes:
        face = (int(statistics.median([b[0] for b in face_boxes])),
                int(statistics.median([b[1] for b in face_boxes])),
                int(statistics.median([b[2] for b in face_boxes])),
                int(statistics.median([b[3] for b in face_boxes])))

    # still rectangles: present in >= 75% of samples, containing the face
    still = [(r, c) for r, c in rect_votes.items() if c >= max(2, int(0.75 * n))]
    still.sort(key=lambda rc: -rc[1])
    for rect, _count in still:
        x, y, w, h = rect
        if face and x <= face[0] <= x + w and y <= face[1] <= y + h:
            return {"rect": rect, "method": "still-border", "face": face,
                    "samples": n, "busy": busy.tolist() if busy is not None else None}
    if still:
        # a still rectangle with no face inside is probably a HUD panel,
        # not a webcam — do NOT use it. Fall through to the face box.
        pass
    if face:
        cx, cy, fw, fh = face
        w = int(fw * 2.2)
        h = int(w * 9 / 16)
        x = max(0, min(src_w - w, cx - w // 2))
        y = max(0, min(src_h - h, cy - h // 2))
        return {"rect": (x, y, w, h), "method": "face-box", "face": face,
                "samples": n, "busy": busy.tolist() if busy is not None else None}
    return None


def detect_facecam(video_path, src_w=1920, src_h=1080, n=8, model_path=None):
    """Sample n frames from the input video with ffmpeg and find the
    streamer's facecam rectangle in source coords (see
    _detect_facecam_from_frames). Returns the detection dict or None.
    model_path: BlazeFace .tflite asset (default: the bundled model).
    Never raises — any failure returns None so the caller falls back.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
        cap.release()
        duration = total / fps if total > 0 and fps > 0 else 0.0
    except Exception:
        return None
    if duration <= 0:
        duration = 10.0
    # sample timestamps spread across the clip (skip the very first frame)
    stamps = [duration * (i + 1) / (n + 1) for i in range(n)]
    ffmpeg = find_ffmpeg()
    tmp_dir = tempfile.mkdtemp(prefix="cfb_layout_")
    try:
        for i, t in enumerate(stamps):
            out = os.path.join(tmp_dir, f"f_{i:04d}.jpg")
            try:
                subprocess.run(
                    [ffmpeg, "-y", "-v", "error", "-ss", f"{t:.2f}",
                     "-i", str(video_path), "-frames:v", "1", out],
                    capture_output=True, check=True, timeout=30)
            except Exception:
                continue
        return _detect_facecam_from_frames(tmp_dir, src_w, src_h,
                                           model_path=model_path)
    except Exception:
        return None
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def choose_layout(detection, src_w=1920, src_h=1080, cap_y_default=CAP_Y_DEFAULT,
                  facecam_pos="auto", caption_pos="auto", ratio="9:16", res=720,
                  facecam_size="medium", facecam_band=50, facecam_band_w=100):
    """Pick the overlay corner + caption Y for the gameplay layout.

    detection: result of detect_facecam() (or None).
    facecam_pos / caption_pos: the user's picks ("auto" = the old
    auto-check behaviour; any other value OVERRIDES the auto choice).
    ratio/res scale the result to the output size.
    Returns a dict with overlay rect (OUTPUT coords), caption Y (output
    coords), the facecam source rect, and the reasons — everything the
    renderer and the report need. For the top-band layout the dict has
    "band": True and "band_h" instead of an overlay rect.
    """
    rp = ratio_params(ratio, res, facecam_size=facecam_size,
                      facecam_band=facecam_band, facecam_band_w=facecam_band_w)
    sx, sy = rp["out_w"] / 720.0, rp["out_h"] / 1280.0
    src_rect = detection["rect"] if detection else None

    def _scale_rect_720(rect):
        x, y, w, h = rect
        return (int(x * sx), int(y * sy), int(w * sx), int(h * sy))

    # --- band layouts (Clipie-style): facecam as a horizontal strip,
    # gameplay on the other side. Caption defaults to the seam (the edge
    # where band meets game) — computed from the actual band size, not
    # the fixed 430: with the 50% band the old fixed seam sat INSIDE
    # the band.
    if facecam_pos in ("top-band", "bottom-band"):
        band_pos = "top" if facecam_pos == "top-band" else "bottom"
        cap = caption_pos if caption_pos != "auto" else "seam"
        if cap == "seam":
            if band_pos == "top":
                cap_y = rp["band_h"] + int(16 * sy)
            else:
                cap_y = rp["out_h"] - rp["band_h"] - rp["strip_h"]
        else:
            cap_y = caption_y(cap, rp)
        return {"mode": "gameplay-band",
                "facecam_src_rect": src_rect,
                "detection_method": detection["method"] if detection else None,
                "overlay": None, "band": True,
                "band_pos": band_pos,
                "band_h": rp["band_h"], "band_w": rp["band_w"],
                "corner": facecam_pos,
                "caption_y": cap_y,
                "caption_y_default": rp["cap_y_default"],
                "corner_scores": [],
                "facecam_bigger_than_peakcut": True,
                "note": (f"{band_pos}-band: facecam as a {rp['band_w']}x"
                         f"{rp['band_h']} strip, gameplay on the other side")}
    # --- stage layout FAMILY: compact facecam band on top, gameplay as
    # large as possible in the middle (full height, side-crop only),
    # captions in a tight empty zone at the bottom.
    if facecam_pos in ("stage", "stage-nofacecam", "stage-topleft",
                       "stage-topright", "stage-side"):
        geom = stage_geometry(rp, facecam_pos)
        cap = caption_pos if caption_pos != "auto" else "bottom"
        cap_y = caption_y(cap, rp)
        cap_band = stage_caption_band(cap, rp)  # (y, h) or None
        nofc = facecam_pos == "stage-nofacecam"
        return {"mode": "gameplay-stage",
                "facecam_src_rect": None if nofc else src_rect,
                "detection_method": detection["method"] if detection else None,
                "overlay": None, "band": True,
                "band_pos": "top",
                "band_h": geom["band_h"], "band_w": rp["out_w"],
                "stage": True, "stage_geom": geom,
                "corner": facecam_pos,
                "caption_y": cap_y,
                "caption_band": cap_band,
                "caption_y_default": rp["cap_y_default"],
                "corner_scores": [],
                "facecam_bigger_than_peakcut": True,
                "note": (f"stage without facecam: game fills the full "
                         f"frame, captions over blurred game below"
                         if nofc else
                         f"stage {facecam_pos}: facecam "
                         f"{geom['fc'][2]}x{geom['fc'][3]} "
                         f"gold-framed, game {rp['out_w']}x{geom['game_h']} "
                         f"({geom['game_frac_h']:.0%} of frame), captions "
                         f"over blurred game below")}
    # --- facecam off: no overlay at all, captions at the chosen position.
    if facecam_pos == "off":
        cap = caption_pos if caption_pos != "auto" else "bottom"
        return {"mode": "gameplay",
                "facecam_src_rect": None,
                "detection_method": detection["method"] if detection else None,
                "overlay": None, "band": False,
                "corner": "off",
                "caption_y": caption_y(cap, rp),
                "caption_y_default": rp["cap_y_default"],
                "corner_scores": [],
                "facecam_bigger_than_peakcut": True,
                "note": "facecam off — plain game visuals"}
    # candidates: (name, x, y) — bottom corners excluded (platform UI zone)
    cands = [("top-right", OUT_W - FC_W - MARGIN, MARGIN),
             ("top-left", MARGIN, MARGIN),
             ("mid-right", OUT_W - FC_W - MARGIN, 420),
             ("mid-left", MARGIN, 420)]
    # where does the streamer's own facecam appear inside the base crop?
    # base crop = centre 608px of the source (x 656..1264) -> output 720 wide.
    CROP_W = 608
    crop_x0 = (src_w - CROP_W) // 2
    busy = None
    if detection:
        busy = detection.get("busy")

    def rect_overlap(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ox = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        oy = max(0, min(ay + ah, by + bh) - max(ay, by))
        return ox * oy

    def busy_score(rect):
        # mean edge density of the grid cells the rect covers (0..1)
        if busy is None:
            return 0.0
        gh, gw = len(busy), len(busy[0])
        x, y, w, h = rect
        # map output rect back to source coords for the grid
        sx = int(x / OUT_W * src_w)
        sy = int(y / OUT_H * src_h)
        sw = int(w / OUT_W * src_w)
        sh = int(h / OUT_H * src_h)
        cells = []
        for r in range(gh):
            for c in range(gw):
                cx0, cy0 = c * src_w // gw, r * src_h // gh
                cx1, cy1 = (c + 1) * src_w // gw, (r + 1) * src_h // gh
                if rect_overlap((sx, sy, sw, sh), (cx0, cy0, cx1 - cx0, cy1 - cy0)) > 0:
                    cells.append(busy[r][c])
        return sum(cells) / len(cells) if cells else 0.0

    # projected facecam rect in output coords (only the part inside the crop)
    proj = None
    if src_rect:
        rx, ry, rw, rh = src_rect
        vx0 = max(rx, crop_x0)
        vx1 = min(rx + rw, crop_x0 + CROP_W)
        if vx1 > vx0:
            proj = ((vx0 - crop_x0) / CROP_W * OUT_W,
                    ry / src_h * OUT_H,
                    (vx1 - vx0) / CROP_W * OUT_W,
                    rh / src_h * OUT_H)

    scored = []
    for name, ox, oy in cands:
        rect = (ox, oy, FC_W, FC_H)
        score = 0.0
        # 1. overlapping the streamer's own facecam in the base crop = bad
        if proj:
            ov = rect_overlap(rect, proj) / (FC_W * FC_H)
            score += ov * 100
        # 2. busy HUD zone = bad
        score += busy_score(rect) * 50
        # 3. top-right preferred on ties (PeakCut-like)
        score += {"top-right": 0, "top-left": 1, "mid-right": 2, "mid-left": 3}[name]
        scored.append((score, name, rect))
    scored.sort(key=lambda s: s[0])
    _score, corner, overlay = scored[0]
    # the user's pick overrides the auto choice (Positions page)
    if facecam_pos in ("top-right", "top-left", "mid-right", "mid-left",
                      "bottom-left", "bottom-right"):
        _kind, fixed = facecam_rect(facecam_pos, rp)
        # facecam_rect returns OUTPUT coords; convert back to 720-space
        # for the caption-Y overlap check below.
        overlay = (int(fixed[0] / sx), int(fixed[1] / sy),
                   int(fixed[2] / sx), int(fixed[3] / sy))
        corner = facecam_pos

    # caption Y: the user's pick wins; otherwise first free line (text band
    # must clear overlay + HUD)
    if caption_pos in ("bottom", "seam", "top"):
        cap_y = caption_y(caption_pos, rp)  # already output coords
        cap_y_720 = int(cap_y / sy)
    else:
        cap_y = cap_y_default
        for cand_y in CAP_Y_CANDIDATES:
            band = (0, cand_y + TEXT_TOP, OUT_W, TEXT_BOTTOM - TEXT_TOP)
            if rect_overlap(band, overlay) > 0:
                continue
            if busy_score(band) > 0.35:
                continue
            cap_y = cand_y
            break
        else:
            cap_y = CAP_Y_CANDIDATES[-1]  # 300 — highest free line we have
        cap_y_720 = cap_y
        cap_y = int(cap_y * sy)

    ov_out = _scale_rect_720(overlay)
    return {"mode": "gameplay",
            "facecam_src_rect": src_rect,
            "detection_method": detection["method"] if detection else None,
            "overlay": {"x": ov_out[0], "y": ov_out[1], "w": ov_out[2], "h": ov_out[3]},
            "band": False,
            "corner": corner,
            "caption_y": cap_y,
            "caption_y_default": int(cap_y_default * sy),
            "corner_scores": [(s[1], round(s[0], 1)) for s in scored],
            "facecam_bigger_than_peakcut": True,
            "ratio": ratio, "res": res,
            "note": ("overlay 288x162 = 40% of 720 width; PeakCut's was "
                     "323x186 in 1080x1920 = 30% of width")}


def fallback_layout(ratio="9:16", res=720, facecam_size="medium",
                    facecam_band=50, facecam_band_w=100,
                    facecam_pos="auto", caption_pos="auto"):
    """No detection deps / no facecam found: safe defaults, no guessing.
    Overlay + caption Y are in OUTPUT coords for the ratio/res.
    facecam_pos="stage" (and the other stage variants): the stage layout
    is still honoured when no facecam is found — it returns a stage
    layout with facecam_src_rect=None, so the renderer builds the stage
    look WITHOUT the facecam (full-frame game + blurred caption zone)
    instead of falling back to plain game mode.
    """
    rp = ratio_params(ratio, res, facecam_size=facecam_size,
                      facecam_band=facecam_band, facecam_band_w=facecam_band_w)
    if facecam_pos in ("stage", "stage-nofacecam", "stage-topleft",
                       "stage-topright", "stage-side"):
        geom = stage_geometry(rp, facecam_pos)
        cap = caption_pos if caption_pos != "auto" else "bottom"
        cap_y = caption_y(cap, rp)
        cap_band = stage_caption_band(cap, rp)  # (y, h) or None
        return {"mode": "gameplay-stage",
                "facecam_src_rect": None,
                "detection_method": None,
                "overlay": None, "band": True,
                "band_pos": "top",
                "band_h": geom["band_h"], "band_w": rp["out_w"],
                "stage": True, "stage_geom": geom,
                "corner": facecam_pos,
                "caption_y": cap_y,
                "caption_band": cap_band,
                "caption_y_default": rp["cap_y_default"],
                "corner_scores": [],
                "facecam_bigger_than_peakcut": True,
                "ratio": ratio, "res": res,
                "note": "stage without facecam — no facecam found, game fills the frame"}
    _kind, rect = facecam_rect("top-right", rp)
    return {"mode": "gameplay",
            "facecam_src_rect": None,
            "detection_method": None,
            "overlay": {"x": rect[0], "y": rect[1], "w": rect[2], "h": rect[3]},
            "band": False,
            "corner": "top-right",
            "caption_y": rp["cap_y_default"],
            "caption_y_default": rp["cap_y_default"],
            "corner_scores": [],
            "facecam_bigger_than_peakcut": True,
            "ratio": ratio, "res": res,
            "note": "fallback — detection unavailable or no facecam found"}
