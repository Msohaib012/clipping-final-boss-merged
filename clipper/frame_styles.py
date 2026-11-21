"""Facecam FRAME designs for the stage layout — ported from the GTA 6
Live Clipper donor (render_short.py FRAME_STYLES and friends).

"gold" is the original (default) — a golden comet travelling around
the rounded-rect frame, 4s per lap. Each design is a named, selectable
frame_style. kind: comet = travelling blob with fading tail (gold/red/
fire share the painter, colours + flicker differ) | line = a bright
arc of uniform brightness sweeping around (no tail) | static = one
still frame, no motion | pulse = the whole ring breathes in/out,
nothing travels. frames: PNGs in one loop (240 = 4s/lap at 60fps;
120 = 2s breathe; 1 = still image looped). ring_alpha: base ring
opacity (dim for the animated ones — the moving light is the star;
bright for static, where the ring IS the design).

Adaptation vs the donor: no BASE/global paths — _stage_frame_assets
takes work_dir as a parameter.
"""

import math
import os

# Facecam FRAME designs for the stage layout. "gold" is the original
# (default) — a golden comet travelling around the rounded-rect frame,
# 4s per lap.
FRAME_STYLES = {
    "gold": {
        "label": "Golden Comet",
        "desc": "Golden light travelling around the frame — the original, calm 4s lap.",
        "kind": "comet", "frames": 240,
        "ring": (255, 200, 0), "ring_alpha": 70,
        "comet": (255, 190, 0), "core": (255, 240, 180),
    },
    "red": {
        "label": "Red Comet",
        "desc": "PeakCut-style hype red comet travelling around the frame, 4s lap.",
        "kind": "comet", "frames": 240,
        "ring": (255, 60, 60), "ring_alpha": 70,
        "comet": (255, 45, 45), "core": (255, 225, 210),
    },
    "line": {
        "label": "Cyan Line",
        "desc": "A bright cyan line sweeping around the frame — no comet blob, just the line, 4s lap.",
        "kind": "line", "frames": 240,
        "ring": (0, 220, 255), "ring_alpha": 60,
        "comet": (0, 220, 255), "core": (210, 248, 255),
    },
    "fire": {
        "label": "Fire Ember",
        "desc": "Orange fire ember with a strong flicker and a long tail travelling around the frame, 4s lap.",
        "kind": "fire", "frames": 240,
        "ring": (255, 140, 0), "ring_alpha": 70,
        "comet": (255, 110, 0), "core": (255, 225, 130),
    },
    "static": {
        "label": "Clean Static",
        "desc": "No motion at all — a clean thin white/silver border, the minimal look.",
        "kind": "static", "frames": 1,
        "ring": (232, 236, 240), "ring_alpha": 235,
        "comet": (232, 236, 240), "core": (255, 255, 255),
    },
    "pulse": {
        "label": "Violet Pulse",
        "desc": "A violet neon ring that breathes in and out — no travelling light, 2s cycle.",
        "kind": "pulse", "frames": 120,
        "ring": (170, 90, 255), "ring_alpha": 180,
        "comet": (170, 90, 255), "core": (225, 200, 255),
    },
    # --- Structurally-different styles: these change the SHAPE /
    # structure, not just the colour.
    "circle": {
        "label": "Circle",
        "desc": "Facecam cropped to a circle with a thin gold ring and a travelling light — no box at all, the round look.",
        "kind": "circle", "frames": 240,
        "ring": (255, 200, 0), "ring_alpha": 110,
        "comet": (255, 190, 0), "core": (255, 240, 180),
        # a circle needs a SQUARE facecam (180x180 at 720p, fits the
        # 188px top band); fc_y is the 720p top margin for it.
        "fc_size": (180, 180), "fc_y": 14,
    },
    "clean": {
        "label": "Clean (no frame)",
        "desc": "No border at all — just a soft drop shadow under the facecam, the minimal floating look.",
        "kind": "clean", "frames": 1,
        "ring": (0, 0, 0), "ring_alpha": 0,
        "comet": (0, 0, 0), "core": (255, 255, 255),
    },
    "bold": {
        "label": "Bold Chunky",
        "desc": "A much thicker chunky border (10px) instead of the thin line — the heavy frame look.",
        "kind": "static", "frames": 1,
        "ring": (255, 200, 0), "ring_alpha": 235,
        "comet": (255, 200, 0), "core": (255, 240, 180),
        "border": 10,  # override: chunky 10px border (standard is 4)
    },
    "corners": {
        "label": "Viewfinder Corners",
        "desc": "Viewfinder style — only the four corner brackets, no full border.",
        "kind": "corners", "frames": 1,
        "ring": (255, 200, 0), "ring_alpha": 235,
        "comet": (255, 200, 0), "core": (255, 240, 180),
    },
    "big": {
        "label": "Big Facecam",
        "desc": "A larger facecam (360x203) with a thin gold frame — the streamer front and centre.",
        "kind": "static", "frames": 1,
        "ring": (255, 200, 0), "ring_alpha": 200,
        "comet": (255, 200, 0), "core": (255, 240, 180),
        # larger facecam (360x203 at 720p); it floats over the top of
        # the game by ~29px — the game area is NOT moved.
        "fc_size": (360, 203),
    },
}
DEFAULT_FRAME_STYLE = "gold"


def _rounded_rect_perimeter(fw, fh, r, step=3.0):
    """Sample points (x, y) clockwise around a rounded rect [0,fw]x[0,fh]
    with corner radius r, roughly `step` px apart. Used to drive the
    travelling golden light around the stage facecam frame."""
    pts = []

    def edge(x0, y0, x1, y1):
        L = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(L / step))
        for i in range(n):
            t = i / n
            pts.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))

    def arc(cx, cy, a0, a1):
        # step so arc length per point is ~step px
        da = math.degrees(step / max(r, 1.0))
        a = a0
        while (a <= a1 if a1 > a0 else a >= a1):
            rad = math.radians(a)
            pts.append((cx + r * math.cos(rad), cy + r * math.sin(rad)))
            a += da if a1 > a0 else -da

    edge(r, 0, fw - r, 0)            # top
    arc(fw - r, r, -90, 0)           # top-right corner
    edge(fw, r, fw, fh - r)          # right
    arc(fw - r, fh - r, 0, 90)       # bottom-right corner
    edge(fw - r, fh, r, fh)          # bottom
    arc(r, fh - r, 90, 180)          # bottom-left corner
    edge(0, fh - r, 0, r)            # left
    arc(r, r, 180, 270)              # top-left corner
    return pts


def _frame_style_geom(sg, frame_style, facecam_pos, out_w, out_h):
    """Apply a frame style's GEOMETRY overrides to a stage geometry dict.
    Returns (fc_rect, frame_rect, border) — the facecam rect (x,y,w,h),
    the frame rect (x,y,w,h), and the border thickness to build the
    frame assets with. Styles without overrides return the standard
    geometry unchanged. The GAME area is NOT moved: a bigger facecam
    (big) simply floats over the top of the game by ~29px — the
    facecam is an overlay, and the overlap reads as intentional. The
    anchor follows the facecam position (centre stays centred, left
    stays left, right stays right, side stays vertically centred)."""
    spec = FRAME_STYLES.get(frame_style) or FRAME_STYLES[DEFAULT_FRAME_STYLE]
    s = out_h / 1280.0
    fc_x, fc_y, fc_w, fc_h = sg["fc"]
    border = sg["border"]
    glow_pad = sg["glow_pad"]
    top_margin = sg["top_margin"]
    if "fc_size" in spec:
        w720, h720 = spec["fc_size"]
        fc_w = max(2, int(w720 * s))
        fc_h = max(2, int(h720 * s))
        fc_w, fc_h = (fc_w // 2) * 2, (fc_h // 2) * 2
        if facecam_pos == "stage-topleft":
            fc_x = top_margin
        elif facecam_pos == "stage-topright":
            fc_x = out_w - fc_w - top_margin
        elif facecam_pos == "stage-side":
            fc_x = top_margin
        else:  # "stage" (top-centre) and anything else: centred
            fc_x = (out_w - fc_w) // 2
        if facecam_pos == "stage-side":
            fc_y = (out_h - fc_h) // 2
        elif "fc_y" in spec:
            fc_y = int(spec["fc_y"] * s)
        else:
            fc_y = top_margin
    if "border" in spec:
        border = max(3, int(spec["border"] * s))
    ext = border + glow_pad
    # keep the frame fully on-canvas: a thick border (bold) or a low
    # top margin (circle) can push the frame rect off the top/left
    # edge, clipping the ring — shift the facecam in until it fits.
    if fc_y < ext:
        fc_y = ext
    if fc_x < ext:
        fc_x = ext
    frame_rect = (fc_x - ext, fc_y - ext, fc_w + 2 * ext, fc_h + 2 * ext)
    return (fc_x, fc_y, fc_w, fc_h), frame_rect, border


def _paint_frame(kind, f, n, geom, spec):
    """Paint ONE frame of the stage facecam frame sequence (RGBA).
    kind: comet | line | fire | static | pulse | circle | clean |
    corners (see FRAME_STYLES).
    geom: {"fw","fh","ring","base","perim","n_per","trail","core_r",
           "border","glow_pad"} — shared geometry from _stage_frame_assets.
    spec: the FRAME_STYLES entry (colours).
    - comet: travelling blob with a fading tail + white-hot core + soft
      halo (the original golden logic, colours from spec).
    - fire: same travelling idea, but a longer tail, a bigger blob and a
      strong double-sine flicker — reads as flame, not light.
    - line: a bright arc of FIXED length and UNIFORM brightness sweeping
      around — no tail fade, no blob — reads as a rotating line.
    - static: just the base ring (the design is the ring itself); the
      caller only ever asks for frame 0 of 1.
    - pulse: the whole ring breathes — alpha oscillates 0.35..1 over the
      loop, with a soft halo that breathes with it; nothing travels.
    - circle: like comet (travelling blob + tail), but the perimeter in
      geom is a CIRCLE — the facecam is round, so the light travels
      around the round frame. (Mask/ring are circular too, built by
      _stage_frame_assets.)
    - clean: NO ring at all — just a soft drop shadow under the facecam
      (a dark blurred rounded rect, offset down-right, drawn under the
      facecam by the filter order). The caller only asks for frame 0.
    - corners: viewfinder style — only the four corner brackets, no
      full ring. The caller only asks for frame 0."""
    from PIL import Image, ImageDraw, ImageFilter
    fw, fh = geom["fw"], geom["fh"]
    base = geom["base"]
    if kind == "static":
        return base
    if kind == "clean":
        # drop shadow only: a dark rounded rect, offset down-right by
        # ~6px and blurred, sitting under the facecam (the filter
        # overlays the glow BEFORE the facecam, so the shadow peeks out
        # behind its bottom-right edge). No ring, no border.
        ext = geom["border"] + geom["glow_pad"]
        sh = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
        sd = ImageDraw.Draw(sh)
        off = max(4, geom["glow_pad"] // 2)
        sd.rounded_rectangle(
            [ext + off, ext + off,
             fw - 1 - (ext - off), fh - 1 - (ext - off)],
            radius=geom.get("radius", 18), fill=(0, 0, 0, 110))
        return sh.filter(ImageFilter.GaussianBlur(radius=6))
    if kind == "corners":
        # viewfinder: four L-shaped brackets hugging the facecam corners,
        # no full ring. Bracket length ~44px, thickness = border + 2.
        ext = geom["border"] + geom["glow_pad"]
        bw = max(4, geom["border"] + 2)
        bl = 44
        img = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        col = spec["ring"] + (spec["ring_alpha"],)
        x0, y0 = ext, ext
        x1, y1 = fw - 1 - ext, fh - 1 - ext
        # each bracket = two bars forming an L
        for (cx, cy, sx, sy) in [(x0, y0, 1, 1), (x1, y0, -1, 1),
                                 (x0, y1, 1, -1), (x1, y1, -1, -1)]:
            # horizontal bar
            hx0 = min(cx, cx + sx * bl)
            hx1 = max(cx, cx + sx * bl)
            d.rectangle([hx0, cy - bw // 2 if sy > 0 else cy - bw // 2,
                         hx1, cy + bw // 2], fill=col)
            # vertical bar
            vy0 = min(cy, cy + sy * bl)
            vy1 = max(cy, cy + sy * bl)
            d.rectangle([cx - bw // 2, vy0, cx + bw // 2, vy1], fill=col)
        halo = img.filter(ImageFilter.GaussianBlur(
            radius=max(3, geom["glow_pad"] // 2)))
        return Image.alpha_composite(halo, img)
    if kind == "pulse":
        # breathe: 0.35..1 over one loop (120 frames = 2s at 60fps)
        ph = 0.5 + 0.5 * math.sin(2 * math.pi * f / n)
        a = 0.35 + 0.65 * ph
        alpha = base.split()[3].point(lambda v: int(v * a))
        frame = Image.merge("RGBA", base.split()[:3] + (alpha,))
        halo = frame.filter(ImageFilter.GaussianBlur(
            radius=max(3, geom["glow_pad"] // 2)))
        return Image.alpha_composite(halo, frame)
    perim, n_per = geom["perim"], geom["n_per"]
    head = int((f / n) * n_per) % n_per
    if kind == "line":
        arc_len = max(10, int(n_per * 0.25))  # 25% of the perimeter
        lw = max(3, geom["border"] + 1)
        line_img = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
        ld = ImageDraw.Draw(line_img)
        for k in range(arc_len):
            idx = (head - k) % n_per
            px, py = perim[idx]
            ld.ellipse([px - lw, py - lw, px + lw, py + lw],
                       fill=spec["comet"] + (235,))
        hx, hy = perim[head]
        ld.ellipse([hx - lw - 2, hy - lw - 2, hx + lw + 2, hy + lw + 2],
                   fill=spec["core"] + (255,))
        halo = line_img.filter(ImageFilter.GaussianBlur(
            radius=max(3, geom["glow_pad"] // 2)))
        frame = Image.alpha_composite(base, halo)
        return Image.alpha_composite(frame, line_img)
    # comet / fire: travelling blob with fading tail
    if kind == "fire":
        trail = max(8, int(n_per * 0.30))  # longer tail than comet
        core_r = geom["core_r"] + 2
        flick = 0.75 + 0.25 * math.sin(f * 0.9) * math.sin(f * 0.37)
    else:
        trail = geom["trail"]
        core_r = geom["core_r"]
        flick = 0.88 + 0.12 * math.sin(f * 0.55)
    comet = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
    cd = ImageDraw.Draw(comet)
    for k in range(trail, -1, -1):
        idx = (head - k) % n_per
        px, py = perim[idx]
        a = int(240 * flick * (1.0 - k / (trail + 1)) ** 1.3)
        rr = max(2, int(core_r * (0.6 + 0.4 * (1.0 - k / (trail + 1)))))
        cd.ellipse([px - rr, py - rr, px + rr, py + rr],
                   fill=spec["comet"] + (a,))
    hx, hy = perim[head]
    cd.ellipse([hx - core_r, hy - core_r, hx + core_r, hy + core_r],
               fill=spec["core"] + (255,))
    halo = comet.filter(ImageFilter.GaussianBlur(
        radius=max(3, geom["glow_pad"] // 2)))
    frame = Image.alpha_composite(base, halo)
    return Image.alpha_composite(frame, comet)


def _stage_frame_assets(fc_w, fc_h, border, radius, glow_pad, work_dir,
                        glow_frames=None, frame_style="gold"):
    """Rounded-corner mask + facecam FRAME for the stage layout.

    frame_style picks one of FRAME_STYLES (default "gold"); unknown
    values fall back to gold. glow_frames defaults to the style's own
    loop length (240 = 4s/lap for the travelling designs, 120 = 2s
    breathe for pulse, 1 = still for static).

    mask.png: fc_w x fc_h, white rounded rect on transparent — used by
    alphamerge to round the facecam video's corners.
    stage_glow_%03d.png: a looping PNG sequence (glow_frames frames,
    one full lap around the frame per loop for the travelling kinds) —
    a dim base ring in the style's colour + the style's moving element
    (comet / line / fire) or the breathing ring (pulse) or just the
    ring (static). Returns (mask_path, glow_pattern, glow_frames) where
    glow_pattern is the ffmpeg image-sequence path
    (e.g. .../stage_glow_%03d.png). work_dir is created if needed.
    """
    from PIL import Image, ImageDraw, ImageFilter, ImageChops
    os.makedirs(work_dir, exist_ok=True)
    spec = FRAME_STYLES.get(frame_style) or FRAME_STYLES[DEFAULT_FRAME_STYLE]
    if glow_frames is None:
        glow_frames = spec["frames"]
    ext = border + glow_pad
    fw, fh = fc_w + 2 * ext, fc_h + 2 * ext
    is_circle = (spec["kind"] == "circle")
    # --- mask: rounded rect, white inside (circle: a true circle) ---
    mask = Image.new("RGBA", (fc_w, fc_h), (0, 0, 0, 0))
    if is_circle:
        ImageDraw.Draw(mask).ellipse(
            [0, 0, fc_w - 1, fc_h - 1], fill=(255, 255, 255, 255))
    else:
        ImageDraw.Draw(mask).rounded_rectangle(
            [0, 0, fc_w - 1, fc_h - 1], radius=radius,
            fill=(255, 255, 255, 255))
    mask_path = os.path.join(work_dir, "stage_mask.png")
    mask.save(mask_path)
    # --- ring mask: THIN ring (border px) hugging the facecam edge,
    # NOT a thick solid band. Facecam sits at [ext,ext] inside the
    # frame image; the ring is the band from the facecam edge outward
    # by `border`. (circle: the ring is the band between two concentric
    # circles.)
    if is_circle:
        cx, cy = fw / 2.0, fh / 2.0
        r_in = fc_w / 2.0
        r_out = r_in + border
        outer = Image.new("L", (fw, fh), 0)
        ImageDraw.Draw(outer).ellipse(
            [cx - r_out, cy - r_out, cx + r_out, cy + r_out], fill=255)
        inner = Image.new("L", (fw, fh), 0)
        ImageDraw.Draw(inner).ellipse(
            [cx - r_in, cy - r_in, cx + r_in, cy + r_in], fill=255)
        ring = ImageChops.subtract(outer, inner)
    else:
        outer = Image.new("L", (fw, fh), 0)
        ImageDraw.Draw(outer).rounded_rectangle(
            [ext - border, ext - border,
             fw - 1 - (ext - border), fh - 1 - (ext - border)],
            radius=radius + border, fill=255)
        inner = Image.new("L", (fw, fh), 0)
        ImageDraw.Draw(inner).rounded_rectangle(
            [ext, ext, fw - 1 - ext, fh - 1 - ext], radius=radius, fill=255)
        ring = ImageChops.subtract(outer, inner)
    # base ring in the style's colour (dim for the animated kinds — the
    # moving light is the star; bright for static, where the ring IS
    # the design)
    base = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
    base.paste(Image.new("RGBA", (fw, fh),
                         spec["ring"] + (spec["ring_alpha"],)), (0, 0), ring)
    # perimeter points (clockwise) along the ring's centreline, for the
    # travelling kinds: the centreline rect is inset by (ext - border/2)
    # from the frame edge, radius radius + border/2.
    # (circle: points on the circle of radius fc/2 + border/2.)
    if is_circle:
        cx, cy = fw / 2.0, fh / 2.0
        r_mid = fc_w / 2.0 + border / 2.0
        n_per = max(24, int(2 * math.pi * r_mid / 3.0))
        perim = [(cx + r_mid * math.cos(2 * math.pi * i / n_per),
                  cy + r_mid * math.sin(2 * math.pi * i / n_per))
                 for i in range(n_per)]
    else:
        _cl = ext - border // 2
        perim = _rounded_rect_perimeter(
            fw - 2 * _cl, fh - 2 * _cl, radius + border // 2, step=3.0)
        perim = [(x + _cl, y + _cl) for (x, y) in perim]
        n_per = len(perim)
    trail = max(8, int(n_per * 0.22))   # comet arc ~22% of the perimeter
    core_r = max(4, border + 3)         # comet blob radius (bigger = clearer)
    geom = {"fw": fw, "fh": fh, "ring": ring, "base": base,
            "perim": perim, "n_per": n_per, "trail": trail,
            "core_r": core_r, "border": border, "glow_pad": glow_pad,
            "radius": radius}
    glow_pattern = os.path.join(work_dir, "stage_glow_%03d.png")
    for f in range(glow_frames):
        frame = _paint_frame(spec["kind"], f, glow_frames, geom, spec)
        frame.save(glow_pattern % (f + 1))
    return mask_path, glow_pattern, glow_frames
