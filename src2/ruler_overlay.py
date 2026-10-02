"""
ruler_overlay.py
Synthetic ruler overlay for building the corrected (style-matched) ruler
CAV concept set -- see concept_images_ruler_matched/.

Four categories, mixed by default in these rough proportions (the exact
split for the concept-set build is enforced deterministically in
build_ruler_matched_concepts.py, not left to this module's random
dispatch):
  ~10% "edge" + style "strip"  -- thick/prominent ruler entering from one
      image edge (Thick_Ruler-style). Rare in the real data (9 of 940
      ruler images), so kept rare here too.
  ~30% "lesion"                -- ticks a few mm/cm from one side of the
      lesion, pointing back toward it, mirroring add_synthetic_ruler() in
      ruler_bias_counterfactual_test.py -- but never touching the lesion.
  ~40% "edge" + style "ticks"  -- faint ruler entering from one image
      edge, elsewhere on the skin (Normal_Ruler-style), no baseline scale
      line, ticks pointing inward toward the lesion.
  ~20% "short_ruler"           -- a single short ruler (3-4 scale marks,
      not a full row) entering from one image edge or corner, bigger and
      bolder than the "ticks" style but smaller than "strip".

None of the four ever overlaps the lesion bbox -- every mode is checked
(or geometrically capped) against it.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage


def find_lesion_bbox(pil_img):
    """Rough lesion bbox: darkest/brownest blob nearest image center,
    thin hair-like structures removed via binary opening. Same approach
    as ruler_bias_counterfactual_test.py -- kept local here so this
    module has no TF/model dependency.
    """
    img = np.array(pil_img.convert('RGB')).astype(np.float32)
    h, w, _ = img.shape
    gray = img.mean(axis=2)

    mask = gray < np.percentile(gray, 35)
    mask = ndimage.binary_opening(mask, structure=np.ones((5, 5)))
    labeled, n_labels = ndimage.label(mask)

    if n_labels == 0:
        return (w // 4, h // 4, 3 * w // 4, 3 * h // 4)

    center_label = labeled[h // 2, w // 2]
    if center_label != 0:
        best_label = center_label
    else:
        sizes = ndimage.sum(mask, labeled, range(1, n_labels + 1))
        best_label = int(np.argmax(sizes)) + 1

    y_slice, x_slice = ndimage.find_objects(labeled == best_label)[0]
    return (x_slice.start, y_slice.start, x_slice.stop, y_slice.stop)


def _band_bbox(p0, u, n, span, inward_extent, outward_extent):
    """Axis-aligned bounding box of the ruler's rendered band -- baseline
    +/- span along u, extended by inward_extent toward the lesion side
    and outward_extent away from it. Conservative under rotation (the
    small +/-7deg angle only ever makes the true band smaller than this).
    """
    corners = np.array([
        p0 - u * span + n * inward_extent,
        p0 + u * span + n * inward_extent,
        p0 - u * span - n * outward_extent,
        p0 + u * span - n * outward_extent,
    ])
    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    return x0, y0, x1, y1


def _bbox_overlap(a, b, margin=0.0):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    bx0, by0, bx1, by1 = bx0 - margin, by0 - margin, bx1 + margin, by1 + margin
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def _draw_edge_ruler(edge, style, rng, scale, w, h):
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # baseline along the chosen edge: direction u, inward normal n
    ang = np.deg2rad(rng.uniform(-7, 7))
    if edge in ("bottom", "top"):
        u = np.array([np.cos(ang), np.sin(ang)])
        n = np.array([0.0, -1.0]) if edge == "bottom" else np.array([0.0, 1.0])
        inset = rng.uniform(0.02, 0.08) * scale
        y0 = h - inset if edge == "bottom" else inset
        p0 = np.array([w / 2 + rng.uniform(-0.1, 0.1) * w, y0])
        span = w * 1.2
    else:
        u = np.array([np.sin(ang), np.cos(ang)])
        n = np.array([1.0, 0.0]) if edge == "left" else np.array([-1.0, 0.0])
        inset = rng.uniform(0.02, 0.08) * scale
        x0 = inset if edge == "left" else w - inset
        p0 = np.array([x0, h / 2 + rng.uniform(-0.1, 0.1) * h])
        span = h * 1.2

    if style == "strip":
        depth = rng.uniform(0.08, 0.14) * scale          # visible ruler width
        # white body of the ruler, extending outward past the image edge
        a = p0 - u * span
        b = p0 + u * span
        body = [tuple(a), tuple(b), tuple(b - n * (depth + scale)),
                tuple(a - n * (depth + scale))]
        shade = int(rng.uniform(225, 245))
        d.polygon(body, fill=(shade, shade, int(shade * 0.98), 255))
        # thick mm ticks hanging from the baseline into the strip
        spacing = rng.uniform(0.020, 0.028) * scale
        tick_len = rng.uniform(0.55, 0.75) * depth
        tick_w = max(3, int(0.007 * scale))
        t = -span + rng.uniform(0, spacing)
        i = 0
        while t < span:
            q = p0 + u * t
            ln = tick_len * (1.35 if i % 5 == 0 else 1.0)
            d.line([tuple(q), tuple(q - n * ln)], fill=(15, 15, 20, 255),
                   width=tick_w)
            t += spacing
            i += 1
        blur = rng.uniform(0.5, 1.0)
        # ticks/body point outward (away from center); the baseline itself
        # is the only part that can reach toward the lesion
        band = _band_bbox(p0, u, n, span, inward_extent=0.0,
                           outward_extent=depth + scale)
    else:  # "ticks"
        spacing = rng.uniform(0.014, 0.026) * scale
        tick_len = rng.uniform(0.018, 0.040) * scale
        tick_w = max(2, int(rng.uniform(0.0020, 0.0035) * scale))
        shade = int(rng.uniform(10, 60))
        alpha = int(rng.uniform(150, 230))
        row_len = rng.uniform(0.35, 0.95) * span * 0.8
        t0 = rng.uniform(-0.3, 0.3) * span / 2
        t = t0 - row_len / 2
        i = 0
        while t < t0 + row_len / 2:
            q = p0 + u * t
            ln = tick_len * (1.4 if i % 5 == 0 else 1.0)
            d.line([tuple(q), tuple(q + n * ln)],
                   fill=(shade, shade, shade, alpha), width=tick_w)
            t += spacing
            i += 1
        blur = rng.uniform(0.8, 1.6)
        # ticks point inward, toward the lesion -- that's the reach that matters
        row_span = row_len / 2 + abs(t0)
        band = _band_bbox(p0, u, n, row_span,
                           inward_extent=tick_len * 1.4, outward_extent=0.0)

    overlay = overlay.filter(ImageFilter.GaussianBlur(blur))
    return overlay, band


def _draw_lesion_ruler(lesion_bbox, rng, w, h, contact=None):
    """Ticks on one side of the lesion bbox, always pointing back toward
    it (never outward into blank skin -- that wouldn't read as a ruler
    measuring anything).

    contact "side" (the default): short ticks that stay strictly within
        the gap -- close to the lesion but never touching it. This is
        what the concept-set build always uses.
    contact "alongside": long ticks that cross the gap and reach across
        the lesion boundary, the way add_synthetic_ruler() in
        ruler_bias_counterfactual_test.py lays a ruler flush against the
        lesion for measurement. Not used by default -- pass explicitly
        if you specifically want the overlapping variant.
    """
    if contact is None:
        contact = "side"

    x0, y0, x1, y1 = lesion_bbox
    sides = []
    if x0 > 0.05 * w:
        sides.append("left")
    if w - x1 > 0.05 * w:
        sides.append("right")
    if y0 > 0.05 * h:
        sides.append("top")
    if h - y1 > 0.05 * h:
        sides.append("bottom")
    if not sides:
        sides = ["left", "right", "top", "bottom"]
    side = sides[rng.integers(0, len(sides))]

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    scale = min(w, h)
    if contact == "alongside":
        # small gap, long ticks -- crosses the gap and reaches across the
        # lesion boundary, same as the counterfactual test's ruler
        gap = max(2, scale // 60)
        tick_len_major = rng.uniform(0.045, 0.065) * scale
    else:  # "side" -- gap set well clear of a normal, clearly-visible tick
        # (same tick-length range as the edge "ticks" style), plus a fixed
        # margin so Gaussian blur bleed can't reach the lesion either
        BLUR_MARGIN = max(4, int(0.006 * scale))
        tick_len_major = rng.uniform(0.018, 0.035) * scale
        gap = tick_len_major + BLUR_MARGIN + rng.uniform(0.01, 0.03) * scale
    tick_len_minor = tick_len_major / 2
    tick_w = max(2, int(rng.uniform(0.0020, 0.0035) * scale))
    shade = int(rng.uniform(10, 45))
    alpha = int(rng.uniform(170, 235))
    tick_color = (shade, shade, shade, alpha)
    edge_color = (shade + 30, shade + 30, shade + 30, int(alpha * 0.7))

    # ticks always point back toward the lesion, same as a real ruler laid
    # next to it -- "side" vs "alongside" differ in reach (tick_len above),
    # not direction, so a "side" ruler never reads as pointing away from
    # the subject it's supposedly measuring
    if side in ("left", "right"):
        y_start, y_end = max(0, y0), min(h, y1)
        if y_end <= y_start:
            y_start, y_end = 0, h
        rx = min(w - 1, x1 + gap) if side == "right" else max(0, x0 - gap)
        sign = -1 if side == "right" else 1
        d.line([rx, y_start, rx, y_end], fill=edge_color, width=2)
        tick_spacing = max(6, int((y_end - y_start) // 15))
        for i, y in enumerate(range(int(y_start), int(y_end), tick_spacing)):
            tick_len = tick_len_major if i % 5 == 0 else tick_len_minor
            d.line([rx, y, rx + sign * tick_len, y], fill=tick_color, width=tick_w)
    else:
        x_start, x_end = max(0, x0), min(w, x1)
        if x_end <= x_start:
            x_start, x_end = 0, w
        ry = min(h - 1, y1 + gap) if side == "bottom" else max(0, y0 - gap)
        sign = -1 if side == "bottom" else 1
        d.line([x_start, ry, x_end, ry], fill=edge_color, width=2)
        tick_spacing = max(6, int((x_end - x_start) // 15))
        for i, x in enumerate(range(int(x_start), int(x_end), tick_spacing)):
            tick_len = tick_len_major if i % 5 == 0 else tick_len_minor
            d.line([x, ry, x, ry + sign * tick_len], fill=tick_color, width=tick_w)

    overlay = overlay.filter(ImageFilter.GaussianBlur(rng.uniform(0.8, 1.6)))
    return overlay


def _draw_ruler_fragment(edge, rng, scale, w, h):
    """A single short, clearly-visible ruler -- just 3-4 scale marks, not
    a row spanning the whole side -- near one edge, biased toward one end
    of that edge so it reads as glimpsed from a corner.
    """
    ang = np.deg2rad(rng.uniform(-10, 10))
    if edge in ("bottom", "top"):
        u = np.array([np.cos(ang), np.sin(ang)])
        n = np.array([0.0, -1.0]) if edge == "bottom" else np.array([0.0, 1.0])
        inset = rng.uniform(0.02, 0.08) * scale
        y0 = h - inset if edge == "bottom" else inset
        corner_sign = -1 if rng.random() < 0.5 else 1
        p0 = np.array([w / 2 + corner_sign * rng.uniform(0.30, 0.45) * w, y0])
    else:
        u = np.array([np.sin(ang), np.cos(ang)])
        n = np.array([1.0, 0.0]) if edge == "left" else np.array([-1.0, 0.0])
        inset = rng.uniform(0.02, 0.08) * scale
        x0 = inset if edge == "left" else w - inset
        corner_sign = -1 if rng.random() < 0.5 else 1
        p0 = np.array([x0, h / 2 + corner_sign * rng.uniform(0.30, 0.45) * h])

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # sized between the faint ticks style and the thick ruler strip --
    # bigger/bolder than a normal tick mark, but well short of a strip
    spacing = rng.uniform(0.020, 0.032) * scale
    tick_len = rng.uniform(0.040, 0.058) * scale
    tick_w = max(3, int(rng.uniform(0.0040, 0.0055) * scale))
    shade = int(rng.uniform(10, 55))
    alpha = int(rng.uniform(190, 240))  # stronger than a full row -- a
    # short fragment needs to read clearly as "ruler" on its own
    n_ticks = int(rng.integers(3, 5))   # just 3-4 scale marks, not a row
    row_len = spacing * (n_ticks - 1)

    t = -row_len / 2
    for i in range(n_ticks):
        q = p0 + u * t
        ln = tick_len * (1.4 if i % 4 == 0 else 1.0)
        d.line([tuple(q), tuple(q + n * ln)], fill=(shade, shade, shade, alpha),
               width=tick_w)
        t += spacing

    overlay = overlay.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.2)))
    band = _band_bbox(p0, u, n, row_len / 2, inward_extent=tick_len * 1.4,
                       outward_extent=0.0)
    return overlay, band


def _draw_short_ruler(lesion_bbox, rng, w, h, scale, avoid_lesion=True,
                       lesion_margin=0.03):
    """Exactly one short ruler (3-4 scale marks, see _draw_ruler_fragment)
    entering from one edge or corner -- one ruler per image, not several.
    Tries the other edges rather than settling for one whose band
    overlaps the lesion.
    """
    margin = lesion_margin * scale
    edge_order = list(rng.permutation(["bottom", "top", "left", "right"]))
    fallback = None
    for e in edge_order:
        overlay, band = _draw_ruler_fragment(e, rng, scale, w, h)
        if fallback is None:
            fallback = overlay
        if not avoid_lesion or not _bbox_overlap(band, lesion_bbox, margin):
            return overlay
    return fallback


def add_ruler(img, style=None, edge=None, seed=None, placement=None,
              contact=None, lesion_bbox=None, avoid_lesion=True,
              lesion_margin=0.03):
    """Add a synthetic ruler like those in the legacy ISIC images.

    style "ticks": the common faint style (Normal_Ruler folder) -- a row of
        thin dark tick marks directly on the skin near an edge, sometimes at
        an angle, slightly blurred.
    style "strip": the rare prominent style (Thick_Ruler folder) -- the white
        edge of a physical ruler entering the frame, with thick dark mm ticks.
        Only used with placement "edge" -- a straight ruler strip laid across
        a lesion isn't a pattern that occurs in the data, so this is ignored
        (and never sampled) under placement "lesion".
    edge: "bottom" | "top" | "left" | "right" (random if None). Only used
        with placement "edge".
    placement: "edge" (ruler enters from one image border), "lesion"
        (ticks a few mm/cm from one side of the lesion), or "short_ruler"
        (a single short ruler -- 3-4 scale marks, see _draw_ruler_fragment
        -- entering from one edge or corner; bigger/bolder than the
        "ticks" style but smaller than "strip"). Random if None, using
        this module's default ~10/30/40/20 mix (strip-edge / lesion /
        ticks-edge / short_ruler) -- see the module docstring. The
        concept-set build assigns categories itself rather than relying
        on this default, to hit exact proportions.
    contact: only used with placement "lesion" -- "side" (the default)
        keeps ticks short enough to stay within the gap, so no lesion
        pixels are touched; "alongside" uses longer ticks that cross the
        gap and reach across the lesion boundary, mirroring
        add_synthetic_ruler() in ruler_bias_counterfactual_test.py (the
        Layer-3 causal test). Not sampled by default.
    lesion_bbox: (x0, y0, x1, y1) in pixel coords; auto-detected via
        find_lesion_bbox() if None.
    avoid_lesion: in placement "edge"/"short_ruler", try the other edges
        rather than one whose rendered ruler band would overlap the
        lesion bbox. Ignored in placement "lesion" (that mode is
        lesion-relative by design and never overlaps by construction).
    lesion_margin: extra padding (as a fraction of image scale) added
        around the lesion bbox when checking for overlap.

    For the combined condition apply the ruler FIRST, then the vignette
    (add_vignette(add_ruler(img))), so the lens falloff dims the ruler the
    way it does in the real images.
    """
    rng = np.random.default_rng(seed)
    if placement is None:
        roll = rng.random()
        if style is not None:
            placement = "edge"
        elif roll < 0.10:
            placement, style = "edge", "strip"
        elif roll < 0.40:
            placement = "lesion"
        elif roll < 0.80:
            placement, style = "edge", "ticks"
        else:
            placement = "short_ruler"

    im = img.convert("RGB")
    w, h = im.size
    scale = min(w, h)

    if lesion_bbox is None:
        lesion_bbox = find_lesion_bbox(im)

    if placement == "lesion":
        overlay = _draw_lesion_ruler(lesion_bbox, rng, w, h, contact=contact)
        return Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")

    if placement == "short_ruler":
        overlay = _draw_short_ruler(lesion_bbox, rng, w, h, scale,
                                     avoid_lesion=avoid_lesion,
                                     lesion_margin=lesion_margin)
        return Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")

    if style is None:
        style = ["ticks", "strip"][rng.integers(0, 2)]

    edge_order = [edge] if edge is not None else \
        list(rng.permutation(["bottom", "top", "left", "right"]))

    margin = lesion_margin * scale
    fallback = None
    for candidate_edge in edge_order:
        overlay, band = _draw_edge_ruler(candidate_edge, style, rng, scale, w, h)
        if fallback is None:
            fallback = overlay
        if not avoid_lesion or not _bbox_overlap(band, lesion_bbox, margin):
            return Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")

    # every edge's band overlapped the lesion (rare -- very large or
    # off-center lesion) -- fall back to the first candidate rather than
    # silently dropping the ruler from this image
    return Image.alpha_composite(im.convert("RGBA"), fallback).convert("RGB")
