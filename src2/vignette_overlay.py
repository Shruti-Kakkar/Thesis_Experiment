"""
vignette_overlay.py
Synthetic circular vignette (dark corners) overlay for building the
vignette_present CAV concept set -- see concept_images_vignette_matched/.

For the combined condition, apply the ruler FIRST, then the vignette --
add_vignette(add_ruler(img)) -- so the lens falloff dims the ruler the
way it does in real images.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def add_vignette(img, radius_frac=None, feather_frac=None, center_jitter=0.04,
                  dark_level=4, seed=None):
    """
    Applies a synthetic circular vignette (dark corners) to img.
    seed must be passed explicitly for reproducibility -- default_rng(None)
    would otherwise pull fresh OS entropy per call, making a full build
    non-reproducible.
    """
    rng = np.random.default_rng(seed)

    if radius_frac is None:
        radius_frac = rng.uniform(0.46, 0.58)
    if feather_frac is None:
        feather_frac = rng.uniform(0.04, 0.12)

    im = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = im.shape[:2]
    r = radius_frac * min(h, w)
    cx = w / 2 + rng.uniform(-center_jitter, center_jitter) * w
    cy = h / 2 + rng.uniform(-center_jitter, center_jitter) * h

    mask_im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask_im).ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    mask_im = mask_im.filter(ImageFilter.GaussianBlur(radius=feather_frac * r))
    mask = np.asarray(mask_im, np.float32)[..., None] / 255.0

    out = im * mask + dark_level * (1.0 - mask)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def add_generic_border(img, radius_frac=0.58, feather_frac=0.08, dark_level=4):
    """
    Negative control for the vignette counterfactual test: darkens a
    plain RECTANGULAR border (uniform inset from all four edges) instead
    of a circular vignette, with the border width solved so the total
    darkened AREA exactly matches add_vignette(radius_frac=radius_frac)'s
    darkened area (everything outside the circle) for this image's actual
    size. Same feather_frac/dark_level as the real vignette -- only the
    SHAPE differs (circular vs rectangular), isolating whether the
    counterfactual effect is vignette-shape-specific or just generic
    edge-darkening.

    Deterministic -- no seed/jitter, matching the fixed-parameter
    convention used for the real vignette counterfactual test.
    """
    im = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = im.shape[:2]

    r = radius_frac * min(h, w)
    circle_area = np.pi * r * r
    target_dark_area = max(0.0, w * h - circle_area)

    # Solve for uniform inset b: w*h - (w-2b)(h-2b) = target_dark_area
    # => 4b^2 - 2b(w+h) + target_dark_area = 0
    a_coef, b_coef, c_coef = 4.0, -2.0 * (w + h), target_dark_area
    disc = max(0.0, b_coef * b_coef - 4 * a_coef * c_coef)
    b = (-b_coef - np.sqrt(disc)) / (2 * a_coef)  # smaller root
    b = float(np.clip(b, 0, min(w, h) / 2 - 1))

    mask_im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask_im).rectangle([b, b, w - 1 - b, h - 1 - b], fill=255)
    mask_im = mask_im.filter(ImageFilter.GaussianBlur(radius=feather_frac * r))
    mask = np.asarray(mask_im, np.float32)[..., None] / 255.0

    out = im * mask + dark_level * (1.0 - mask)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
