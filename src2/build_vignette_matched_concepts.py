"""
build_vignette_matched_concepts.py
Builds the vignette_present CAV concept set from the same style-matched
clean image pool used for the corrected ruler_present CAV
(Clean_resorted -- rectangular framed, ruler-free, hand-sorted).

Paired design: every clean image gets a vignette-applied positive copy in
concept_images_vignette_matched/vignette_present/positive/, and an
untouched-frame negative copy in .../negative/. Ruler status (present /
absent) is assigned once per image and applied IDENTICALLY to both the
positive and negative branch before the vignette step -- so ruler can't
correlate with vignette in either direction, and any CAV direction this
produces is attributable to vignette alone. This mirrors the pairing
logic of build_ruler_matched_concepts.py so the two CAVs are directly
comparable (e.g. via cosine similarity) later.

Ruler assignment uses the largest-remainder method for an exact 50/50
split (not left to chance), then a seeded shuffle so ruler status doesn't
correlate with file order. When a ruler is assigned, category is drawn
from the same edge-style pool as the ruler-matched build (ticks_edge /
thick_edge / short_ruler only -- "lesion_side" is skipped here since
lesion-adjacent placement isn't needed to balance ruler for this concept,
and keeping ruler off the lesion border removes one extra degree of
variation between the two scripts' outputs).

Vignette circle size (radius_frac) is fixed at 0.58 for every image,
rather than a continuous range.

A manifest CSV records ruler status, category, and vignette params
(radius_frac, feather_frac) per image for later spot-checks.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import os
import csv
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ruler_overlay import add_ruler
from vignette_overlay import add_vignette

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(PROJECT_ROOT, "datasets", "ruler_sorted", "Clean_resorted")
OUT_ROOT = os.path.join(PROJECT_ROOT, "concept_images_vignette_matched")
OUT_POS = os.path.join(OUT_ROOT, "vignette_present", "positive")
OUT_NEG = os.path.join(OUT_ROOT, "vignette_present", "negative")
MANIFEST_PATH = os.path.join(OUT_ROOT, "category_manifest.csv")

BASE_SEED = 43  # deliberately different from ruler build's BASE_SEED=42
                # so the ruler-subset assignment here isn't a mirror image
                # of the ruler build's own shuffle order

RULER_FRAC = 0.5

RULER_CATEGORIES = [
    ("thick_edge", 0.20, dict(placement="edge", style="strip")),
    ("ticks_edge", 0.55, dict(placement="edge", style="ticks")),
    ("short_ruler", 0.25, dict(placement="short_ruler")),
]

VIGNETTE_RADIUS_FRAC = 0.58

os.makedirs(OUT_POS, exist_ok=True)
os.makedirs(OUT_NEG, exist_ok=True)

files = sorted(
    f for f in os.listdir(CLEAN_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)
n = len(files)
print(f"Found {n} clean images in {CLEAN_DIR}")

# --- exact 50/50 ruler split (largest-remainder) ---
n_ruler = round(n * RULER_FRAC)
n_noruler = n - n_ruler
ruler_flags = [True] * n_ruler + [False] * n_noruler
np.random.default_rng(BASE_SEED).shuffle(ruler_flags)
assert len(ruler_flags) == n

# --- exact category split among the ruler subset (largest-remainder) ---
n_r = sum(ruler_flags)
raw = [p * n_r for _, p, _ in RULER_CATEGORIES]
counts = [int(x) for x in raw]
remainder = n_r - sum(counts)
order = sorted(range(len(RULER_CATEGORIES)), key=lambda i: -(raw[i] - counts[i]))
for i in range(remainder):
    counts[order[i]] += 1

cat_labels = []
for (name, _, _), c in zip(RULER_CATEGORIES, counts):
    cat_labels += [name] * c
np.random.default_rng(BASE_SEED + 1).shuffle(cat_labels)
assert len(cat_labels) == n_r
kwargs_by_name = {name: kw for name, _, kw in RULER_CATEGORIES}

cat_iter = iter(cat_labels)
print("Ruler split:", {"ruler": n_r, "no_ruler": n_noruler})
print("Ruler category counts:", {name: c for (name, _, _), c in zip(RULER_CATEGORIES, counts)})

n_ok, n_failed = 0, 0
manifest_rows = []
for i, (fname, has_ruler) in enumerate(zip(files, ruler_flags)):
    src_path = os.path.join(CLEAN_DIR, fname)
    try:
        img = Image.open(src_path)
        img.load()
    except Exception as e:
        print(f"  skip (unreadable): {fname} ({e})")
        n_failed += 1
        continue

    img_seed = BASE_SEED + i

    # --- ruler stage: identical result gets used for BOTH branches ---
    category = None
    if has_ruler:
        category = next(cat_iter)
        src_rgb = np.array(img.convert("RGB")).astype(int)
        base = None
        for attempt in range(5):
            candidate = add_ruler(img, seed=img_seed + attempt * 10007,
                                   **kwargs_by_name[category])
            changed = (np.abs(np.array(candidate).astype(int) - src_rgb).sum(axis=2)
                       > 10).sum()
            if changed > 300:
                base = candidate
                break
        if base is None:
            print(f"  WARNING: no visible ruler after 5 attempts, keeping last: {fname}")
            base = candidate
    else:
        base = img.convert("RGB")

    # --- negative branch: base as-is, rectangular frame, ruler status preserved ---
    base.save(os.path.join(OUT_NEG, fname))

    # --- positive branch: vignette applied on top of the SAME base ---
    radius_frac = VIGNETTE_RADIUS_FRAC
    feather_frac = np.random.default_rng(img_seed + 600000).uniform(0.04, 0.12)
    vign = add_vignette(base, radius_frac=radius_frac, feather_frac=feather_frac,
                         seed=img_seed)

    # sanity check: vignette should visibly darken the corners vs. base
    base_arr = np.array(base).astype(int)
    vign_arr = np.array(vign).astype(int)
    corner_changed = np.abs(vign_arr[:20, :20] - base_arr[:20, :20]).sum() > 500
    if not corner_changed:
        print(f"  WARNING: vignette barely visible in corner check: {fname}")

    vign.save(os.path.join(OUT_POS, fname))

    manifest_rows.append((fname, has_ruler, category or "", round(radius_frac, 4),
                           round(feather_frac, 4)))
    n_ok += 1

with open(MANIFEST_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "has_ruler", "ruler_category", "radius_frac", "feather_frac"])
    writer.writerows(manifest_rows)

print(f"\nDone. {n_ok} paired images written, {n_failed} skipped.")
print(f"  positives (vignette): {OUT_POS}")
print(f"  negatives (rectangular): {OUT_NEG}")
print(f"  manifest: {MANIFEST_PATH}")