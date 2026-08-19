"""
build_ruler_matched_concepts.py
Builds the corrected ruler_present CAV concept set from the style-matched
clean image pool (Clean_resorted -- vignette confound already sorted out
by hand), using the synthetic ruler overlay in ruler_overlay.py.

Paired design: every clean image gets a ruler-overlaid positive copy in
concept_images_ruler_matched/ruler_present/, and an untouched copy in
concept_images_ruler_matched/clean/ -- same lesion, same framing, only the
ruler pixels differ, so lesion morphology can't leak into the CAV
direction the way it could with an independent-samples split.

Every image is assigned to exactly one of four categories, in fixed
proportions (not left to add_ruler()'s internal random dispatch, so the
mix is exact rather than approximate):
  10% thick_edge  -- style "strip" at one image edge (Thick_Ruler-style,
                      rare in the real data: 9 of 940 ruler images).
  30% lesion_side -- ticks a few mm/cm from one side of the lesion,
                      pointing toward it, never touching it.
  40% ticks_edge  -- style "ticks" at one image edge, elsewhere on the
                      skin (Normal_Ruler-style), no baseline scale line.
  20% short_ruler -- a single short ruler (3-4 scale marks, not a full
                      row) entering from one edge or corner, bigger/
                      bolder than "ticks_edge" but smaller than "strip".
None of the four ever overlaps the lesion bbox.

Category assignment uses the largest-remainder method so the counts sum
exactly to the pool size, then a seeded shuffle so category doesn't
correlate with file order. A manifest CSV records which category each
image got, for later spot-checks or subgroup analysis.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import os
import csv
import shutil
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ruler_overlay import add_ruler

PROJECT_ROOT = os.path.expanduser(
    "~/scratch/dev-uos/projects/VTCAV_Dermatology"
)
CLEAN_DIR = os.path.join(PROJECT_ROOT, "datasets", "ruler_sorted", "Clean_resorted")
OUT_ROOT = os.path.join(PROJECT_ROOT, "concept_images_ruler_matched")
OUT_POS = os.path.join(OUT_ROOT, "ruler_present")
OUT_NEG = os.path.join(OUT_ROOT, "clean")
MANIFEST_PATH = os.path.join(OUT_ROOT, "category_manifest.csv")

BASE_SEED = 42

CATEGORIES = [
    ("thick_edge",  0.10, dict(placement="edge", style="strip")),
    ("lesion_side", 0.30, dict(placement="lesion", contact="side")),
    ("ticks_edge",  0.40, dict(placement="edge", style="ticks")),
    ("short_ruler", 0.20, dict(placement="short_ruler")),
]

os.makedirs(OUT_POS, exist_ok=True)
os.makedirs(OUT_NEG, exist_ok=True)

files = sorted(
    f for f in os.listdir(CLEAN_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
)
n = len(files)
print(f"Found {n} clean images in {CLEAN_DIR}")

# largest-remainder method: exact proportional counts summing to n
raw = [p * n for _, p, _ in CATEGORIES]
counts = [int(x) for x in raw]
remainder = n - sum(counts)
order = sorted(range(len(CATEGORIES)), key=lambda i: -(raw[i] - counts[i]))
for i in range(remainder):
    counts[order[i]] += 1

labels = []
for (name, _, _), c in zip(CATEGORIES, counts):
    labels += [name] * c
np.random.default_rng(BASE_SEED).shuffle(labels)
assert len(labels) == n

kwargs_by_name = {name: kw for name, _, kw in CATEGORIES}
print("Category counts:", {name: c for (name, _, _), c in zip(CATEGORIES, counts)})

n_ok, n_failed = 0, 0
manifest_rows = []
for i, (fname, label) in enumerate(zip(files, labels)):
    src_path = os.path.join(CLEAN_DIR, fname)
    try:
        img = Image.open(src_path)
        img.load()
    except Exception as e:
        print(f"  skip (unreadable): {fname} ({e})")
        n_failed += 1
        continue

    # verify the overlay actually drew something before trusting it -- a
    # geometry edge case (see ISIC_0000057_downsampled.jpg, where the old
    # tick-row anchor let the whole row land off-canvas) could otherwise
    # silently produce a "ruler_present" image with no ruler in it at all.
    # Compared in-memory, pre-JPEG, so recompression noise can't mask this.
    src_rgb = np.array(img.convert("RGB")).astype(int)
    ruler_img = None
    for attempt in range(5):
        candidate = add_ruler(img, seed=BASE_SEED + i + attempt * 10007,
                               **kwargs_by_name[label])
        changed = (np.abs(np.array(candidate).astype(int) - src_rgb).sum(axis=2)
                   > 10).sum()
        if changed > 300:
            ruler_img = candidate
            break
    if ruler_img is None:
        print(f"  WARNING: no visible ruler after 5 attempts, keeping last: {fname}")
        ruler_img = candidate

    ruler_img.save(os.path.join(OUT_POS, fname))
    shutil.copy2(src_path, os.path.join(OUT_NEG, fname))
    manifest_rows.append((fname, label))
    n_ok += 1

with open(MANIFEST_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "category"])
    writer.writerows(manifest_rows)

print(f"\nDone. {n_ok} paired images written, {n_failed} skipped.")
print(f"  positives (ruler): {OUT_POS}")
print(f"  negatives (clean): {OUT_NEG}")
print(f"  manifest: {MANIFEST_PATH}")
