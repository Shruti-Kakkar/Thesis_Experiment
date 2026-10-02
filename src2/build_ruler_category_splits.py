"""
build_ruler_category_splits.py

PURPOSE: builds the four per-style ruler_present concept sets used by
VisualTCAV_derma_global_ruler_bias_by_category.py (thick_edge,
lesion_side, ticks_edge, short_ruler). No new images are generated: each
category directory holds symlinks into the pooled set that
build_ruler_matched_concepts.py builds, split by its
category_manifest.csv. A positive image and its paired negative share a
filename, so both land in the same category.

Output layout:
    concept_images_ruler_matched_<category>/ruler_present/{positive,negative}/

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import csv
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOLED_DIR = os.path.join(PROJECT_ROOT, "concept_images_ruler_matched")
MANIFEST = os.path.join(POOLED_DIR, "category_manifest.csv")
CATEGORIES = ["thick_edge", "lesion_side", "ticks_edge", "short_ruler"]
SPLITS = ["positive", "negative"]

with open(MANIFEST) as f:
    manifest = {row["filename"]: row["category"] for row in csv.DictReader(f)}

for category in CATEGORIES:
    files = sorted(f for f, c in manifest.items() if c == category)
    created = existing = 0
    for split in SPLITS:
        src_dir = os.path.join(POOLED_DIR, "ruler_present", split)
        dst_dir = os.path.join(PROJECT_ROOT, f"concept_images_ruler_matched_{category}", "ruler_present", split)
        os.makedirs(dst_dir, exist_ok=True)
        for filename in files:
            src = os.path.join(src_dir, filename)
            dst = os.path.join(dst_dir, filename)
            if not os.path.exists(src):
                raise FileNotFoundError(src)
            if os.path.lexists(dst):
                if os.path.realpath(dst) != os.path.realpath(src):
                    raise RuntimeError(f"{dst} exists but does not point to {src}")
                existing += 1
                continue
            os.symlink(os.path.relpath(src, dst_dir), dst)
            created += 1
    print(f"{category:<12} {len(files):>4} images per split -- {created} links created, {existing} already present")
