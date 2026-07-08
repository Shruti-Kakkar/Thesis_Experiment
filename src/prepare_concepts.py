"""
prepare_concepts.py
Prepares concept images from Derm7pt dataset for Visual-TCAV.

Concepts follow Lucieri et al. (2020) where applicable:
- pigment_network split into: typical, atypical (2 CAVs)
- streaks split into: regular, irregular (2 CAVs)
- dots_and_globules split into: regular, irregular (2 CAVs)
- regression_structures: present vs absent (1 CAV)
- blue_whitish_veil: present vs absent (1 CAV)
- pigmentation: all non-absent vs absent (1 CAV) [beyond Lucieri]
- vascular_structures: all non-absent vs absent (1 CAV) [beyond Lucieri]

Total: 9 concept CAVs + 1 random baseline

Splitting regular/irregular follows Lucieri et al. Table I and their
finding that regular and irregular variants have OPPOSITE TCAV directions
(e.g. regular streaks push toward NV, irregular streaks push toward MEL).
Merging them into one concept would cancel out this signal entirely.

References:
- Visual-TCAV (De Santis et al., 2024) Section 3.1, 4.1
- TCAV (Kim et al., 2018)
- Lucieri et al. (2020) Table I — concept split definitions
- Kawahara et al. (2019) — Derm7pt dataset
- Argenziano et al. (1998) — 7-point checklist clinical ground truth
Author: Shruti Kakkar
"""

import os
import shutil
import random
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 1. PATHS
# ─────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DERM7PT_DIR   = os.path.join(BASE_DIR, "datasets", "release_v0")
META_CSV      = os.path.join(DERM7PT_DIR, "meta", "meta.csv")
TRAIN_IDX_CSV = os.path.join(DERM7PT_DIR, "meta", "train_indexes.csv")

# images/ root — derm column already contains subfolder e.g. "NEL/Nel026.jpg"
IMAGES_DIR    = os.path.join(DERM7PT_DIR, "images")

ISIC_TRAIN_DIR = os.path.join(BASE_DIR, "datasets", "ISIC_2019_Training_Input",
                               "ISIC_2019_Training_Input")

# All outputs stay inside VTCAV_Dermatology
CONCEPT_OUT   = os.path.join(BASE_DIR, "concept_images")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# 2. CONCEPT DEFINITIONS
#
# KEY DESIGN DECISION — split vs merge:
# Lucieri et al. (2020) Table I treated regular and irregular variants
# as SEPARATE concepts, not merged. Their Fig. 4 shows why this matters:
# ST_R (regular streaks) → positive for NV (benign)
# ST_IR (irregular streaks) → positive for MEL (malignant)
# These are opposite directions. Merging cancels the signal.
# We follow the same split for: pigment_network, streaks, dots_and_globules.
#
# pigmentation and vascular_structures are NOT in Lucieri et al.
# They are our additions using remaining Derm7pt annotations.
# These are noted explicitly as extensions beyond Lucieri in the thesis.
#
# Concepts with small positive counts (noted below) produce less stable
# CAVs. Results for these are interpreted with caution in the thesis.
# ─────────────────────────────────────────────
CONCEPTS = {

    # ── Pigment Network (split, following Lucieri Table I) ──────────────
    # typical PN → benign, indicates NV
    # atypical PN → malignant clue, indicates MEL (major criterion)
    "pigment_network_typical": {
        "column":   "pigment_network",
        "positive": ["typical"],
        "negative": ["absent"],
        # expected: HIGH for NV, LOW for MEL
    },
    "pigment_network_atypical": {
        "column":   "pigment_network",
        "positive": ["atypical"],
        "negative": ["absent"],
        # expected: HIGH for MEL, LOW for NV
    },

    # ── Streaks (split, following Lucieri Table I) ───────────────────────
    # regular streaks → benign, symmetric distribution
    # irregular streaks → malignant clue for MEL (minor criterion)
    # NOTE: streaks_regular will have ~40 positives — interpret with caution
    "streaks_regular": {
        "column":   "streaks",
        "positive": ["regular"],
        "negative": ["absent"],
        # expected: HIGH for NV — but small sample, treat as exploratory
    },
    "streaks_irregular": {
        "column":   "streaks",
        "positive": ["irregular"],
        "negative": ["absent"],
        # expected: HIGH for MEL
    },

    # ── Pigmentation (not in Lucieri — our extension) ────────────────────
    # irregular pigmentation is a minor criterion for MEL
    # Derm7pt does not distinguish regular/irregular pigmentation cleanly
    # so we use all non-absent as positive (presence-focused)
    "pigmentation": {
        "column":   "pigmentation",
        "positive": ["diffuse irregular", "localized irregular",
                     "diffuse regular",   "localized regular"],
        "negative": ["absent"],
        # expected: moderately positive for MEL (irregular subtypes)
    },

    # ── Regression Structures (single concept, following Lucieri) ────────
    # presence highly indicative of melanoma (minor criterion)
    # NOTE: 96 positives — moderate reliability
    "regression_structures": {
        "column":   "regression_structures",
        "positive": ["blue areas", "white areas", "combinations"],
        "negative": ["absent"],
        # expected: HIGH for MEL
    },

    # ── Dots and Globules (split, following Lucieri Table I) ─────────────
    # regular D&G → benign, symmetric centre distribution
    # irregular D&G → malignant clue for MEL (minor criterion)
    "dots_and_globules_regular": {
        "column":   "dots_and_globules",
        "positive": ["regular"],
        "negative": ["absent"],
        # expected: HIGH for NV
    },
    "dots_and_globules_irregular": {
        "column":   "dots_and_globules",
        "positive": ["irregular"],
        "negative": ["absent"],
        # expected: HIGH for MEL
    },

    # ── Blue-Whitish Veil (single concept, following Lucieri) ────────────
    # strongest single indicator of melanoma (major criterion)
    # NOTE: 74 positives — moderate reliability
    "blue_whitish_veil": {
        "column":   "blue_whitish_veil",
        "positive": ["present"],
        "negative": ["absent"],
        # expected: HIGH for MEL (strongest signal)
    },

    # ── Vascular Structures (not in Lucieri — our extension) ─────────────
    # arborizing vessels → BCC indicator
    # dotted/hairpin vessels → AK/SCC indicator
    # NOTE: only 66 positives — least reliable CAV, interpret with caution
    "vascular_structures": {
        "column":   "vascular_structures",
        "positive": ["arborizing", "comma", "dotted", "hairpin",
                     "linear irregular", "within regression", "wreath"],
        "negative": ["absent"],
        # expected: positive for BCC, AK, SCC, VASC — exploratory for MEL/NV
    },
}

# ─────────────────────────────────────────────
# 3. HELPERS
# ─────────────────────────────────────────────
def resolve_path(images_dir, fp):
    """
    Try the exact path first. If not found, try case-insensitive
    subfolder match. Handles Derm7pt metadata capitalisation issues
    (e.g. meta.csv says FCl/ but disk has FCL/).
    """
    exact = os.path.join(images_dir, fp)
    if os.path.exists(exact):
        return exact

    parts = fp.replace("\\", "/").split("/")
    if len(parts) != 2:
        return exact

    subfolder, filename = parts
    try:
        entries = os.listdir(images_dir)
    except OSError:
        return exact

    for entry in entries:
        if entry.lower() == subfolder.lower():
            candidate = os.path.join(images_dir, entry, filename)
            if os.path.exists(candidate):
                return candidate

    return exact


def copy_images(filepaths, dest_dir, concept_name, split):
    """
    Copy images to destination directory.
    Returns (copied, skipped).
    Prints every missing filepath so you can investigate if skipped > 0.
    """
    os.makedirs(dest_dir, exist_ok=True)
    copied        = 0
    skipped       = 0
    missing_paths = []

    for fp in filepaths:
        src = resolve_path(IMAGES_DIR, fp)
        if not os.path.exists(src):
            skipped += 1
            missing_paths.append(src)
            continue
        dst = os.path.join(dest_dir, os.path.basename(fp))
        shutil.copy2(src, dst)
        copied += 1

    status = "✓" if skipped == 0 else "!"
    print(f"  {status} {concept_name} [{split}]: {copied} copied, {skipped} not found")

    if missing_paths:
        print(f"    Missing files:")
        for mp in missing_paths:
            print(f"      {mp}")

    return copied, skipped

# ─────────────────────────────────────────────
# 4. LOAD AND FILTER METADATA
#
# Filter to Derm7pt train split only (train_indexes.csv).
# Prevents test images leaking into concept training set.
# ─────────────────────────────────────────────
print("Loading Derm7pt metadata...")
df_all = pd.read_csv(META_CSV)
print(f"  Total cases in meta.csv: {len(df_all)}")

train_idx = pd.read_csv(TRAIN_IDX_CSV)["indexes"].tolist()
df = df_all.iloc[train_idx].reset_index(drop=True)
print(f"  Cases after train-split filter: {len(df)}")
print(f"  Columns: {list(df.columns)}\n")

# ─────────────────────────────────────────────
# 5. PREPARE CONCEPT IMAGES
#
# Output structure:
#   concept_images/
#     <concept_name>/
#       positive/    <- images where concept is present
#       negative/    <- images where concept is absent
#     random/        <- 500 ISIC 2019 in-domain baseline images
# ─────────────────────────────────────────────
print(f"Output directory: {CONCEPT_OUT}\n")
os.makedirs(CONCEPT_OUT, exist_ok=True)

total_copied  = 0
total_skipped = 0

for concept_name, config in CONCEPTS.items():
    col        = config["column"]
    pos_labels = config["positive"]
    neg_labels = config["negative"]

    pos_df = df[df[col].isin(pos_labels)]
    neg_df = df[df[col].isin(neg_labels)]

    print(f"Concept: {concept_name}")
    print(f"  Rows matched — positive: {len(pos_df)}, negative: {len(neg_df)}")

    # Warn if positive count is small
    if len(pos_df) < 80:
        print(f"  ⚠ Small positive set ({len(pos_df)} images) — CAV may be less stable")

    pos_paths = pos_df["derm"].dropna().tolist()
    neg_paths = neg_df["derm"].dropna().tolist()

    pos_dir = os.path.join(CONCEPT_OUT, concept_name, "positive")
    neg_dir = os.path.join(CONCEPT_OUT, concept_name, "negative")

    c1, s1 = copy_images(pos_paths, pos_dir, concept_name, "positive")
    c2, s2 = copy_images(neg_paths, neg_dir, concept_name, "negative")

    total_copied  += c1 + c2
    total_skipped += s1 + s2
    print()

# ─────────────────────────────────────────────
# 6. PREPARE RANDOM SET (500 ISIC 2019 images)
#
# In-domain negative baseline for statistical significance testing.
# Using ISIC 2019 dermoscopy images (same domain as test images)
# ensures CAV learns the concept, not domain shift.
# See: Visual-TCAV Section 4.1, Lucieri et al. (2020) Section IV-B.
# ─────────────────────────────────────────────
print("Preparing random set (500 ISIC 2019 images)...")

all_isic_images = [
    f for f in os.listdir(ISIC_TRAIN_DIR)
    if f.lower().endswith(".jpg")
]
print(f"  Total ISIC 2019 training images found: {len(all_isic_images)}")

random_sample = random.sample(all_isic_images, 500)
random_dir    = os.path.join(CONCEPT_OUT, "random")
os.makedirs(random_dir, exist_ok=True)

rand_copied = 0
for fname in random_sample:
    src = os.path.join(ISIC_TRAIN_DIR, fname)
    dst = os.path.join(random_dir, fname)
    shutil.copy2(src, dst)
    rand_copied += 1

print(f"  ✓ Random set: {rand_copied} images copied\n")

# ─────────────────────────────────────────────
# 7. SUMMARY
# ─────────────────────────────────────────────
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'concept':<35}  {'pos':>5}  {'neg':>5}  note")
print("-" * 60)

for concept_name in CONCEPTS:
    pos_dir = os.path.join(CONCEPT_OUT, concept_name, "positive")
    neg_dir = os.path.join(CONCEPT_OUT, concept_name, "negative")
    pos_count = len(os.listdir(pos_dir)) if os.path.exists(pos_dir) else 0
    neg_count = len(os.listdir(neg_dir)) if os.path.exists(neg_dir) else 0
    note = "⚠ small" if pos_count < 80 else ""
    print(f"  {concept_name:<33}  {pos_count:>5}  {neg_count:>5}  {note}")

print("-" * 60)
rand_count = len(os.listdir(random_dir)) if os.path.exists(random_dir) else 0
print(f"  {'random':<33}  {rand_count:>5} images")
print("=" * 60)
print(f"\nTotal files copied : {total_copied}")
print(f"Total files skipped: {total_skipped}", end="")
print(" ✓" if total_skipped == 0 else " ← investigate missing paths above")
print("\nConcept preparation complete!")