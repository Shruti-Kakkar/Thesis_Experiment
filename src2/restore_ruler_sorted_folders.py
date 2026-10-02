"""
restore_ruler_sorted_folders.py

PURPOSE: rebuilds the two hand-labelled image folders from the label
files in labels/ and the ISIC 2019 images:
  - datasets/ruler_sorted/<folder>/  (training images sorted by hand into
    Normal_Ruler, Thick_Ruler, Clean, Doubtful; Clean_resorted is the
    ruler-free, rectangular-framed subset of Clean). Source pool for the
    ruler_present and vignette_present concept sets.
  - datasets/ruler_sorted_test/<CLASS>_test_<ruler|clean>/  (the 200 MEL
    and 200 NV test images listed by extract_global_vtcav_test_images.py,
    labelled by hand for a physical ruler). Used for the ground-truth
    check and the counterfactual tests.

Expects datasets/ISIC_2019_Training_Input/ISIC_2019_Training_Input/ and
datasets/test_images_by_class/ (from src/organise_test_images.py).

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import csv
import os
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
LABELS_DIR = os.path.join(PROJECT_ROOT, "labels")
TRAIN_IMAGES_DIR = os.path.join(DATASETS_DIR, "ISIC_2019_Training_Input", "ISIC_2019_Training_Input")
TEST_IMAGES_DIR = os.path.join(DATASETS_DIR, "test_images_by_class")


def restore(rows, src_of, dst_of, label):
    copied = existing = 0
    for row in rows:
        src, dst = src_of(row), dst_of(row)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            existing += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
    print(f"{label}: {len(rows)} images -- {copied} copied, {existing} already present")


with open(os.path.join(LABELS_DIR, "ruler_sorted_training.csv")) as f:
    restore(
        list(csv.DictReader(f)),
        lambda r: os.path.join(TRAIN_IMAGES_DIR, r["filename"]),
        lambda r: os.path.join(DATASETS_DIR, "ruler_sorted", r["folder"], r["filename"]),
        "ruler_sorted",
    )

with open(os.path.join(LABELS_DIR, "ruler_sorted_test.csv")) as f:
    restore(
        list(csv.DictReader(f)),
        lambda r: os.path.join(TEST_IMAGES_DIR, r["class"], r["filename"]),
        lambda r: os.path.join(DATASETS_DIR, "ruler_sorted_test", f"{r['class']}_test_{r['ruler']}", r["filename"]),
        "ruler_sorted_test",
    )
