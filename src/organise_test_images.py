"""
organise_test_images.py
Organises ISIC 2019 test images into per-class subfolders
using the ground truth CSV, ready for GlobalVisualTCAV.
Author: Shruti Kakkar
"""

import os
import sys
import pandas as pd

# ─────────────────────────────────────────────
# PATHS — hardcoded absolute paths for reliability
# ─────────────────────────────────────────────
PROJECT_ROOT  = os.path.expanduser(
    "~/scratch/dev-uos/projects/VTCAV_Dermatology"
)
TEST_INPUT    = os.path.join(PROJECT_ROOT, "datasets",
                             "ISIC_2019_Test_Input",
                             "ISIC_2019_Test_Input")
GT_CSV        = os.path.join(PROJECT_ROOT, "datasets",
                             "ISIC_2019_Test_GroundTruth.csv")
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "datasets",
                             "test_images_by_class")

# ─────────────────────────────────────────────
# VALIDATE PATHS BEFORE DOING ANYTHING
# ─────────────────────────────────────────────
print(f"PROJECT_ROOT : {PROJECT_ROOT}")
print(f"TEST_INPUT   : {TEST_INPUT}")
print(f"GT_CSV       : {GT_CSV}")
print(f"OUTPUT_DIR   : {OUTPUT_DIR}")
print()

for path, label in [(TEST_INPUT, "TEST_INPUT"), (GT_CSV, "GT_CSV")]:
    if not os.path.exists(path):
        print(f"ERROR: {label} not found: {path}")
        sys.exit(1)
    print(f"  OK: {label}")

# ─────────────────────────────────────────────
# CLASSES — UNK excluded
# ─────────────────────────────────────────────
CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# ─────────────────────────────────────────────
# LOAD GROUND TRUTH
# ─────────────────────────────────────────────
print("\nLoading ground truth CSV...")
df = pd.read_csv(GT_CSV)
print(f"  Total rows: {len(df)}")

# ─────────────────────────────────────────────
# CREATE CLASS FOLDERS AND SYMLINK IMAGES
# ─────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nCreating per-class folders in: {OUTPUT_DIR}")

counts      = {cls: 0 for cls in CLASSES}
skipped_unk = 0
skipped_missing = 0

for _, row in df.iterrows():

    image_id = row["image"]   # e.g. "ISIC_0034321"

    # Find the class with value 1.0
    assigned_class = None
    for cls in CLASSES:
        if cls in row and float(row[cls]) == 1.0:
            assigned_class = cls
            break

    # Skip UNK
    if assigned_class is None:
        skipped_unk += 1
        continue

    # Source image
    src = os.path.join(TEST_INPUT, image_id + ".jpg")
    if not os.path.exists(src):
        skipped_missing += 1
        continue

    # Destination
    dest_dir = os.path.join(OUTPUT_DIR, assigned_class)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, image_id + ".jpg")

    # Symlink if not already there
    if not os.path.exists(dest):
        os.symlink(src, dest)

    counts[assigned_class] += 1

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print(f"\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")
total = 0
for cls in CLASSES:
    print(f"  {cls:<6}  {counts[cls]:>5} images")
    total += counts[cls]
print(f"{'─'*50}")
print(f"  {'Total':<6}  {total:>5} images organised")
print(f"  Skipped UNK     : {skipped_unk}")
print(f"  Skipped missing : {skipped_missing}")
print(f"{'='*50}")
print(f"\nOutput: {OUTPUT_DIR}")
print("Done. Ready for run_vtcav.py.")