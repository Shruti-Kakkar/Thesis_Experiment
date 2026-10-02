"""
artifact_pool_coverage.py

PURPOSE: the coverage figures in Limitations ("Source coverage of the
artifact concept images"): the hand-reviewed pool that the ruler_present
and vignette_present base images are drawn from (datasets/ruler_sorted/),
its ISIC identifier range, and its share of the ISIC 2019 training set.
Clean_resorted is a re-sort of images already in Clean, so the pool is
the union of all sorted folders. No model compute.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import json
import os
import re
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SORTED_DIR = os.path.join(PROJECT_ROOT, "datasets", "ruler_sorted")
GROUND_TRUTH = os.path.join(PROJECT_ROOT, "datasets", "ISIC_2019_Training_GroundTruth.csv")
OUT_PATH = os.path.join(PROJECT_ROOT, "outputs2", "artifact_pool_coverage.json")

folders = sorted(os.listdir(SORTED_DIR))
pool = set()
for folder in folders:
    files = os.listdir(os.path.join(SORTED_DIR, folder))
    pool |= set(files)
    print(f"  {folder:<16} {len(files):>5} images")

ids = [int(re.search(r"ISIC_(\d+)", f).group(1)) for f in pool]
n_training = len(pd.read_csv(GROUND_TRUTH))
result = {
    "reviewed_pool": len(pool),
    "isic_id_range": [min(ids), max(ids)],
    "training_set": n_training,
    "pct_of_training_set": 100 * len(pool) / n_training,
}
print(f"Reviewed pool: {len(pool)} unique images, ISIC IDs {min(ids)}..{max(ids)}, "
      f"{len(pool)}/{n_training} = {result['pct_of_training_set']:.1f}% of the training set")

with open(OUT_PATH, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved: {OUT_PATH}")
