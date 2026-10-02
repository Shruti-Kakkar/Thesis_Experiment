"""
rq4_square_image_rates.py

PURPOSE: the square-image rates in the RQ4 sampling-procedure paragraph
(Section 5.6.1): the share of naturally square (unpadded) images in the
full MEL/NV test pools, and in the biased (files[:n]) vs. corrected
(random) vignette local-screening samples. Image size only -- no model
compute.

Inputs:
  - datasets/test_images_by_class/{MEL,NV}/
  - outputs2/vignette_bias_local_screening_matched_biased_sample_archive/attributions.csv
  - outputs2/vignette_bias_local_screening_matched/attributions.csv

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import json
import os
import pandas as pd
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "datasets", "test_images_by_class")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs2")
CLASSES = ["MEL", "NV"]
SAMPLES = {
    "biased (files[:n])": "vignette_bias_local_screening_matched_biased_sample_archive",
    "corrected (random)": "vignette_bias_local_screening_matched",
}
OUT_PATH = os.path.join(OUTPUTS_DIR, "rq4_square_image_rates.json")


def is_square(class_name, filename):
    width, height = Image.open(os.path.join(TEST_IMAGES_DIR, class_name, filename)).size
    return width == height


results = {"full_pool": {}, "screening_samples": {}}
for cls in CLASSES:
    files = [f for f in os.listdir(os.path.join(TEST_IMAGES_DIR, cls)) if f.lower().endswith(".jpg")]
    n_square = sum(is_square(cls, f) for f in files)
    results["full_pool"][cls] = {"square": n_square, "total": len(files), "pct": 100 * n_square / len(files)}
    print(f"Full {cls} test pool: {n_square}/{len(files)} square ({100 * n_square / len(files):.1f}%)")

for label, results_dir in SAMPLES.items():
    df = pd.read_csv(os.path.join(OUTPUTS_DIR, results_dir, "attributions.csv"))
    results["screening_samples"][label] = {}
    for cls in CLASSES:
        files = df.loc[df["class"] == cls, "filename"]
        n_square = sum(is_square(cls, f) for f in files)
        results["screening_samples"][label][cls] = {"square": n_square, "total": len(files)}
        print(f"Vignette screening sample, {label}, {cls}: {n_square}/{len(files)} square")

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUT_PATH}")
