"""
concept_set_sizes.py

PURPOSE: the concept-set size table (tab:concept_stats, Section 5.1):
positive and nominal (absent-only) negative image counts per derm7pt
concept, the hard-negative-augmented negative pool used for CAV training
for the six regularity-paired concepts (absent images + the sibling
variant's positives, as in VisualTCAV_derma_global_padonly_seed2.py), and
the random baseline set size. Counts the folders prepare_concepts.py
builds -- no model compute.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONCEPT_DIR = os.path.join(PROJECT_ROOT, "concept_images")
OUT_PATH = os.path.join(PROJECT_ROOT, "outputs2", "concept_set_sizes.json")

CONCEPTS = [
    "pigment_network_typical", "pigment_network_atypical",
    "streaks_regular", "streaks_irregular",
    "pigmentation", "regression_structures",
    "dots_and_globules_regular", "dots_and_globules_irregular",
    "blue_whitish_veil", "vascular_structures",
]
SIBLING = {
    "pigment_network_typical":     "pigment_network_atypical",
    "pigment_network_atypical":    "pigment_network_typical",
    "streaks_regular":             "streaks_irregular",
    "streaks_irregular":           "streaks_regular",
    "dots_and_globules_regular":   "dots_and_globules_irregular",
    "dots_and_globules_irregular": "dots_and_globules_regular",
}


def count(*parts):
    return len(os.listdir(os.path.join(CONCEPT_DIR, *parts)))


sizes = {}
print(f"{'Concept':<30} {'Positive':>8} {'Negative':>8} {'Augmented':>9}")
for concept in CONCEPTS:
    positive = count(concept, "positive")
    negative = count(concept, "negative")
    augmented = negative + count(SIBLING[concept], "positive") if concept in SIBLING else None
    sizes[concept] = {"positive": positive, "negative": negative, "augmented_negative": augmented}
    print(f"{concept:<30} {positive:>8} {negative:>8} {augmented if augmented else '':>9}")
sizes["random"] = count("random")
print(f"{'random (ISIC 2019)':<30} {sizes['random']:>8}")

with open(OUT_PATH, "w") as f:
    json.dump(sizes, f, indent=2)
print(f"\nSaved: {OUT_PATH}")
