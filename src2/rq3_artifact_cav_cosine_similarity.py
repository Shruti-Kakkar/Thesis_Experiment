"""
rq3_artifact_cav_cosine_similarity.py

PURPOSE: the "Cosine similarity" column of the RQ3 CAV validation-accuracy
table (tab:rq3_cav_accuracy, Section 5.5.1): cosine similarity, and the
corresponding angle, between the pooled ruler_present and vignette_present
CAV directions at each of the four swept layers (Equation 4.x,
eq:cosine_similarity). Reads the CAVs already cached by
VisualTCAV_derma_global_ruler_bias_matched.py and
VisualTCAV_derma_global_vignette_matched.py -- no retraining, no model
compute.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import json
import os
import numpy as np
from joblib import load

PROJECT_ROOT = os.path.expanduser(
    "~/scratch/dev-uos/projects/VTCAV_Dermatology"
)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)  # joblib needs VisualTCAV's classes to unpickle

LAYERS = ["conv5_block1_out", "conv5_block2_out", "conv5_block3_out", "post_relu"]
MAX_EXAMPLES = 200
N_CAV_RUNS = 20

CONCEPTS = {
    "ruler_present":    "padonly_seed2_rulerbias_matched",
    "vignette_present": "padonly_seed2_vignette_matched",
}

OUT_PATH = os.path.join(PROJECT_ROOT, "outputs2", "rq3_artifact_cav_cosine_similarity.json")


def load_direction(concept, model_tag, layer_name):
    path = os.path.join(
        PROJECT_ROOT, "outputs2", f"vtcav_{model_tag}", "cache", "resnet50v2",
        f"cav_{concept}_positive_{MAX_EXAMPLES}_neg_{N_CAV_RUNS}runs_{layer_name}.joblib",
    )
    direction = load(path).cav.direction
    return direction.numpy() if hasattr(direction, 'numpy') else np.asarray(direction)


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


results = {}
print(f"{'Layer':<20} {'cosine':>8} {'angle':>8}")
for layer_name in LAYERS:
    ruler = load_direction("ruler_present", CONCEPTS["ruler_present"], layer_name)
    vignette = load_direction("vignette_present", CONCEPTS["vignette_present"], layer_name)
    sim = cosine_sim(ruler, vignette)
    angle = float(np.degrees(np.arccos(np.clip(sim, -1.0, 1.0))))
    results[layer_name] = {"cosine_similarity": sim, "angle_deg": angle}
    print(f"{layer_name:<20} {sim:>8.3f} {angle:>7.1f}°")

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUT_PATH}")
