# SUPERSEDED: not used for any result in the submitted thesis.
# Replaced by: nothing -- diagnostic heatmap for the short_ruler CAV, not a reported result.
# Kept as a record of how the work developed.

"""
ruler_short_ruler_local_check.py
Diagnostic check (not a formal reported result) on the short_ruler
per-category CAV's suspiciously high-but-unstable attribution (see
[[vtcav_ruler_per_category_decomposition]]): scores a small representative
sample of real MEL test images with the short_ruler-ONLY CAV
(padonly_seed2_rulerbias_matched_short_ruler, already cached from the
by-category training -- no retraining needed) and generates a full
LocalVisualTCAV heatmap for the highest-scoring image, to see WHERE the
attribution concentrates: on tick-mark-like real content (hair, fine dark
lines, an actual ruler if present) would support the hypothesis that the
CAV's poorly-constrained direction ended up pointing toward a generic
"tick-like texture" signal rather than a genuinely short-ruler-specific
one; landing somewhere unrelated (smooth skin, lesion body) would suggest
it's just noise.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
for _gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)

MODEL_TAG = "padonly_seed2_rulerbias_matched_short_ruler"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models2", "resnet50v2_isic2019_final_padonly_seed2.keras"
)
VTCAV_DIR   = os.path.join(PROJECT_ROOT, "outputs2", f"vtcav_{MODEL_TAG}")
MODELS_DIR  = os.path.join(VTCAV_DIR, "models")
CACHE_DIR   = os.path.join(VTCAV_DIR, "cache")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "datasets", "test_images_by_class")
CONCEPT_DIR = os.path.join(PROJECT_ROOT, "concept_images_ruler_matched_short_ruler")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs2", "ruler_short_ruler_local_check")

os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_SUBDIR    = os.path.join(MODELS_DIR, "resnet50v2")
GRAPH_FILENAME  = "resnet50v2_isic2019_final.keras"
LABELS_FILENAME = "isic2019_classes.txt"
os.makedirs(MODEL_SUBDIR, exist_ok=True)
graph_dest = os.path.join(MODEL_SUBDIR, GRAPH_FILENAME)
if not os.path.exists(graph_dest):
    os.symlink(SOURCE_MODEL_PATH, graph_dest)
CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
labels_dest = os.path.join(MODEL_SUBDIR, LABELS_FILENAME)
if not os.path.exists(labels_dest):
    with open(labels_dest, 'w') as f:
        for cls in CLASS_NAMES:
            f.write(cls + "\n")

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from VisualTCAV import LocalVisualTCAV, Model
from tensorflow.keras.applications.resnet_v2 import (
    preprocess_input as preprocess_resnet_v2
)

CONCEPTS = ["ruler_present/positive"]
LAYERS = ["post_relu"]
N_IMAGES = 10
SCREENING_SAMPLE_SEED = 5


def pick_test_images(class_name, n):
    class_dir = os.path.join(TEST_IMAGES_DIR, class_name)
    files = sorted(f for f in os.listdir(class_dir) if f.lower().endswith('.jpg'))
    class_offset = sum(ord(c) for c in class_name)
    rng = np.random.default_rng(SCREENING_SAMPLE_SEED + class_offset)
    idx = rng.choice(len(files), size=min(n, len(files)), replace=False)
    return [files[i] for i in sorted(idx)]


def make_local_tcav(true_class, rel_path):
    local_tcav = LocalVisualTCAV(
        test_image_filename=rel_path,
        m_steps=50,
        target_class=true_class,
        model=Model(
            model_name="resnet50v2",
            graph_path_filename=GRAPH_FILENAME,
            label_path_filename=LABELS_FILENAME,
            preprocessing_function=preprocess_resnet_v2,
            max_examples=200,
            resize_mode='pad',
        ),
        models_dir=MODELS_DIR,
        cache_dir=CACHE_DIR,
        test_images_dir=TEST_IMAGES_DIR,
        concept_images_dir=CONCEPT_DIR,
        negative_suffix="negative",
    )
    local_tcav.setLayers(layer_names=LAYERS)
    local_tcav.setConcepts(concept_names=CONCEPTS)
    return local_tcav


targets = [("MEL", f) for f in pick_test_images("MEL", N_IMAGES)]
print(f"Scoring {len(targets)} real MEL images with the short_ruler-only CAV\n")

scored = []
for true_class, fname in targets:
    rel_path = os.path.join(true_class, fname)
    local_tcav = make_local_tcav(true_class, rel_path)
    local_tcav.predict()
    local_tcav.explain(cache_cav=True, cache_random=True, n_cav_runs=20)
    attribution = float(
        local_tcav.computations["post_relu"]["ruler_present/positive"].attributions[0]
    )
    scored.append((fname, attribution))
    print(f"  {fname}: attribution={attribution:.5f}")

scored.sort(key=lambda x: x[1], reverse=True)
top_fname, top_attr = scored[0]
print(f"\nTop scorer: {top_fname} (attribution={top_attr:.5f}) -- generating full heatmap")

rel_path = os.path.join("MEL", top_fname)
local_tcav = make_local_tcav("MEL", rel_path)
local_tcav.predict()
local_tcav.explain(cache_cav=True, cache_random=True, n_cav_runs=20)

image_label = os.path.splitext(top_fname)[0]

def _save_instead_of_show(image_label=image_label, top_attr=top_attr):
    fname_out = os.path.join(RESULTS_DIR, f"top_short_ruler_{image_label}_MEL.png")
    fig = plt.gcf()
    fig.text(
        0.5, 1.0, f"{image_label}  |  true class: MEL  |  "
        f"short_ruler-only CAV attribution={top_attr:.5f}",
        ha='center', va='top', fontsize=12, color='black', fontweight='bold',
        bbox=dict(facecolor='orange', alpha=0.9, edgecolor='black', pad=4),
    )
    plt.savefig(fname_out, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"Saved: {fname_out}")
plt.show = _save_instead_of_show

local_tcav.plot()

print(f"\nAll done. Results in: {RESULTS_DIR}")
