"""
VisualTCAV_derma_global_ruler_bias_by_category.py
Same as VisualTCAV_derma_global_ruler_bias_matched.py, but trains a
SEPARATE 'ruler_present' CAV per ruler-style category (thick_edge,
lesion_side, ticks_edge, short_ruler) instead of the pooled mix of all
four -- a sensitivity/decomposition check on whether the pooled CAV's
lower val_acc (0.68-0.72, vs vignette's 0.94-0.99) is driven by
heterogeneity across styles, and whether any one style dominates the
pooled attribution result.

Concept images are symlinks into the already-built
concept_images_ruler_matched/ruler_present/{positive,negative}/, split
by category per concept_images_ruler_matched/category_manifest.csv --
no new synthetic images generated, this only re-partitions what
build_ruler_matched_concepts.py already built.

Each category gets its own MODEL_TAG (padonly_seed2_rulerbias_matched_
<category>) so CAV caches never collide with the pooled run's or each
other's -- see [[feedback_cav_cache_invalidation]].

Usage:
    python VisualTCAV_derma_global_ruler_bias_by_category.py --category thick_edge
    python VisualTCAV_derma_global_ruler_bias_by_category.py --category lesion_side
    python VisualTCAV_derma_global_ruler_bias_by_category.py --category ticks_edge
    python VisualTCAV_derma_global_ruler_bias_by_category.py --category short_ruler

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import argparse
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--category', required=True,
                     choices=['thick_edge', 'lesion_side', 'ticks_edge', 'short_ruler'])
parser.add_argument('--cav-seed', type=int, default=42,
                     help="base seed for the 20 CAV train/val splits (run i uses "
                          "cav_seed+i) -- change to check whether a val_acc result "
                          "is a stable property of the concept or a fluke of one "
                          "particular set of splits. The CAV cache key does NOT "
                          "include the seed, so cache and results directories are "
                          "suffixed with _cavseed<N> to keep seeds from overwriting "
                          "each other.")
args = parser.parse_args()
CATEGORY = args.category
CAV_SEED = args.cav_seed

# ─────────────────────────────────────────────
# 0. MODEL VARIANT TAG — isolated per category, and from the pooled run's cache
# ─────────────────────────────────────────────
MODEL_TAG = f"padonly_seed2_rulerbias_matched_{CATEGORY}"

# ─────────────────────────────────────────────
# 1. PATHS
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser(
    "~/scratch/dev-uos/projects/VTCAV_Dermatology"
)

SOURCE_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models2", "resnet50v2_isic2019_final_padonly_seed2.keras"
)

# Per-seed suffix: the CAV cache key excludes the seed, so a shared cache dir
# would silently reuse another seed's CAVs (and a shared results dir would
# overwrite its outputs).
SEED_TAG = f"cavseed{CAV_SEED}"
VTCAV_DIR       = os.path.join(PROJECT_ROOT, "outputs2", f"vtcav_{MODEL_TAG}_{SEED_TAG}")
MODELS_DIR      = os.path.join(VTCAV_DIR, "models")
CACHE_DIR       = os.path.join(VTCAV_DIR, "cache")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "datasets", "test_images_by_class")  # shared, unchanged
RESULTS_DIR     = os.path.join(PROJECT_ROOT, "outputs2", f"vtcav_results_{MODEL_TAG}_{SEED_TAG}")

CONCEPT_DIR = os.path.join(PROJECT_ROOT, f"concept_images_ruler_matched_{CATEGORY}")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

positive_dir = os.path.join(CONCEPT_DIR, "ruler_present", "positive")
negative_dir = os.path.join(CONCEPT_DIR, "ruler_present", "negative")
if not (os.path.isdir(positive_dir) and os.path.isdir(negative_dir)):
    raise FileNotFoundError(
        f"Expected {positive_dir} and {negative_dir} -- category symlink "
        f"split must be built first."
    )
n_pos = len(os.listdir(positive_dir))
n_neg = len(os.listdir(negative_dir))
print(f"Category '{CATEGORY}' concept set: {n_pos} positive, {n_neg} negative "
      f"(from {CONCEPT_DIR})")

# ─────────────────────────────────────────────
# 2. MODEL FOLDER STRUCTURE
# ─────────────────────────────────────────────
MODEL_SUBDIR    = os.path.join(MODELS_DIR, "resnet50v2")
GRAPH_FILENAME  = "resnet50v2_isic2019_final.keras"
LABELS_FILENAME = "isic2019_classes.txt"

os.makedirs(MODEL_SUBDIR, exist_ok=True)

graph_dest = os.path.join(MODEL_SUBDIR, GRAPH_FILENAME)
if not os.path.exists(graph_dest):
    os.symlink(SOURCE_MODEL_PATH, graph_dest)
    print(f"Symlinked model to: {graph_dest}")
else:
    print(f"Model symlink already present: {graph_dest}")

CLASSES_ALL = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
labels_dest = os.path.join(MODEL_SUBDIR, LABELS_FILENAME)
if not os.path.exists(labels_dest):
    with open(labels_dest, 'w') as f:
        for cls in CLASSES_ALL:
            f.write(cls + "\n")
    print(f"Labels file written: {labels_dest}")

# ─────────────────────────────────────────────
# 3. IMPORTS
# ─────────────────────────────────────────────
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
from joblib import load

from VisualTCAV import GlobalVisualTCAV, Model
from tensorflow.keras.applications.resnet_v2 import (
    preprocess_input as preprocess_resnet_v2
)
print("VisualTCAV imported successfully.\n")

# ─────────────────────────────────────────────
# 4. CONCEPT — same name as the pooled run ("ruler_present/positive")
# ─────────────────────────────────────────────
CONCEPTS = ["ruler_present/positive"]

# ─────────────────────────────────────────────
# 5. TARGET CLASSES — same MEL/NV comparison as the pooled run
# ─────────────────────────────────────────────
CLASSES = ["MEL", "NV"]

# ─────────────────────────────────────────────
# 6. LAYERS — same 4 layers as the pooled run
# ─────────────────────────────────────────────
LAYERS = [
    "conv5_block1_out",
    "conv5_block2_out",
    "conv5_block3_out",
    "post_relu",
]

# ─────────────────────────────────────────────
# 7. RUN GLOBAL VISUAL-TCAV PER CLASS
# ─────────────────────────────────────────────
results = {
    "category": CATEGORY,
    "cav_seed": CAV_SEED,
    "n_positive": n_pos,
    "n_negative": n_neg,
    "cav_val_acc": {},
    "attribution": {},
}

for target_class in CLASSES:

    test_class_dir = os.path.join(TEST_IMAGES_DIR, target_class)
    if not os.path.exists(test_class_dir):
        print(f"[SKIP] No test folder for: {target_class}")
        continue

    n_test = len([
        f for f in os.listdir(test_class_dir)
        if f.lower().endswith('.jpg')
    ])
    if n_test == 0:
        print(f"[SKIP] No images in: {test_class_dir}")
        continue

    print(f"\n{'='*60}")
    print(f"GlobalVisualTCAV [{MODEL_TAG}] — class: {target_class} ({n_test} images)")
    print(f"{'='*60}")

    global_visual_tcav = GlobalVisualTCAV(
        test_images_folder=target_class,
        target_class=target_class,
        m_steps=50,
        batch_size=20,
        n_cav_runs=20,
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
        cav_seed=CAV_SEED,
    )

    global_visual_tcav.setLayers(layer_names=LAYERS)
    global_visual_tcav.setConcepts(concept_names=CONCEPTS)

    global_visual_tcav.explain(
        cache_cav=True,
        cache_random=False,
    )

    global_visual_tcav.statsInfo()

    results["attribution"][target_class] = {}
    for concept_name in CONCEPTS:
        for layer_name in LAYERS:
            stat = global_visual_tcav.stats[layer_name][concept_name]
            results["attribution"][target_class][layer_name] = {
                "mean": float(stat.mean),
                "std": float(stat.std),
                "ci_begin": float(stat.begin),
                "ci_end": float(stat.end),
                "n_images": int(stat.n),
            }

    plot_path = os.path.join(RESULTS_DIR, f"vtcav_global_ruler_{CATEGORY}_{target_class}.png")
    global_visual_tcav.plot()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"Plot saved: {plot_path}")

# CAV val_acc per layer, read back from the cached CAVs (not kept on the
# GlobalVisualTCAV object after explain())
concept_root = CONCEPTS[0].split('/')[0]
for layer_name in LAYERS:
    cav_path = os.path.join(
        CACHE_DIR, "resnet50v2",
        f"cav_{concept_root}_positive_200_neg_20runs_{layer_name}.joblib",
    )
    val_accs = np.asarray(load(cav_path).cav.val_accs)
    results["cav_val_acc"][layer_name] = {
        "mean": float(val_accs.mean()),
        "std": float(val_accs.std()),
    }

results_path = os.path.join(RESULTS_DIR, "results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results JSON saved: {results_path}")

print(f"\n{'='*60}")
print(f"All classes complete. [{MODEL_TAG}, {SEED_TAG}]")
print(f"Results: {RESULTS_DIR}")
print(f"Cache  : {CACHE_DIR}")
print(f"{'='*60}")
