"""
VisualTCAV_derma_global_ruler_bias_matched.py
Same as VisualTCAV_derma_global_ruler_bias.py, but trains the CORRECTED
'ruler_present' CAV: positive = synthetic ruler overlay on style-matched
clean images (concept_images_ruler_matched/ruler_present/positive,
built by build_ruler_matched_concepts.py), negative = the SAME images
unmodified (.../negative) -- a paired design, so lesion morphology and
photographic style can't leak into the CAV direction the way vignette
style did in the original (positive=Normal_Ruler, negative=Clean, two
different real-photo folders with different dermoscope framing).

Test images (what GlobalVisualTCAV explains) are UNCHANGED -- still the
full MEL/NV test sets from TEST_IMAGES_DIR -- only the CAV's own
training source changed. That's what makes this comparable to the
original confounded run: same targets explained, different concept.

Uses a fresh MODEL_TAG ("padonly_seed2_rulerbias_matched") so the CAV
cache can never collide with the original run's cache -- the cache key
in VisualTCAV._compute_cavs() is keyed by concept_name + n_runs + layer
+ max_examples, NOT by concept_images_dir, so reusing the original
MODEL_TAG here would silently reload the confounded CAV instead of
training a new one.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 0. MODEL VARIANT TAG — isolated from the confounded run's cache
# ─────────────────────────────────────────────
MODEL_TAG = "padonly_seed2_rulerbias_matched"

# ─────────────────────────────────────────────
# 1. PATHS
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser(
    "~/scratch/dev-uos/projects/VTCAV_Dermatology"
)

SOURCE_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models2", "resnet50v2_isic2019_final_padonly_seed2.keras"
)

VTCAV_DIR       = os.path.join(PROJECT_ROOT, "outputs2", f"vtcav_{MODEL_TAG}")
MODELS_DIR      = os.path.join(VTCAV_DIR, "models")
CACHE_DIR       = os.path.join(VTCAV_DIR, "cache")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "datasets", "test_images_by_class")  # shared, unchanged
RESULTS_DIR     = os.path.join(PROJECT_ROOT, "outputs2", f"vtcav_results_{MODEL_TAG}")

# Corrected concept set -- built by build_ruler_matched_concepts.py, real
# files already laid out as ruler_present/{positive,negative}, no symlink
# farm needed here (unlike the original confounded script).
CONCEPT_DIR = os.path.join(PROJECT_ROOT, "concept_images_ruler_matched")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

positive_dir = os.path.join(CONCEPT_DIR, "ruler_present", "positive")
negative_dir = os.path.join(CONCEPT_DIR, "ruler_present", "negative")
if not (os.path.isdir(positive_dir) and os.path.isdir(negative_dir)):
    raise FileNotFoundError(
        f"Expected {positive_dir} and {negative_dir} -- run "
        f"build_ruler_matched_concepts.py first."
    )
n_pos = len(os.listdir(positive_dir))
n_neg = len(os.listdir(negative_dir))
print(f"Corrected concept set: {n_pos} positive, {n_neg} negative "
      f"(paired, from {CONCEPT_DIR})")

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

from VisualTCAV import GlobalVisualTCAV, Model
from tensorflow.keras.applications.resnet_v2 import (
    preprocess_input as preprocess_resnet_v2
)
print("VisualTCAV imported successfully.\n")

# ─────────────────────────────────────────────
# 4. CONCEPT — same name as the confounded run ("ruler_present/positive"),
# only concept_images_dir + MODEL_TAG differ, so downstream code that
# indexes by this string (e.g. local_tcav.computations[layer][name])
# doesn't need to change.
# ─────────────────────────────────────────────
CONCEPTS = ["ruler_present/positive"]

# ─────────────────────────────────────────────
# 5. TARGET CLASSES — same MEL/NV comparison as the confounded run
# ─────────────────────────────────────────────
CLASSES = ["MEL", "NV"]

# ─────────────────────────────────────────────
# 6. LAYERS — same 4 layers as the confounded run
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
    )

    global_visual_tcav.setLayers(layer_names=LAYERS)
    global_visual_tcav.setConcepts(concept_names=CONCEPTS)

    global_visual_tcav.explain(
        cache_cav=True,
        cache_random=False,
    )

    global_visual_tcav.statsInfo()

    plot_path = os.path.join(RESULTS_DIR, f"vtcav_global_ruler_matched_{target_class}.png")
    global_visual_tcav.plot()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"Plot saved: {plot_path}")

print(f"\n{'='*60}")
print(f"All classes complete. [{MODEL_TAG}]")
print(f"Results: {RESULTS_DIR}")
print(f"Cache  : {CACHE_DIR}")
print(f"{'='*60}")
