"""
vignette_bias_common_matched.py
Same role as ruler_bias_common_matched.py, but points at the
'vignette_present' CAV -- concept_images_vignette_matched, and its own
MODEL_TAG so the CAV cache can't collide with any other run's
(VisualTCAV._compute_cavs()'s cache key is concept_name + n_runs +
layer + max_examples, NOT concept_images_dir -- reusing another
MODEL_TAG here would silently reload the wrong CAV).

Test images (TEST_IMAGES_DIR, N_PER_CLASS=25 each of MEL/NV) are a
SEEDED RANDOM sample per class, not the first N files alphabetically
(that was ruler_bias_common_matched.py's approach, and it turned out to
be a real bug there too -- filenames are clustered by acquisition batch,
so "first 25 alphabetically" drew 0/25 square (real-vignette-style)
images against a true population rate of ~85% (MEL) / ~59% (NV) square,
completely missing the acquisition style where the vignette effect is
actually large). A random sample is representative in expectation
instead. Imported by vignette_bias_local_screening_matched.py and
vignette_bias_finalize_matched.py, mirroring the ruler-matched pair.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import os

import numpy as np

# Shared GPU on this machine -- claim memory incrementally instead of
# grabbing it all upfront, so we can coexist with other processes'
# allocations instead of failing outright when headroom is tight.
import tensorflow as tf
for _gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)

MODEL_TAG = "padonly_seed2_vignette_matched"

PROJECT_ROOT = os.path.expanduser(
    "~/scratch/dev-uos/projects/VTCAV_Dermatology"
)
SOURCE_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models2", "resnet50v2_isic2019_final_padonly_seed2.keras"
)
VTCAV_DIR   = os.path.join(PROJECT_ROOT, "outputs2", f"vtcav_{MODEL_TAG}")
MODELS_DIR  = os.path.join(VTCAV_DIR, "models")
CACHE_DIR   = os.path.join(VTCAV_DIR, "cache")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "datasets", "test_images_by_class")
CONCEPT_DIR = os.path.join(PROJECT_ROOT, "concept_images_vignette_matched")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs2", "vignette_bias_local_screening_matched")

os.makedirs(RESULTS_DIR, exist_ok=True)

positive_dir = os.path.join(CONCEPT_DIR, "vignette_present", "positive")
negative_dir = os.path.join(CONCEPT_DIR, "vignette_present", "negative")
if not (os.path.isdir(positive_dir) and os.path.isdir(negative_dir)):
    raise FileNotFoundError(
        f"Expected {positive_dir} and {negative_dir} -- run "
        f"build_vignette_matched_concepts.py first."
    )

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

CONCEPTS = ["vignette_present/positive"]
LAYERS = ["post_relu"]   # only -- keeps this fast, matches ruler_bias_common_matched.py
N_PER_CLASS = 25
N_TOP_TO_PLOT = 3         # generate full heatmaps only for the top N MEL outliers

SCREENING_SAMPLE_SEED = 7  # seeded so the random sample is reproducible

RESULTS_CSV = os.path.join(RESULTS_DIR, "attributions.csv")


def pick_test_images(class_name, n):
    class_dir = os.path.join(TEST_IMAGES_DIR, class_name)
    files = sorted(f for f in os.listdir(class_dir) if f.lower().endswith('.jpg'))
    # offset derived from characters, not Python's hash() -- string hashing
    # is randomized per-process (PYTHONHASHSEED) unless disabled, which
    # would make this "seeded" sample silently non-reproducible across runs
    class_offset = sum(ord(c) for c in class_name)
    rng = np.random.default_rng(SCREENING_SAMPLE_SEED + class_offset)
    idx = rng.choice(len(files), size=min(n, len(files)), replace=False)
    return [files[i] for i in sorted(idx)]


def build_targets():
    targets = []
    for cls in ["MEL", "NV"]:
        for fname in pick_test_images(cls, N_PER_CLASS):
            targets.append((cls, fname))
    return targets


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
