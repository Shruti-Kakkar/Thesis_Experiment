"""
ruler_bias_common_matched.py
Same as ruler_bias_common.py, but points at the CORRECTED 'ruler_present'
CAV -- concept_images_ruler_matched instead of concept_images_ruler, and
a fresh MODEL_TAG so the CAV cache can't collide with the confounded
run's (VisualTCAV._compute_cavs()'s cache key is concept_name + n_runs +
layer + max_examples, NOT concept_images_dir -- reusing the original
MODEL_TAG here would silently reload the confounded CAV).

Test images explained (TEST_IMAGES_DIR, N_PER_CLASS=25 each of MEL/NV)
are UNCHANGED from ruler_bias_common.py -- only the CAV's training
source differs. Imported by ruler_bias_local_screening_matched.py and
ruler_bias_finalize_matched.py, mirroring the original pair.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import os

# Shared GPU on this machine -- claim memory incrementally instead of
# grabbing it all upfront, so we can coexist with other processes'
# allocations instead of failing outright when headroom is tight.
import tensorflow as tf
for _gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)

MODEL_TAG = "padonly_seed2_rulerbias_matched"

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
CONCEPT_DIR = os.path.join(PROJECT_ROOT, "concept_images_ruler_matched")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs2", "ruler_bias_local_screening_matched")

os.makedirs(RESULTS_DIR, exist_ok=True)

positive_dir = os.path.join(CONCEPT_DIR, "ruler_present", "positive")
negative_dir = os.path.join(CONCEPT_DIR, "ruler_present", "negative")
if not (os.path.isdir(positive_dir) and os.path.isdir(negative_dir)):
    raise FileNotFoundError(
        f"Expected {positive_dir} and {negative_dir} -- run "
        f"build_ruler_matched_concepts.py first."
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

CONCEPTS = ["ruler_present/positive"]
LAYERS = ["post_relu"]   # only -- keeps this fast, matches ruler_bias_common.py
N_PER_CLASS = 25
N_TOP_TO_PLOT = 3         # generate full heatmaps only for the top N MEL outliers

RESULTS_CSV = os.path.join(RESULTS_DIR, "attributions.csv")


def pick_test_images(class_name, n):
    class_dir = os.path.join(TEST_IMAGES_DIR, class_name)
    files = sorted(f for f in os.listdir(class_dir) if f.lower().endswith('.jpg'))
    return files[:n]


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
