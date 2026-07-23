"""
VisualTCAV_derma_global_padonly_seed2.py
Runs GlobalVisualTCAV on the padding-only (seed 2) ResNet50V2 model for all
8 ISIC 2019 classes using 10 Derm7pt dermoscopic concepts.

ADAPTED FROM VisualTCAV_derma_global.py (Run 1 / Run 2, naive-resize model) TO:
  - Point at the padding-only seed2 model (models2/resnet50v2_isic2019_final_padonly_seed2.keras)
  - Use a COMPLETELY SEPARATE models_dir / cache_dir / results_dir

WHY THE SEPARATE CACHE DIR IS NOT OPTIONAL:
Cache filenames in this pipeline (e.g. "test_fmaps_MEL_post_relu.joblib",
"neg_acts_<concept>_200_post_relu.joblib") encode class/concept/layer but NOT
which model produced them. If this script pointed at the old cache_dir
(outputs/vtcav/cache), it would silently load Run 1/Run 2's cached activations
-- computed on the naive-resize model -- instead of recomputing for this model,
with no error or warning. MODEL_TAG below drives every path that could be
affected by this, so this can't happen.

Follows the modified VisualTCAV.py which uses:
  - Concept-specific negative images (Derm7pt "absent" labelled)
    instead of random ISIC images for CAV training
  - 20 CAV runs per concept per layer (different random splits)
  - Mean attribution across 20 runs reported

Author: Shruti Kakkar
References:
  - De Santis et al. (2024) Visual-TCAV
  - Lucieri et al. (2020) TCAV on dermoscopy (20 CAV runs)
  - Kim et al. (2018) TCAV
"""

import sys
sys.dont_write_bytecode = True

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 0. MODEL VARIANT TAG — drives every isolated path below
# ─────────────────────────────────────────────
MODEL_TAG = "padonly_seed2"

# ─────────────────────────────────────────────
# 1. PATHS
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser(
    "~/scratch/dev-uos/projects/VTCAV_Dermatology"
)

# Source model: from models2/ (the padding-only training line), NOT models/
SOURCE_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models2", "resnet50v2_isic2019_final_padonly_seed2.keras"
)

# Everything below is namespaced by MODEL_TAG so nothing can collide with
# Run 1 / Run 2 (naive-resize model) outputs, or with any future variant
VTCAV_DIR       = os.path.join(PROJECT_ROOT, "outputs2", f"vtcav_{MODEL_TAG}")
MODELS_DIR      = os.path.join(VTCAV_DIR, "models")
CACHE_DIR       = os.path.join(VTCAV_DIR, "cache")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "datasets", "test_images_by_class")  # shared, read-only
CONCEPT_DIR     = os.path.join(PROJECT_ROOT, "concept_images")                     # shared, read-only
RESULTS_DIR     = os.path.join(PROJECT_ROOT, "outputs2", f"vtcav_results_{MODEL_TAG}")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 2. MODEL FOLDER STRUCTURE
# KerasModelWrapper expects:
#   models_dir/resnet50v2/resnet50v2_isic2019_final.keras
#   models_dir/resnet50v2/isic2019_classes.txt
# The internal filename stays the same (GRAPH_FILENAME) -- only the
# symlink TARGET changes, and it lives inside this run's own MODELS_DIR,
# so it can't be confused with Run 1/2's symlink of the same name.
# ─────────────────────────────────────────────
MODEL_SUBDIR    = os.path.join(MODELS_DIR, "resnet50v2")
GRAPH_FILENAME  = "resnet50v2_isic2019_final.keras"
LABELS_FILENAME = "isic2019_classes.txt"

os.makedirs(MODEL_SUBDIR, exist_ok=True)

graph_dest = os.path.join(MODEL_SUBDIR, GRAPH_FILENAME)
if not os.path.exists(graph_dest):
    os.symlink(SOURCE_MODEL_PATH, graph_dest)
    print(f"Symlinked model to: {graph_dest}")
    print(f"  -> source: {SOURCE_MODEL_PATH}")
else:
    # Sanity check: confirm the existing symlink actually points where we expect,
    # in case this MODEL_TAG folder was reused from a previous partial run.
    actual_target = os.path.realpath(graph_dest)
    expected_target = os.path.realpath(SOURCE_MODEL_PATH)
    if actual_target != expected_target:
        raise RuntimeError(
            f"Existing symlink at {graph_dest} points to {actual_target}, "
            f"but expected {expected_target}. Refusing to proceed -- delete "
            f"the stale symlink/cache manually if you intend to change models "
            f"under this MODEL_TAG."
        )
    print(f"Symlink already correct: {graph_dest} -> {actual_target}")

# Write labels file — alphabetical order matching training (unchanged --
# architecture and class order are identical to Run 1/2, only preprocessing differs)
CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
labels_dest = os.path.join(MODEL_SUBDIR, LABELS_FILENAME)
if not os.path.exists(labels_dest):
    with open(labels_dest, 'w') as f:
        for cls in CLASSES:
            f.write(cls + "\n")
    print(f"Labels file written: {labels_dest}")

# ─────────────────────────────────────────────
# 3. IMPORTS
# VisualTCAV.py is the shared engine -- imported from src/ (not duplicated
# into src2/) so both model variants use the identical, unmodified engine code.
# Only these wrapper scripts differ.
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
# 4. CONCEPTS
# ─────────────────────────────────────────────
CONCEPTS = [
    "pigment_network_typical/positive",
    "pigment_network_atypical/positive",
    "streaks_regular/positive",
    "streaks_irregular/positive",
    "pigmentation/positive",
    "regression_structures/positive",
    "dots_and_globules_regular/positive",
    "dots_and_globules_irregular/positive",
    "blue_whitish_veil/positive",
    "vascular_structures/positive",
]

# ─────────────────────────────────────────────
# 5. LAYERS
# ─────────────────────────────────────────────
LAYERS = [
    "conv5_block1_out",
    "conv5_block2_out",
    "conv5_block3_out",
    "post_relu",
]

# ─────────────────────────────────────────────
# 6. RUN GLOBAL VISUAL-TCAV PER CLASS
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
            # CRITICAL: this model was trained with tf.image.resize_with_pad
            # (aspect-preserving), not naive stretch. VisualTCAV.py's image
            # loading must match, or every concept/test image gets fed to
            # this model distorted in a way it never saw during training.
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

    plot_path = os.path.join(RESULTS_DIR, f"vtcav_global_{target_class}.png")
    global_visual_tcav.plot()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"Plot saved: {plot_path}")

print(f"\n{'='*60}")
print(f"All classes complete. [{MODEL_TAG}]")
print(f"Results: {RESULTS_DIR}")
print(f"Cache  : {CACHE_DIR}")
print(f"{'='*60}")