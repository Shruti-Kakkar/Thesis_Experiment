"""
VisualTCAV_derma_global.py
Runs GlobalVisualTCAV on the trained ResNet50V2 model for all
8 ISIC 2019 classes using 10 Derm7pt dermoscopic concepts.

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
# 1. PATHS — all inside VTCAV_Dermatology/
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser(
    "~/scratch/dev-uos/projects/VTCAV_Dermatology"
)

VTCAV_DIR       = os.path.join(PROJECT_ROOT, "outputs", "vtcav")
MODELS_DIR      = os.path.join(VTCAV_DIR, "models")
CACHE_DIR       = os.path.join(VTCAV_DIR, "cache")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "datasets", "test_images_by_class")
CONCEPT_DIR     = os.path.join(PROJECT_ROOT, "concept_images")
RESULTS_DIR     = os.path.join(PROJECT_ROOT, "outputs", "vtcav_results")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 2. MODEL FOLDER STRUCTURE
# KerasModelWrapper expects:
#   models_dir/resnet50v2/resnet50v2_isic2019_final.keras
#   models_dir/resnet50v2/isic2019_classes.txt
# ─────────────────────────────────────────────
MODEL_SUBDIR    = os.path.join(MODELS_DIR, "resnet50v2")
GRAPH_FILENAME  = "resnet50v2_isic2019_final.keras"
LABELS_FILENAME = "isic2019_classes.txt"

os.makedirs(MODEL_SUBDIR, exist_ok=True)

# Symlink model into expected location
graph_dest = os.path.join(MODEL_SUBDIR, GRAPH_FILENAME)
if not os.path.exists(graph_dest):
    os.symlink(
        os.path.join(PROJECT_ROOT, "models", GRAPH_FILENAME),
        graph_dest
    )
    print(f"Symlinked model to: {graph_dest}")

# Write labels file — alphabetical order matching training
CLASSES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
labels_dest = os.path.join(MODEL_SUBDIR, LABELS_FILENAME)
if not os.path.exists(labels_dest):
    with open(labels_dest, 'w') as f:
        for cls in CLASSES:
            f.write(cls + "\n")
    print(f"Labels file written: {labels_dest}")

# ─────────────────────────────────────────────
# 3. IMPORTS
# ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from VisualTCAV import GlobalVisualTCAV, Model
from tensorflow.keras.applications.resnet_v2 import (
    preprocess_input as preprocess_resnet_v2
)
print("VisualTCAV imported successfully.\n")

# ─────────────────────────────────────────────
# 4. CONCEPTS
# Format: "<concept_root>/positive"
# VisualTCAV resolves positive images from:
#   concept_images/<concept_root>/positive/
# Negative images loaded from:
#   concept_images/<concept_root>/negative/
# (Derm7pt "absent" labelled images — clinically verified)
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
# ResNet50V2 conv5 blocks + post_relu
# All have spatial resolution 7x7x2048
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
    print(f"GlobalVisualTCAV — class: {target_class} ({n_test} images)")
    print(f"{'='*60}")

    global_visual_tcav = GlobalVisualTCAV(

        # Target class
        test_images_folder=target_class,
        target_class=target_class,

        # Integrated gradients steps (50 as in notebook)
        m_steps=50,

        # Batch size
        batch_size=20,

        # Number of CAV runs per concept per layer
        # Following Lucieri et al. (2020) who used 20 runs
        n_cav_runs=20,

        # Model
        model=Model(
            model_name="resnet50v2",
            graph_path_filename=GRAPH_FILENAME,
            label_path_filename=LABELS_FILENAME,
            preprocessing_function=preprocess_resnet_v2,
            max_examples=200,
        ),

        # Paths — all inside VTCAV_Dermatology
        models_dir=MODELS_DIR,
        cache_dir=CACHE_DIR,
        test_images_dir=TEST_IMAGES_DIR,
        concept_images_dir=CONCEPT_DIR,

        # negative_suffix: tells VisualTCAV where to find
        # concept-specific negative images for each concept
        # Loads from: concept_images/<concept_root>/negative/
        negative_suffix="negative",
    )

    # Set layers
    global_visual_tcav.setLayers(layer_names=LAYERS)

    # Set concepts
    global_visual_tcav.setConcepts(concept_names=CONCEPTS)

    # Run — caching enabled
    global_visual_tcav.explain(
        cache_cav=True,
        cache_random=False,  # not used anymore — negatives are concept-specific
    )

    # Print stats table
    global_visual_tcav.statsInfo()

    # Save plot
    plot_path = os.path.join(RESULTS_DIR, f"vtcav_global_{target_class}.png")
    global_visual_tcav.plot()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"Plot saved: {plot_path}")

print(f"\n{'='*60}")
print("All classes complete.")
print(f"Results: {RESULTS_DIR}")
print(f"Cache  : {CACHE_DIR}")
print(f"{'='*60}")
