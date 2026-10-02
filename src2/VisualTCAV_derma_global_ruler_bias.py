"""
VisualTCAV_derma_global_ruler_bias.py
Layer 2 of the bias-detection experiment: trains a 'ruler_present' CAV
(positive = Normal_Ruler, negative = Clean) and runs GlobalVisualTCAV to
check whether the model's MEL/NV predictions are disproportionately
explained by ruler presence -- the artifact-bias question motivated by
Winkler et al. (2019, JAMA Dermatology) and grounded in your own Layer 1
finding (MEL 53.6% ruler prevalence vs NV 42.4%, ground truth).

SCOPE (deliberate, see conversation notes):
  - Single concept: ruler_present (positive=Normal_Ruler, negative=Clean;
    Thick_Ruler and Doubtful intentionally excluded from CAV training)
  - Target classes: MEL and NV only -- this is your primary, hypothesis-
    grounded comparison, matching both the literature precedent (Winkler
    et al. used only nevi + melanoma) and your own Layer 1 data (the only
    two classes with a solid ground-truth ruler-prevalence baseline).
    Add other classes to CLASSES below as optional negative controls if
    you want them later -- one-line change, no other edits needed.
  - All 4 layers kept (not just post_relu) -- unlike the clinical
    concepts, a ruler is a low-level geometric/textural feature (straight
    high-contrast lines), so it's a genuinely open question whether
    earlier conv5 layers pick it up more strongly than the clinical
    concepts did. Worth seeing where the signal actually lives, not
    assuming post_relu the way we did before.

Uses a fresh, isolated MODEL_TAG ("padonly_seed2_rulerbias") so this
never collides with your existing hard-negative clinical-concept cache
or results.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 0. MODEL VARIANT TAG — isolated from all other experiments
# ─────────────────────────────────────────────
MODEL_TAG = "padonly_seed2_rulerbias"

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
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "datasets", "test_images_by_class")  # shared
RESULTS_DIR     = os.path.join(PROJECT_ROOT, "outputs2", f"vtcav_results_{MODEL_TAG}")

# Ruler source folders (already on voxel, per your scp transfer)
RULER_SORTED_DIR = os.path.join(PROJECT_ROOT, "datasets", "ruler_sorted")
NORMAL_RULER_DIR = os.path.join(RULER_SORTED_DIR, "Normal_Ruler")
CLEAN_DIR        = os.path.join(RULER_SORTED_DIR, "Clean")

# Dedicated concept-images folder for this experiment only (kept separate
# from the shared concept_images/ used by the 10 clinical concepts)
CONCEPT_DIR = os.path.join(PROJECT_ROOT, "concept_images_ruler")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(CONCEPT_DIR, "ruler_present"), exist_ok=True)

# ─────────────────────────────────────────────
# 2. SYMLINK Normal_Ruler / Clean INTO THE EXPECTED
#    {concept_root}/positive, {concept_root}/negative STRUCTURE
# ─────────────────────────────────────────────
positive_link = os.path.join(CONCEPT_DIR, "ruler_present", "positive")
negative_link = os.path.join(CONCEPT_DIR, "ruler_present", "negative")

if not os.path.exists(positive_link):
    os.symlink(NORMAL_RULER_DIR, positive_link)
    print(f"Symlinked positive concept images: {positive_link} -> {NORMAL_RULER_DIR}")
else:
    print(f"Positive concept symlink already present: {positive_link}")

if not os.path.exists(negative_link):
    os.symlink(CLEAN_DIR, negative_link)
    print(f"Symlinked negative concept images: {negative_link} -> {CLEAN_DIR}")
else:
    print(f"Negative concept symlink already present: {negative_link}")

# ─────────────────────────────────────────────
# 3. MODEL FOLDER STRUCTURE
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
# 4. IMPORTS
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
# 5. CONCEPT — just the one
# ─────────────────────────────────────────────
CONCEPTS = ["ruler_present/positive"]

# ─────────────────────────────────────────────
# 6. TARGET CLASSES — MEL/NV primary comparison.
# To add a negative-control class later, e.g.:
#   CLASSES = ["MEL", "NV", "BCC"]
# ─────────────────────────────────────────────
CLASSES = ["MEL", "NV"]

# ─────────────────────────────────────────────
# 7. LAYERS — all 4 kept deliberately, see docstring
# ─────────────────────────────────────────────
LAYERS = [
    "conv5_block1_out",
    "conv5_block2_out",
    "conv5_block3_out",
    "post_relu",
]

# ─────────────────────────────────────────────
# 8. RUN GLOBAL VISUAL-TCAV PER CLASS
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

    plot_path = os.path.join(RESULTS_DIR, f"vtcav_global_ruler_{target_class}.png")
    global_visual_tcav.plot()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"Plot saved: {plot_path}")

print(f"\n{'='*60}")
print(f"All classes complete. [{MODEL_TAG}]")
print(f"Results: {RESULTS_DIR}")
print(f"Cache  : {CACHE_DIR}")
print(f"{'='*60}")