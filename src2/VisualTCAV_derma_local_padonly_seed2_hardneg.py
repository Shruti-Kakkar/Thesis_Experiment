"""
VisualTCAV_derma_local_padonly_seed2_hardneg.py
Runs LocalVisualTCAV on a set of example images to generate spatial
concept heatmaps -- "where in this image does the model see
pigment_network_atypical" etc.

SCOPE (expanded from the initial 4-image, 2-concept pilot):
  - 4 example images per class, all 8 classes (32 images total, auto-picked
    -- first 4 .jpg files found in each class folder, alphabetically)
  - All 10 concepts
  - Only post_relu (kept to one layer deliberately -- this already
    produces 32 images x 10 concepts = 320 heatmap PNGs; adding more
    layers would multiply that further)

Reuses the SAME model, MODEL_TAG, and EXTRA_NEGATIVES config as the
hard-negative Global run, so this reuses the already-computed CAV cache
(n_cav_runs=20) rather than recomputing anything -- should be fast per
image; the main cost here is simply the number of images x concepts.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 0. MODEL VARIANT TAG — same as the hard-negative Global/t-test run,
# so this reuses that exact CAV cache instead of recomputing.
# ─────────────────────────────────────────────
MODEL_TAG = "padonly_seed2_hardneg"

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
CONCEPT_DIR     = os.path.join(PROJECT_ROOT, "concept_images")                     # shared
RESULTS_DIR     = os.path.join(PROJECT_ROOT, "outputs2", f"vtcav_local_results_{MODEL_TAG}")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 2. MODEL FOLDER STRUCTURE — reuses the same symlink the Global script
# already created for this MODEL_TAG.
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
    print(f"Symlink already present: {graph_dest}")

CLASSES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
labels_dest = os.path.join(MODEL_SUBDIR, LABELS_FILENAME)
if not os.path.exists(labels_dest):
    with open(labels_dest, 'w') as f:
        for cls in CLASSES:
            f.write(cls + "\n")
    print(f"Labels file written: {labels_dest}")

# ─────────────────────────────────────────────
# 3. IMPORTS
# ─────────────────────────────────────────────
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from VisualTCAV import LocalVisualTCAV, Model
from tensorflow.keras.applications.resnet_v2 import (
    preprocess_input as preprocess_resnet_v2
)
print("VisualTCAV imported successfully.\n")

# ─────────────────────────────────────────────
# 4. EXTRA NEGATIVES — must match the hard-negative Global run exactly,
# so the CAV cache key lines up and gets reused instead of recomputed.
# ─────────────────────────────────────────────
EXTRA_NEGATIVES = {
    "pigment_network_typical":  ["pigment_network_atypical/positive"],
    "pigment_network_atypical": ["pigment_network_typical/positive"],
    "streaks_regular":          ["streaks_irregular/positive"],
    "streaks_irregular":        ["streaks_regular/positive"],
    "dots_and_globules_regular":   ["dots_and_globules_irregular/positive"],
    "dots_and_globules_irregular": ["dots_and_globules_regular/positive"],
}

# ─────────────────────────────────────────────
# 5. CONCEPTS + LAYERS — deliberately small for this first look
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
LAYERS = ["post_relu"]  # kept to one layer deliberately -- see file count note below

# ─────────────────────────────────────────────
# 6. PICK EXAMPLE IMAGES — first 2 .jpg files found per class,
# alphabetically. No need to know filenames in advance.
# ─────────────────────────────────────────────
def pick_example_images(class_name, n=2):
    class_dir = os.path.join(TEST_IMAGES_DIR, class_name)
    files = sorted(
        f for f in os.listdir(class_dir)
        if f.lower().endswith('.jpg')
    )
    picked = files[:n]
    print(f"  {class_name}: picked {picked}")
    return [os.path.join(class_name, f) for f in picked]  # relative to TEST_IMAGES_DIR

print("Picking example images:")
example_images = []
for cls in CLASSES:
    example_images += [(cls, rel_path) for rel_path in pick_example_images(cls, n=4)]

# ─────────────────────────────────────────────
# 7. RUN LOCAL VISUAL-TCAV PER IMAGE
# ─────────────────────────────────────────────
for true_class, rel_image_path in example_images:

    image_label = os.path.splitext(os.path.basename(rel_image_path))[0]
    print(f"\n{'='*60}")
    print(f"LocalVisualTCAV — image: {rel_image_path} (true class: {true_class})")
    print(f"{'='*60}")

    local_tcav = LocalVisualTCAV(
        test_image_filename=rel_image_path,
        m_steps=50,
        target_class=true_class,   # explain the TRUE label, not top-3 predictions

        # Model
        model=Model(
            model_name="resnet50v2",
            graph_path_filename=GRAPH_FILENAME,
            label_path_filename=LABELS_FILENAME,
            preprocessing_function=preprocess_resnet_v2,
            max_examples=200,
            resize_mode='pad',
        ),

        # Paths
        models_dir=MODELS_DIR,
        cache_dir=CACHE_DIR,
        test_images_dir=TEST_IMAGES_DIR,
        concept_images_dir=CONCEPT_DIR,
        negative_suffix="negative",
        extra_negative_concepts=EXTRA_NEGATIVES,
    )

    local_tcav.setLayers(layer_names=LAYERS)
    local_tcav.setConcepts(concept_names=CONCEPTS)

    # Predict first -- explain() requires this to have run
    predictions = local_tcav.predict()
    predictions.info(num_of_classes=3)

    # n_cav_runs=20 MUST match the Global script's n_cav_runs, or this
    # builds a differently-keyed cache instead of reusing the existing one.
    local_tcav.explain(cache_cav=True, cache_random=True, n_cav_runs=20)

    # MODIFICATION: LocalVisualTCAV.plot() calls plt.show(), which is a
    # no-op on a headless server (no display over SSH) -- would silently
    # produce nothing. Redirect it to save each concept's figure to disk
    # instead, without touching VisualTCAV.py itself.
    #
    # ALSO stamps the ISIC ID and true class directly onto the figure
    # (not just the filename) -- this survives being cropped, uploaded,
    # or viewed out of order, unlike relying on filenames alone.
    save_counter = {"i": 0}
    def _save_instead_of_show():
        idx = save_counter["i"]
        concept_name = (
            local_tcav.concepts[idx] if idx < len(local_tcav.concepts)
            else f"concept{idx}"
        )
        safe_concept = concept_name.replace("/", "_")
        fname = os.path.join(
            RESULTS_DIR, f"local_{image_label}_{true_class}_{safe_concept}.png"
        )

        # Stamp ID + true class as a visible banner burned into the image
        fig = plt.gcf()
        fig.text(
            0.5, 1.0,
            f"{image_label}  |  true class: {true_class}",
            ha='center', va='top', fontsize=13, color='black',
            fontweight='bold',
            bbox=dict(facecolor='yellow', alpha=0.85, edgecolor='black', pad=4),
        )

        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f"  Saved: {fname}")
        save_counter["i"] += 1
    plt.show = _save_instead_of_show

    local_tcav.plot()

print(f"\n{'='*60}")
print("All example images complete.")
print(f"Results: {RESULTS_DIR}")
print(f"{'='*60}")