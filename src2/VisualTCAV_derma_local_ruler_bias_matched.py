"""
VisualTCAV_derma_local_ruler_bias_matched.py
Same as VisualTCAV_derma_local_ruler_bias.py, but explains images using
the CORRECTED 'ruler_present' CAV (see
VisualTCAV_derma_global_ruler_bias_matched.py for the concept-set design
and why MODEL_TAG must differ from the confounded run).

KEY POINT: the 8 example images themselves are UNCHANGED -- still the
same real, ground-truth-sorted Normal_Ruler/Clean photos as the
confounded run (2 each: MEL+ruler, NV+ruler, MEL+clean, NV+clean). Only
the CAV doing the explaining is different. That's deliberate: the
question this script answers is "does attribution land on the physical
ruler in a REAL photo", and holding the explained images fixed while
swapping the CAV is what makes the confounded-vs-corrected heatmaps
directly comparable.

Reuses the SAME model, MODEL_TAG, and ruler_present CAV cache as
VisualTCAV_derma_global_ruler_bias_matched.py (n_cav_runs=20 must match
exactly to hit that cache) -- should not need to recompute the CAV
itself.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# ─────────────────────────────────────────────
# 0. MODEL VARIANT TAG — same as the Global matched run
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

VTCAV_DIR   = os.path.join(PROJECT_ROOT, "outputs2", f"vtcav_{MODEL_TAG}")
MODELS_DIR  = os.path.join(VTCAV_DIR, "models")
CACHE_DIR   = os.path.join(VTCAV_DIR, "cache")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs2", f"vtcav_local_results_{MODEL_TAG}")

# Real photos being explained -- unchanged from the confounded run
RULER_SORTED_DIR = os.path.join(PROJECT_ROOT, "datasets", "ruler_sorted")
NORMAL_RULER_DIR = os.path.join(RULER_SORTED_DIR, "Normal_Ruler")
CLEAN_DIR        = os.path.join(RULER_SORTED_DIR, "Clean")

# Corrected concept set the CAV is trained from -- different from what's explained
CONCEPT_DIR = os.path.join(PROJECT_ROOT, "concept_images_ruler_matched")
TRAIN_CSV   = os.path.join(PROJECT_ROOT, "datasets", "ISIC_2019_Training_GroundTruth.csv")

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

# ─────────────────────────────────────────────
# 2. MODEL FOLDER STRUCTURE — reuses the Global matched run's symlink
# ─────────────────────────────────────────────
MODEL_SUBDIR    = os.path.join(MODELS_DIR, "resnet50v2")
GRAPH_FILENAME  = "resnet50v2_isic2019_final.keras"
LABELS_FILENAME = "isic2019_classes.txt"

os.makedirs(MODEL_SUBDIR, exist_ok=True)

graph_dest = os.path.join(MODEL_SUBDIR, GRAPH_FILENAME)
if not os.path.exists(graph_dest):
    os.symlink(SOURCE_MODEL_PATH, graph_dest)
    print(f"Symlinked model to: {graph_dest}")

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
labels_dest = os.path.join(MODEL_SUBDIR, LABELS_FILENAME)
if not os.path.exists(labels_dest):
    with open(labels_dest, 'w') as f:
        for cls in CLASS_NAMES:
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

CONCEPTS = ["ruler_present/positive"]
LAYERS = [
    "conv5_block1_out",
    "conv5_block2_out",
    "conv5_block3_out",
    "post_relu",
]

# ─────────────────────────────────────────────
# 4. GROUND-TRUTH ID -> CLASS LOOKUP (same logic as ruler_prevalence_by_class.py)
# ─────────────────────────────────────────────
df = pd.read_csv(TRAIN_CSV)
df['label'] = df[CLASS_NAMES].values.argmax(axis=1)
df['class'] = df['label'].apply(lambda i: CLASS_NAMES[i])

def normalize_id(filename):
    base = os.path.splitext(filename)[0]
    return base.replace('_downsampled', '')

id_to_class = dict(zip(df['image'].apply(normalize_id), df['class']))

# ─────────────────────────────────────────────
# 5. PICK EXAMPLES: same 2 each of MEL+ruler, NV+ruler, MEL+clean, NV+clean
# ─────────────────────────────────────────────
def pick_examples(folder_path, target_classes, n_per_class=2):
    picks = {cls: [] for cls in target_classes}
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        cls = id_to_class.get(normalize_id(fname))
        if cls in picks and len(picks[cls]) < n_per_class:
            picks[cls].append(fname)
        if all(len(v) >= n_per_class for v in picks.values()):
            break
    return picks

ruler_picks = pick_examples(NORMAL_RULER_DIR, ["MEL", "NV"], n_per_class=2)
clean_picks = pick_examples(CLEAN_DIR, ["MEL", "NV"], n_per_class=2)

print("Picked examples:")
print(f"  Normal_Ruler MEL: {ruler_picks['MEL']}")
print(f"  Normal_Ruler NV:  {ruler_picks['NV']}")
print(f"  Clean MEL:        {clean_picks['MEL']}")
print(f"  Clean NV:         {clean_picks['NV']}")

# (folder_name, subfolder_relative_path, true_class, has_ruler, filename)
example_images = []
for cls, files in ruler_picks.items():
    for f in files:
        example_images.append(("Normal_Ruler", cls, True, f))
for cls, files in clean_picks.items():
    for f in files:
        example_images.append(("Clean", cls, False, f))

# ─────────────────────────────────────────────
# 6. RUN LOCAL VISUAL-TCAV PER IMAGE
# ─────────────────────────────────────────────
for subfolder, true_class, has_ruler, fname in example_images:

    rel_image_path = os.path.join(subfolder, fname)
    image_label = os.path.splitext(fname)[0]
    ruler_tag = "ruler" if has_ruler else "noruler"

    print(f"\n{'='*60}")
    print(f"LocalVisualTCAV [{MODEL_TAG}] — image: {rel_image_path} "
          f"(true class: {true_class}, has_ruler: {has_ruler})")
    print(f"{'='*60}")

    local_tcav = LocalVisualTCAV(
        test_image_filename=rel_image_path,
        m_steps=50,
        target_class=true_class,

        model=Model(
            model_name="resnet50v2",
            graph_path_filename=GRAPH_FILENAME,
            label_path_filename=LABELS_FILENAME,
            preprocessing_function=preprocess_resnet_v2,
            max_examples=200,   # must match Global matched script's value -- cache key depends on it
            resize_mode='pad',
        ),

        models_dir=MODELS_DIR,
        cache_dir=CACHE_DIR,
        test_images_dir=RULER_SORTED_DIR,   # base dir; rel_image_path is relative to this
        concept_images_dir=CONCEPT_DIR,
        negative_suffix="negative",
    )

    local_tcav.setLayers(layer_names=LAYERS)
    local_tcav.setConcepts(concept_names=CONCEPTS)

    predictions = local_tcav.predict()
    predictions.info(num_of_classes=3)

    # n_cav_runs=20 MUST match the Global matched script's n_cav_runs
    local_tcav.explain(cache_cav=True, cache_random=True, n_cav_runs=20)

    # Stamp ID + true class + ruler ground truth directly onto the image,
    # then save (redirecting plt.show(), same as the clinical-concepts script)
    save_counter = {"i": 0}
    def _save_instead_of_show():
        idx = save_counter["i"]
        concept_name = (
            local_tcav.concepts[idx] if idx < len(local_tcav.concepts)
            else f"concept{idx}"
        )
        safe_concept = concept_name.replace("/", "_")
        fname_out = os.path.join(
            RESULTS_DIR,
            f"local_matched_{image_label}_{true_class}_{ruler_tag}_{safe_concept}.png"
        )

        fig = plt.gcf()
        fig.text(
            0.5, 1.0,
            f"{image_label}  |  true class: {true_class}  |  "
            f"ruler present: {'YES' if has_ruler else 'NO'}",
            ha='center', va='top', fontsize=12, color='black',
            fontweight='bold',
            bbox=dict(
                facecolor='yellow' if has_ruler else '#ccffcc',
                alpha=0.9, edgecolor='black', pad=4
            ),
        )

        plt.savefig(fname_out, dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f"  Saved: {fname_out}")
        save_counter["i"] += 1
    plt.show = _save_instead_of_show

    local_tcav.plot()

print(f"\n{'='*60}")
print("All example images complete.")
print(f"Results: {RESULTS_DIR}")
print(f"{'='*60}")
