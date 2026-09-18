"""
regenerate_local_heatmap_figures.py

Regenerates the thesis's 3 Local Visual-TCAV heatmap figures with a
clean original-image panel next to the heatmap overlay (instead of the
overlay alone) and a colorbar on the heatmap, so a reader can check a
figure's spatial claim (e.g. "the heatmap concentrates on the ruler's
tick marks, leaving the lesion above it unmarked") directly against the
unmodified image, and read the color scale instead of having to infer it.

Does NOT call LocalVisualTCAV.plot() (the shared library method in
src/VisualTCAV.py, used by many other experiments) -- builds a custom
figure directly from local_tcav.computations and local_tcav.imgs/
resized_imgs instead, so nothing in the shared library changes.

Source images identified by matching each figure's exact reported
attribution value against outputs2/vtcav_local_results_padonly_seed2_hardneg/
(the 32-image x 10-concept illustrative run Methodology already
documents) and outputs2/ruler_bias_ground_truth_verification_matched/
top_attribution_heatmaps/ (the ground-truth ruler run):
  - pigment_network_atypical/MEL: ISIC_0034329, attrib=0.039 (exact match)
  - vascular_structures/BCC (misclassified as SCC): ISIC_0034323,
    attrib=0.035 toward true label BCC (exact match)
  - ruler_present, rank-1 MEL ground-truth image: ISIC_0035130

CAVs are reused from the existing cache in every case (cache_cav=True,
matching n_cav_runs=20/max_examples=200/EXTRA_NEGATIVES exactly as the
original driver scripts) -- nothing is retrained.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import os
import numpy as np
import PIL.Image
import PIL.ImageFilter
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from prettytable import PrettyTable  # noqa: F401 (kept for parity with driver scripts, unused here)

PROJECT_ROOT = os.path.expanduser("~/scratch/dev-uos/projects/VTCAV_Dermatology")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from VisualTCAV import LocalVisualTCAV, Model, colormap
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_resnet_v2

THESIS_IMAGES_DIR = os.path.expanduser(
    "~/scratch/dev-uos/projects/Master_Thesis/Latex/"
    "Cognitive_Science_Universitaet_Osnabrueck/images"
)

GRAPH_FILENAME = "resnet50v2_isic2019_final.keras"
LABELS_FILENAME = "isic2019_classes.txt"
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "datasets", "test_images_by_class")


def build_model(max_examples=200):
    return Model(
        model_name="resnet50v2",
        graph_path_filename=GRAPH_FILENAME,
        label_path_filename=LABELS_FILENAME,
        preprocessing_function=preprocess_resnet_v2,
        max_examples=max_examples,
        resize_mode='pad',
    )


def render_figure(local_tcav, concept_name, layer_name, out_path, n_exemplars=3):
    """Custom replacement for LocalVisualTCAV.plot(): clean original +
    heatmap-with-colorbar side by side, same table and exemplar row."""
    concept_layer = local_tcav.computations[layer_name][concept_name]

    heatmap = tf.image.resize(
        np.expand_dims(concept_layer.concept_map, axis=2),
        [local_tcav.imgs[0].shape[0], local_tcav.imgs[0].shape[1]]
    )
    heatmap = np.reshape(heatmap, (heatmap.shape[0], heatmap.shape[1]))
    max_value = np.max(concept_layer.concept_map)
    heatmap = np.array(
        PIL.Image.fromarray(np.uint8(heatmap * 255), 'L')
        .filter(PIL.ImageFilter.GaussianBlur(radius=20))
    ) / 255
    if np.max(heatmap) > 0 and np.max(heatmap) < max_value:
        heatmap = (heatmap / np.max(heatmap)) * max_value

    fig = plt.figure(figsize=(9, 7.5))
    gs = GridSpec(3, 2, height_ratios=[1.1, 5, 1.6], hspace=0.12, wspace=0.08)

    # Row 0: Class / Attrib. table, spanning both columns
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis('off')
    rows = []
    for c in range(local_tcav.n_classes):
        attribution = concept_layer.attributions[c]
        class_name = (
            local_tcav.target_class if local_tcav.target_class is not None
            else local_tcav.predictions[0][c].class_name
        )
        attribution_str = f"{attribution:.3f}" if attribution >= 0.001 else f"{attribution:.1e}"
        rows.append([class_name, attribution_str])
    table = ax_table.table(
        cellText=rows, colLabels=["Class", "Attrib."],
        cellLoc='center', loc='center',
        colColours=["silver", "silver"],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(0.6, 1.8)

    # Row 1, col 0: clean original image
    ax_orig = fig.add_subplot(gs[1, 0])
    ax_orig.imshow(local_tcav.imgs[0])
    ax_orig.set_title("Original image", fontsize=10)
    ax_orig.axis('off')

    # Row 1, col 1: heatmap overlay, with colorbar
    ax_heat = fig.add_subplot(gs[1, 1])
    ax_heat.imshow(local_tcav.imgs[0])
    colormap.imshow(heatmap)
    ax_heat.set_title("Concept response overlay", fontsize=10)
    ax_heat.axis('off')
    sm = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=colormap.getLinearSegmentedColormap())
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Rescaled concept response\n(0 = none, 1 = strongest in image)", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    # Row 2: exemplar thumbnails, spanning both columns
    concept_images = local_tcav.model.activation_generator.get_images_for_concept(concept_name, False)
    n_show = min(n_exemplars, len(concept_images))
    gs_thumbs = gs[2, :].subgridspec(1, n_show, wspace=0.05)
    for i in range(n_show):
        ax_t = fig.add_subplot(gs_thumbs[0, i])
        ax_t.imshow(concept_images[i])
        ax_t.axis('off')
        if i == 0:
            ax_t.set_title("Positive concept exemplars used to train this CAV", fontsize=8, loc='left', x=0)

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"Saved: {out_path}")


# =========================================================================
# 1. pigment_network_atypical / MEL / ISIC_0034329
# =========================================================================
MODEL_TAG_1 = "padonly_seed2_hardneg"
VTCAV_DIR_1 = os.path.join(PROJECT_ROOT, "outputs2", f"vtcav_{MODEL_TAG_1}")
EXTRA_NEGATIVES = {
    "pigment_network_typical":  ["pigment_network_atypical/positive"],
    "pigment_network_atypical": ["pigment_network_typical/positive"],
    "streaks_regular":          ["streaks_irregular/positive"],
    "streaks_irregular":        ["streaks_regular/positive"],
    "dots_and_globules_regular":   ["dots_and_globules_irregular/positive"],
    "dots_and_globules_irregular": ["dots_and_globules_regular/positive"],
}

local_tcav_1 = LocalVisualTCAV(
    test_image_filename=os.path.join("MEL", "ISIC_0034329.jpg"),
    m_steps=50,
    target_class="MEL",
    model=build_model(),
    models_dir=os.path.join(VTCAV_DIR_1, "models"),
    cache_dir=os.path.join(VTCAV_DIR_1, "cache"),
    test_images_dir=TEST_IMAGES_DIR,
    concept_images_dir=os.path.join(PROJECT_ROOT, "concept_images"),
    negative_suffix="negative",
    extra_negative_concepts=EXTRA_NEGATIVES,
)
local_tcav_1.setLayers(layer_names=["post_relu"])
local_tcav_1.setConcepts(concept_names=["pigment_network_atypical/positive"])
local_tcav_1.predict()
local_tcav_1.explain(cache_cav=True, cache_random=True, n_cav_runs=20)
render_figure(
    local_tcav_1, "pigment_network_atypical/positive", "post_relu",
    os.path.join(THESIS_IMAGES_DIR, "local_heatmap_pigment_network_atypical.png"),
)

# =========================================================================
# 2. vascular_structures / BCC / ISIC_0034323
# =========================================================================
local_tcav_2 = LocalVisualTCAV(
    test_image_filename=os.path.join("BCC", "ISIC_0034323.jpg"),
    m_steps=50,
    target_class="BCC",
    model=build_model(),
    models_dir=os.path.join(VTCAV_DIR_1, "models"),
    cache_dir=os.path.join(VTCAV_DIR_1, "cache"),
    test_images_dir=TEST_IMAGES_DIR,
    concept_images_dir=os.path.join(PROJECT_ROOT, "concept_images"),
    negative_suffix="negative",
    extra_negative_concepts=EXTRA_NEGATIVES,
)
local_tcav_2.setLayers(layer_names=["post_relu"])
local_tcav_2.setConcepts(concept_names=["vascular_structures/positive"])
local_tcav_2.predict()
local_tcav_2.explain(cache_cav=True, cache_random=True, n_cav_runs=20)
render_figure(
    local_tcav_2, "vascular_structures/positive", "post_relu",
    os.path.join(THESIS_IMAGES_DIR, "local_heatmap_vascular_structures_misclassified.png"),
)

# =========================================================================
# 3. ruler_present, corrected CAV, rank-1 MEL ground-truth: ISIC_0035130
# =========================================================================
MODEL_TAG_3 = "padonly_seed2_rulerbias_matched"
VTCAV_DIR_3 = os.path.join(PROJECT_ROOT, "outputs2", f"vtcav_{MODEL_TAG_3}")

local_tcav_3 = LocalVisualTCAV(
    test_image_filename=os.path.join("MEL", "ISIC_0035130.jpg"),
    m_steps=50,
    target_class="MEL",
    model=build_model(),
    models_dir=os.path.join(VTCAV_DIR_3, "models"),
    cache_dir=os.path.join(VTCAV_DIR_3, "cache"),
    test_images_dir=TEST_IMAGES_DIR,
    concept_images_dir=os.path.join(PROJECT_ROOT, "concept_images_ruler_matched"),
    negative_suffix="negative",
)
local_tcav_3.setLayers(layer_names=["post_relu"])
local_tcav_3.setConcepts(concept_names=["ruler_present/positive"])
local_tcav_3.predict()
local_tcav_3.explain(cache_cav=True, cache_random=True, n_cav_runs=20)
render_figure(
    local_tcav_3, "ruler_present/positive", "post_relu",
    os.path.join(THESIS_IMAGES_DIR, "ruler_ground_truth_heatmap.png"),
)

print("\nAll 3 figures regenerated.")
