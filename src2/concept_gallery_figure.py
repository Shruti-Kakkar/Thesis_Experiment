"""
concept_gallery_figure.py

Builds two exemplar-gallery figures for the thesis Methodology chapter,
each showing 2 positive example images per concept directly from the
same concept_images directories the CAV pipeline trains on:

  - concept_gallery_derm7pt.png: the 10 derm7pt-derived clinical concepts
    (Table~\ref{tab:concepts}, Section~\ref{sec:derm7pt}), 2 exemplars each.
  - concept_gallery_artifacts.png: the 2 synthetic artifact concepts
    (ruler_present, vignette_present; Section~\ref{sec:rq3_design}),
    2 exemplars each.
  - ruler_style_real_vs_synthetic.png: for each of the 4 ruler styles
    RQ3 decomposes ruler_present by (thick_edge, ticks_edge, lesion_side,
    short_ruler; Section~\ref{sec:rq3_results}), a real photograph
    genuinely exhibiting that style next to our synthetic overlay
    recreation of it.
  - class_gallery.png: the 8 ISIC 2019 diagnostic classes the
    classifier predicts (Section~\ref{sec:datasets}), 2 example
    training-set images each, picked via the one-hot
    ISIC_2019_Training_GroundTruth.csv rather than a pre-sorted
    per-class folder (no such folder exists for the training set).

These are plain image galleries, no heatmap or model involved, so this
script only needs PIL/matplotlib -- no TensorFlow, no VisualTCAV import,
and no dependency on any particular CAV cache or trained model.

Exemplar selection: within each figure, the first candidate image (sorted
filename order) not already used elsewhere in that same figure is picked,
so the gallery does not repeat the same lesion photo under two different
concept labels purely because it happens to sort first in both
directories.

Author: Shruti Kakkar
"""

import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import PIL.Image

PROJECT_ROOT = os.path.expanduser("~/scratch/dev-uos/projects/VTCAV_Dermatology")
THESIS_IMAGES_DIR = os.path.expanduser(
    "~/scratch/dev-uos/projects/Master_Thesis/Latex/"
    "Cognitive_Science_Universitaet_Osnabrueck/images"
)

DERM7PT_CONCEPT_DIR = os.path.join(PROJECT_ROOT, "concept_images")
RULER_CONCEPT_DIR = os.path.join(PROJECT_ROOT, "concept_images_ruler_matched")
VIGNETTE_CONCEPT_DIR = os.path.join(PROJECT_ROOT, "concept_images_vignette_matched")
TRAIN_IMAGES_DIR = os.path.join(PROJECT_ROOT, "datasets", "ISIC_2019_Training_Input", "ISIC_2019_Training_Input")
TRAIN_GROUND_TRUTH_CSV = os.path.join(PROJECT_ROOT, "datasets", "ISIC_2019_Training_GroundTruth.csv")

N_EXEMPLARS = 2

# Images with a masking/occlusion artifact unrelated to the concept itself
# (e.g. a black staircase-shaped slide-mount edge cutting across the frame),
# excluded purely for gallery legibility -- not excluded from CAV training.
DENYLIST = {"Abl129.jpg"}


def pick_exemplars(positive_dir, n, used):
    """First n filenames (sorted) not already in `used`; adds picks to `used`."""
    candidates = sorted(os.listdir(positive_dir))
    picks = []
    for fname in candidates:
        if fname in used or fname in DENYLIST:
            continue
        picks.append(os.path.join(positive_dir, fname))
        used.add(fname)
        if len(picks) == n:
            break
    if len(picks) < n:
        raise RuntimeError(f"Not enough unused images in {positive_dir}")
    return picks


def draw_concept_block(fig, gs, title_row, image_row, col_start, concept_name, image_paths):
    """A concept's name, centered above its own pair of exemplar thumbnails.

    The label sits in its own axes spanning exactly the two image columns
    below it, rather than squeezed into a narrow side column: a long
    concept name (e.g. dots_and_globules_irregular) previously overflowed
    sideways into the neighbouring concept's thumbnails when right-aligned
    in a cramped label column.
    """
    ax_label = fig.add_subplot(gs[title_row, col_start:col_start + 2])
    ax_label.axis('off')
    ax_label.text(
        0.5, 0.0, concept_name, ha='center', va='bottom',
        fontsize=9, fontfamily='monospace',
    )
    for i, path in enumerate(image_paths):
        ax_img = fig.add_subplot(gs[image_row, col_start + i])
        ax_img.imshow(PIL.Image.open(path))
        ax_img.axis('off')


def build_derm7pt_gallery(out_path):
    concept_order = [
        "pigment_network_atypical", "pigment_network_typical",
        "streaks_irregular", "streaks_regular",
        "pigmentation",
        "regression_structures",
        "dots_and_globules_irregular", "dots_and_globules_regular",
        "blue_whitish_veil", "vascular_structures",
    ]
    left_concepts, right_concepts = concept_order[:5], concept_order[5:]

    used = set()
    exemplars = {}
    for concept in concept_order:
        positive_dir = os.path.join(DERM7PT_CONCEPT_DIR, concept, "positive")
        exemplars[concept] = pick_exemplars(positive_dir, N_EXEMPLARS, used)

    fig = plt.figure(figsize=(10, 8.5))
    n_pairs = 5
    height_ratios = [0.28, 1] * n_pairs
    gs = GridSpec(
        2 * n_pairs, 5, width_ratios=[1, 1, 0.2, 1, 1], height_ratios=height_ratios,
        hspace=0.08, wspace=0.06, top=0.98, bottom=0.02, left=0.02, right=0.98,
    )
    for i in range(n_pairs):
        title_row, image_row = 2 * i, 2 * i + 1
        draw_concept_block(fig, gs, title_row, image_row, 0, left_concepts[i], exemplars[left_concepts[i]])
        draw_concept_block(fig, gs, title_row, image_row, 3, right_concepts[i], exemplars[right_concepts[i]])

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"Saved: {out_path}")


CLASS_FULL_NAMES = {
    "MEL": "melanoma", "NV": "melanocytic nevus", "BCC": "basal cell carcinoma",
    "AK": "actinic keratosis", "BKL": "benign keratosis", "DF": "dermatofibroma",
    "VASC": "vascular lesion", "SCC": "squamous cell carcinoma",
}


def load_training_image_ids_by_class():
    """image_id -> class, from the one-hot ISIC_2019_Training_GroundTruth.csv."""
    class_order = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    ids_by_class = {cls: [] for cls in class_order}
    with open(TRAIN_GROUND_TRUTH_CSV, newline="") as f:
        for row in csv.DictReader(f):
            for cls in class_order:
                if row[cls] == "1.0":
                    ids_by_class[cls].append(row["image"])
                    break
    return ids_by_class


def pick_training_exemplars(image_ids, n, used):
    """Same dedup logic as pick_exemplars, but over a list of training
    image ids (from the ground-truth CSV) rather than a directory listing."""
    picks = []
    for image_id in sorted(image_ids):
        if image_id in used:
            continue
        picks.append(os.path.join(TRAIN_IMAGES_DIR, image_id + ".jpg"))
        used.add(image_id)
        if len(picks) == n:
            break
    if len(picks) < n:
        raise RuntimeError(f"Not enough unused training images for this class")
    return picks


def build_class_gallery(out_path):
    class_order = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    left_classes, right_classes = class_order[:4], class_order[4:]

    ids_by_class = load_training_image_ids_by_class()
    used = set()
    exemplars = {}
    for cls in class_order:
        exemplars[cls] = pick_training_exemplars(ids_by_class[cls], N_EXEMPLARS, used)

    fig = plt.figure(figsize=(10, 7.2))
    n_pairs = 4
    height_ratios = [0.32, 1] * n_pairs
    gs = GridSpec(
        2 * n_pairs, 5, width_ratios=[1, 1, 0.2, 1, 1], height_ratios=height_ratios,
        hspace=0.1, wspace=0.06, top=0.98, bottom=0.02, left=0.02, right=0.98,
    )
    for i in range(n_pairs):
        title_row, image_row = 2 * i, 2 * i + 1
        left_label = f"{left_classes[i]} ({CLASS_FULL_NAMES[left_classes[i]]})"
        right_label = f"{right_classes[i]} ({CLASS_FULL_NAMES[right_classes[i]]})"
        draw_concept_block(fig, gs, title_row, image_row, 0, left_label, exemplars[left_classes[i]])
        draw_concept_block(fig, gs, title_row, image_row, 3, right_label, exemplars[right_classes[i]])

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"Saved: {out_path}")


def pick_pair_filenames(positive_dir, n, used):
    """n filenames (sorted) present in `positive_dir`, not already in `used`.

    Unlike pick_exemplars, this returns bare filenames rather than full
    paths, since the caller needs to look the same filename up in both
    the positive/ and negative/ subfolders of a paired concept set.
    """
    candidates = sorted(os.listdir(positive_dir))
    picks = []
    for fname in candidates:
        if fname in used:
            continue
        picks.append(fname)
        used.add(fname)
        if len(picks) == n:
            break
    if len(picks) < n:
        raise RuntimeError(f"Not enough unused images in {positive_dir}")
    return picks


def draw_pair_block(fig, gs, title_row, concept_name, concept_dir, filenames):
    """Concept name above N rows of (clean negative | synthetic-overlay
    positive) pairs, same base image on both sides of each row -- makes
    the paired-construction claim (Section~\ref{sec:rq3_design}: positive
    and negative examples are pixel-identical outside the injected
    region) directly checkable by eye, not just asserted in text.
    """
    ax_label = fig.add_subplot(gs[title_row, :])
    ax_label.axis('off')
    ax_label.text(0.5, 0.0, concept_name, ha='center', va='bottom', fontsize=10, fontfamily='monospace')
    for i, fname in enumerate(filenames):
        image_row = title_row + 1 + i
        for col, subdir in enumerate(["negative", "positive"]):
            ax_img = fig.add_subplot(gs[image_row, col])
            ax_img.imshow(PIL.Image.open(os.path.join(concept_dir, subdir, fname)))
            ax_img.axis('off')


N_PAIRS = 2


def build_artifact_gallery(out_path):
    used_ruler, used_vignette = set(), set()
    ruler_files = pick_pair_filenames(os.path.join(RULER_CONCEPT_DIR, "ruler_present", "positive"), N_PAIRS, used_ruler)
    vignette_files = pick_pair_filenames(os.path.join(VIGNETTE_CONCEPT_DIR, "vignette_present", "positive"), N_PAIRS, used_vignette)

    fig = plt.figure(figsize=(5.2, 8.6))
    rows_per_concept = 1 + N_PAIRS
    gs = GridSpec(
        1 + 2 * rows_per_concept, 2, height_ratios=[0.22] + [0.28, 1, 1] * 2,
        hspace=0.1, wspace=0.06, top=0.98, bottom=0.02, left=0.02, right=0.98,
    )

    for col, header in enumerate(["Clean (negative)", "Synthetic overlay (positive)"]):
        ax_header = fig.add_subplot(gs[0, col])
        ax_header.axis('off')
        ax_header.text(0.5, 0.0, header, ha='center', va='bottom', fontsize=10, fontweight='bold')

    draw_pair_block(fig, gs, 1, "ruler_present", os.path.join(RULER_CONCEPT_DIR, "ruler_present"), ruler_files)
    draw_pair_block(fig, gs, 1 + rows_per_concept, "vignette_present", os.path.join(VIGNETTE_CONCEPT_DIR, "vignette_present"), vignette_files)

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"Saved: {out_path}")


# Real photograph paired with a synthetic overlay recreation, per ruler
# style. The real image for thick_edge and ticks_edge comes from the
# hand-sorted real-photo folder each synthetic style is explicitly named
# after (Thick_Ruler, Normal_Ruler); lesion_side and short_ruler have no
# such dedicated real-photo folder, so their real example was instead
# picked by visual inspection of the closest-matching pattern in the
# hand-labelled ground-truth ruler set used for the generalisation check
# in Section~\ref{sec:rq3_results} (datasets/ruler_sorted_test/).
RULER_STYLE_EXAMPLES = [
    (
        "thick_edge", "Thick",
        "datasets/ruler_sorted/Thick_Ruler/ISIC_0010077.jpg",
        "concept_images_ruler_matched/ruler_present/positive/ISIC_0000081_downsampled.jpg",
    ),
    (
        "ticks_edge", "Normal",
        "datasets/ruler_sorted/Normal_Ruler/ISIC_0000110_downsampled.jpg",
        "concept_images_ruler_matched/ruler_present/positive/ISIC_0000013.jpg",
    ),
    (
        "lesion_side", "Lesion-side",
        "datasets/ruler_sorted/Normal_Ruler/ISIC_0012227_downsampled.jpg",
        "concept_images_ruler_matched/ruler_present/positive/ISIC_0000000.jpg",
    ),
    (
        "short_ruler", "Short",
        "datasets/ruler_sorted_test/MEL_test_ruler/ISIC_0034354.jpg",
        "concept_images_ruler_matched/ruler_present/positive/ISIC_0000003.jpg",
    ),
]


def build_ruler_style_gallery(out_path):
    n_styles = len(RULER_STYLE_EXAMPLES)
    fig = plt.figure(figsize=(6.4, 11))
    gs = GridSpec(
        1 + 2 * n_styles, 2, height_ratios=[0.22] + [0.22, 1] * n_styles,
        hspace=0.1, wspace=0.08, top=0.98, bottom=0.02, left=0.04, right=0.98,
    )

    for col, header in enumerate(["Real photograph", "Synthetic overlay"]):
        ax_header = fig.add_subplot(gs[0, col])
        ax_header.axis('off')
        ax_header.text(0.5, 0.0, header, ha='center', va='bottom', fontsize=11, fontweight='bold')

    for i, (concept_name, plain_name, real_rel_path, synth_rel_path) in enumerate(RULER_STYLE_EXAMPLES):
        title_row, image_row = 1 + 2 * i, 1 + 2 * i + 1
        ax_label = fig.add_subplot(gs[title_row, :])
        ax_label.axis('off')
        ax_label.text(
            0.5, 0.0, f"{plain_name}  ({concept_name})", ha='center', va='bottom',
            fontsize=10, fontfamily='monospace',
        )
        for col, rel_path in enumerate([real_rel_path, synth_rel_path]):
            ax_img = fig.add_subplot(gs[image_row, col])
            ax_img.imshow(PIL.Image.open(os.path.join(PROJECT_ROOT, rel_path)))
            ax_img.axis('off')

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    os.makedirs(THESIS_IMAGES_DIR, exist_ok=True)
    build_derm7pt_gallery(os.path.join(THESIS_IMAGES_DIR, "concept_gallery_derm7pt.png"))
    build_artifact_gallery(os.path.join(THESIS_IMAGES_DIR, "concept_gallery_artifacts.png"))
    build_ruler_style_gallery(os.path.join(THESIS_IMAGES_DIR, "ruler_style_real_vs_synthetic.png"))
    build_class_gallery(os.path.join(THESIS_IMAGES_DIR, "class_gallery.png"))
