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


def build_artifact_gallery(out_path):
    used_ruler, used_vignette = set(), set()
    ruler_exemplars = pick_exemplars(
        os.path.join(RULER_CONCEPT_DIR, "ruler_present", "positive"), N_EXEMPLARS, used_ruler
    )
    vignette_exemplars = pick_exemplars(
        os.path.join(VIGNETTE_CONCEPT_DIR, "vignette_present", "positive"), N_EXEMPLARS, used_vignette
    )

    fig = plt.figure(figsize=(5, 4.4))
    gs = GridSpec(
        4, 2, height_ratios=[0.22, 1, 0.22, 1],
        hspace=0.08, wspace=0.06, top=0.98, bottom=0.02, left=0.02, right=0.98,
    )
    draw_concept_block(fig, gs, 0, 1, 0, "ruler_present", ruler_exemplars)
    draw_concept_block(fig, gs, 2, 3, 0, "vignette_present", vignette_exemplars)

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
        "concept_images_ruler_matched/ruler_present/positive/ISIC_0000010.jpg",
    ),
    (
        "ticks_edge", "Normal",
        "datasets/ruler_sorted/Normal_Ruler/ISIC_0000110_downsampled.jpg",
        "concept_images_ruler_matched/ruler_present/positive/ISIC_0000013.jpg",
    ),
    (
        "lesion_side", "Lesion-side",
        "datasets/ruler_sorted/Thick_Ruler/ISIC_0000176_downsampled.jpg",
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
