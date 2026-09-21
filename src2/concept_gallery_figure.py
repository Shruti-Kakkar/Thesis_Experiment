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


def draw_concept_block(fig, gs, row, col_start, concept_name, image_paths):
    """One row of a block: concept-name label + its exemplar thumbnails."""
    ax_label = fig.add_subplot(gs[row, col_start])
    ax_label.axis('off')
    ax_label.text(
        1.0, 0.5, concept_name, ha='right', va='center',
        fontsize=9, fontfamily='monospace', wrap=True,
    )
    for i, path in enumerate(image_paths):
        ax_img = fig.add_subplot(gs[row, col_start + 1 + i])
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

    fig = plt.figure(figsize=(11, 7.2))
    gs = GridSpec(
        5, 7, width_ratios=[0.62, 1, 1, 0.18, 0.62, 1, 1],
        hspace=0.15, wspace=0.06, top=0.90, bottom=0.02, left=0.02, right=0.98,
    )
    for row in range(5):
        draw_concept_block(fig, gs, row, 0, left_concepts[row], exemplars[left_concepts[row]])
        draw_concept_block(fig, gs, row, 4, right_concepts[row], exemplars[right_concepts[row]])

    fig.suptitle(
        f"{N_EXEMPLARS} positive exemplars per concept, drawn directly from each "
        "concept's own CAV training pool",
        fontsize=10, style='italic', y=0.97,
    )
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

    fig = plt.figure(figsize=(6.4, 3.6))
    gs = GridSpec(2, 3, width_ratios=[0.75, 1, 1], hspace=0.2, wspace=0.06)
    draw_concept_block(fig, gs, 0, 0, "ruler_present", ruler_exemplars)
    draw_concept_block(fig, gs, 1, 0, "vignette_present", vignette_exemplars)

    fig.suptitle(
        f"{N_EXEMPLARS} positive exemplars per artifact concept "
        "(paired synthetic overlay, Section 4.4.3)",
        fontsize=9, style='italic', y=1.02,
    )
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    os.makedirs(THESIS_IMAGES_DIR, exist_ok=True)
    build_derm7pt_gallery(os.path.join(THESIS_IMAGES_DIR, "concept_gallery_derm7pt.png"))
    build_artifact_gallery(os.path.join(THESIS_IMAGES_DIR, "concept_gallery_artifacts.png"))
