"""
plot_sampling_bug_comparison.py

PURPOSE: the RQ4 sampling-procedure figure (Section 5.6.1,
sampling_bug_comparison.png): per-image post_relu attribution from the
local screening pipeline, under the biased files[:n] sample vs. the
corrected random sample, for vignette_present and ruler_present side by
side. Pure data visualization of the per-image CSVs the screening scripts
already saved -- no model compute.

NOTE on what is plotted: each box pools all 50 screened images (25 MEL +
25 NV), exactly as the thesis figure does. The screening scripts score
every image toward its OWN true class (target_class=true_class in
*_bias_common_matched.py), so the 25 NV points are attribution toward NV,
not MEL -- even though the thesis figure's y-axis and caption say "toward
MEL". The label below is kept as-is so this script reproduces the figure
in the thesis; the MEL-only means printed alongside are for reference.

Inputs:
  - biased:    outputs2/{concept}_bias_local_screening_matched_biased_sample_archive/attributions.csv
  - corrected: outputs2/{concept}_bias_local_screening_matched/attributions.csv

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.expanduser(
    "~/scratch/dev-uos/projects/VTCAV_Dermatology"
)
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs2")
TARGET_CLASS = "MEL"

CONCEPTS = {
    "vignette_present": "vignette_bias_local_screening_matched",
    "ruler_present":    "ruler_bias_local_screening_matched",
}

OUT_PATH = os.path.join(OUTPUTS_DIR, "sampling_bug_comparison.png")


def load_attributions(results_dir):
    df = pd.read_csv(os.path.join(OUTPUTS_DIR, results_dir, "attributions.csv"))
    mel_only = df.loc[df["class"] == TARGET_CLASS, "attribution"].to_numpy()
    return df["attribution"].to_numpy(), mel_only


rng = np.random.default_rng(0)  # jitter only
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (concept, results_dir) in zip(axes, CONCEPTS.items()):
    biased, biased_mel = load_attributions(f"{results_dir}_biased_sample_archive")
    corrected, corrected_mel = load_attributions(results_dir)
    print(f"{concept}: biased n={len(biased)} mean={biased.mean():.6f} "
          f"(MEL-only {biased_mel.mean():.6f}) | corrected n={len(corrected)} "
          f"mean={corrected.mean():.6f} (MEL-only {corrected_mel.mean():.6f})")

    ax.boxplot([biased, corrected], showmeans=True, widths=0.5)
    for pos, values in enumerate([biased, corrected], start=1):
        ax.scatter(pos + rng.uniform(-0.08, 0.08, len(values)), values,
                   s=16, alpha=0.5, color="tab:blue", zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["files[:n]\n(biased)", "random\n(corrected)"])
    ax.set_ylabel(f"post_relu attribution ({TARGET_CLASS})")
    ax.set_title(concept)

fig.suptitle("Effect of the files[:n] sampling bug on local screening attribution")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
