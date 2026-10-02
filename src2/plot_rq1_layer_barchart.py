"""
plot_rq1_layer_barchart.py

Regenerates Figure 5.1 (rq1_layer_barchart_MEL.png) of the thesis:
emblem-gated attribution toward MEL, per concept and per layer, mean +/- 2
standard errors over the MEL test set (n=200).

This figure previously had no script tracking how it was produced. It is
rebuilt here directly from the significance-testing pipeline's saved
per-cell mean/std (ttest_results_proper_ig_randomsample.json), the same
true-random-sample corrected results reported in Tables 5.3 and 5.4, so the
figure and the tables it sits beside are drawn from the identical corrected
data. Standard error is derived from the saved per-cell std as
std / sqrt(200), since the underlying 200 per-image attribution scores
are not themselves saved to disk (only their mean and std are).

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(
    PROJECT_ROOT, "outputs2", "vtcav_ttest_proper_ig_padonly_seed2_hardneg_randomsample",
    "ttest_results_proper_ig_randomsample.json"
)
OUT_PATH = os.path.join(PROJECT_ROOT, "outputs2", "thesis_figures", "rq1_layer_barchart_MEL.png")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

TARGET_CLASS = "MEL"
N_IMAGES = 200
LAYERS = ["conv5_block1_out", "conv5_block2_out", "conv5_block3_out", "post_relu"]
CONCEPT_ORDER = [
    "pigment_network_typical", "pigment_network_atypical",
    "streaks_regular", "streaks_irregular", "pigmentation",
    "regression_structures", "dots_and_globules_regular",
    "dots_and_globules_irregular", "blue_whitish_veil", "vascular_structures",
]
CONCEPT_LABELS = {
    "pigment_network_typical": "pigment\nnetwork\ntypical",
    "pigment_network_atypical": "pigment\nnetwork\natypical",
    "streaks_regular": "streaks\nregular",
    "streaks_irregular": "streaks\nirregular",
    "pigmentation": "pigmentation",
    "regression_structures": "regression\nstructures",
    "dots_and_globules_regular": "dots &\nglobules\nregular",
    "dots_and_globules_irregular": "dots &\nglobules\nirregular",
    "blue_whitish_veil": "blue\nwhitish\nveil",
    "vascular_structures": "vascular\nstructures",
}

with open(RESULTS_PATH) as f:
    results = json.load(f)

means = {layer: [] for layer in LAYERS}
errs = {layer: [] for layer in LAYERS}
for layer in LAYERS:
    for concept in CONCEPT_ORDER:
        cell = results[concept][layer][TARGET_CLASS]
        means[layer].append(cell["mean"])
        errs[layer].append(2 * cell["std"] / np.sqrt(N_IMAGES))

x = np.arange(len(CONCEPT_ORDER))
n_layers = len(LAYERS)
bar_width = 0.8 / n_layers
colors = plt.cm.viridis(np.linspace(0, 1, n_layers))

fig, ax = plt.subplots(figsize=(14.5, 6.5), dpi=150)
for i, layer in enumerate(LAYERS):
    offset = (i - (n_layers - 1) / 2) * bar_width
    ax.bar(
        x + offset, means[layer], bar_width,
        yerr=errs[layer], capsize=3,
        color=colors[i], label=layer, error_kw=dict(elinewidth=1.2),
    )

ax.set_xticks(x)
ax.set_xticklabels([CONCEPT_LABELS[c] for c in CONCEPT_ORDER])
ax.set_ylabel("Attribution toward MEL (mean $\\pm$ 2 s.e.)")
ax.set_title("resnet50v2 architecture — MEL target class")
ax.legend(title="Layer", loc="upper left")

fig.tight_layout()
fig.savefig(OUT_PATH)
print(f"Saved: {OUT_PATH}")
