"""
ruler_bias_ground_truth_finalize_matched.py
Reads the CSV built by (possibly several chunked runs of)
ruler_bias_ground_truth_verification_matched.py and checks, per class,
whether the corrected 'ruler_present' CAV's attribution is actually
higher for images with a real ruler than for clean images.

STATISTICAL TEST: two-sided Mann-Whitney U (scipy.stats.mannwhitneyu),
not a t-test -- attribution values are bounded near 0 and typically
right-skewed (most images get ~0, a minority get most of the signal,
same pattern seen in the local screening summary stats), and the NV
ruler group is only 11 images, too few to lean on a normality
assumption. Mann-Whitney tests whether ruler-labeled attributions tend
to be stochastically larger than clean-labeled ones, without assuming a
particular distribution shape.

Also saves a box+strip plot per class (attribution by ground-truth
group) to RESULTS_DIR for a visual sanity check alongside the test.

Run this once after all scoring chunks have completed.

Author: Shruti Kakkar
"""

import csv
import os

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ruler_bias_ground_truth_verification_matched import (
    build_gt_targets, RESULTS_CSV, RESULTS_DIR,
)

targets = build_gt_targets()

with open(RESULTS_CSV, newline='') as f:
    results = list(csv.DictReader(f))
for r in results:
    r['attribution'] = float(r['attribution'])
    r['has_ruler_gt'] = r['has_ruler_gt'] == 'True'

if len(results) != len(targets):
    print(f"WARNING: {len(results)} rows in {RESULTS_CSV}, expected "
          f"{len(targets)} -- some scoring chunks may not have finished yet.\n")

print(f"{'='*70}\nGROUND-TRUTH VERIFICATION -- corrected ruler_present CAV\n{'='*70}")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, cls in zip(axes, ["MEL", "NV"]):
    ruler_vals = np.array([r['attribution'] for r in results
                            if r['class'] == cls and r['has_ruler_gt']])
    clean_vals = np.array([r['attribution'] for r in results
                            if r['class'] == cls and not r['has_ruler_gt']])

    print(f"\n{cls}:")
    print(f"  ruler (n={len(ruler_vals)}): mean={ruler_vals.mean():.5f}  "
          f"median={np.median(ruler_vals):.5f}  std={ruler_vals.std():.5f}")
    print(f"  clean (n={len(clean_vals)}): mean={clean_vals.mean():.5f}  "
          f"median={np.median(clean_vals):.5f}  std={clean_vals.std():.5f}")

    # Outlier-driven check on the ruler group specifically -- small n
    # (11 for NV) means the mean/median above could be dominated by one
    # or two images rather than reflecting a broad pattern.
    if len(ruler_vals) > 0 and ruler_vals.max() > 0:
        n_above_half_max = int(np.sum(ruler_vals > ruler_vals.max() / 2))
        top_share = ruler_vals.max() / ruler_vals.sum() if ruler_vals.sum() > 0 else float('nan')
        print(f"  ruler images above half of this group's max: "
              f"{n_above_half_max}/{len(ruler_vals)} -- "
              f"{'broad pattern' if n_above_half_max > len(ruler_vals) * 0.3 else 'looks outlier-driven'} "
              f"(top image is {top_share:.1%} of the group's attribution sum)")

    if len(ruler_vals) >= 2 and len(clean_vals) >= 2:
        u_stat, p_value = stats.mannwhitneyu(
            ruler_vals, clean_vals, alternative='two-sided'
        )
        direction = "higher" if ruler_vals.mean() > clean_vals.mean() else "NOT higher"
        print(f"  Mann-Whitney U={u_stat:.1f}, p={p_value:.4f} "
              f"-- ruler-group attribution is {direction} than clean-group "
              f"({'significant' if p_value < 0.05 else 'not significant'} at alpha=0.05)")
    else:
        print("  [SKIP] fewer than 2 images in one group -- can't run the test")

    ax.boxplot([clean_vals, ruler_vals], labels=['clean', 'ruler'], showmeans=True)
    for i, vals in enumerate([clean_vals, ruler_vals], start=1):
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, alpha=0.5, s=12, color='C0')
    ax.set_title(f"{cls} (n_clean={len(clean_vals)}, n_ruler={len(ruler_vals)})")
    ax.set_ylabel("post_relu attribution")

fig.suptitle("Corrected ruler_present CAV: attribution vs. ground-truth ruler presence")
fig.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "gt_verification_plot.png")
fig.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close('all')
print(f"\nPlot saved: {plot_path}")
print(f"All done. Results in: {RESULTS_DIR}")
