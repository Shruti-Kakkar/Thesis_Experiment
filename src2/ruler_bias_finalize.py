"""
ruler_bias_finalize.py
Second stage of the ruler-bias local screen: reads the attributions CSV
built by (possibly several chunked runs of) ruler_bias_local_screening.py,
prints per-class summary statistics, and regenerates full heatmaps for
the top N MEL outliers by ruler_present attribution.

Run this once after all scoring chunks have completed.

Author: Shruti Kakkar
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ruler_bias_common import (
    build_targets, make_local_tcav, RESULTS_CSV, RESULTS_DIR, N_TOP_TO_PLOT,
)

targets = build_targets()

with open(RESULTS_CSV, newline='') as f:
    results = list(csv.DictReader(f))
for r in results:
    r['attribution'] = float(r['attribution'])

if len(results) != len(targets):
    print(f"WARNING: {len(results)} rows in {RESULTS_CSV}, expected "
          f"{len(targets)} -- some scoring chunks may not have finished yet.\n")

# ─────────────────────────────────────────────
# SUMMARY STATISTICS
# ─────────────────────────────────────────────
print(f"{'='*60}\nSUMMARY\n{'='*60}")
for cls in ["MEL", "NV"]:
    vals = np.array([r['attribution'] for r in results if r['class'] == cls])
    print(f"\n{cls} (n={len(vals)}):")
    print(f"  mean={vals.mean():.5f}  median={np.median(vals):.5f}  "
          f"std={vals.std():.5f}  max={vals.max():.5f}  min={vals.min():.5f}")
    n_above_half_max = np.sum(vals > vals.max() / 2)
    print(f"  images above half of this class's max: {n_above_half_max}/{len(vals)} "
          f"-- {'broad pattern' if n_above_half_max > len(vals) * 0.3 else 'looks outlier-driven'}")

# ─────────────────────────────────────────────
# AUTO-FLAG + FULL HEATMAP FOR TOP N MEL OUTLIERS
# ─────────────────────────────────────────────
mel_results = sorted(
    [r for r in results if r['class'] == 'MEL'],
    key=lambda r: r['attribution'], reverse=True
)
print(f"\nTop {N_TOP_TO_PLOT} MEL images by ruler_present attribution "
      f"(generating full heatmaps for these):")

for r in mel_results[:N_TOP_TO_PLOT]:
    print(f"  {r['filename']}: {r['attribution']:.5f}")
    image_label = os.path.splitext(r['filename'])[0]
    rel_path = os.path.join('MEL', r['filename'])

    local_tcav = make_local_tcav('MEL', rel_path)
    local_tcav.predict()
    local_tcav.explain(cache_cav=True, cache_random=True, n_cav_runs=20)

    def _save_instead_of_show(r=r, image_label=image_label):
        fname_out = os.path.join(
            RESULTS_DIR, f"top_outlier_{image_label}_MEL_ruler_present.png"
        )
        fig = plt.gcf()
        fig.text(
            0.5, 1.0, f"{image_label}  |  true class: MEL  |  "
            f"attribution={r['attribution']:.5f}",
            ha='center', va='top', fontsize=12, color='black', fontweight='bold',
            bbox=dict(facecolor='orange', alpha=0.9, edgecolor='black', pad=4),
        )
        plt.savefig(fname_out, dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f"    Saved: {fname_out}")
    plt.show = _save_instead_of_show

    local_tcav.plot()

print(f"\nAll done. Results in: {RESULTS_DIR}")
