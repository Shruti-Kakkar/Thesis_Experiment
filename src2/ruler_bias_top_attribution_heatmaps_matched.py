"""
ruler_bias_top_attribution_heatmaps_matched.py
Local VTCAV heatmaps for the highest-attribution ground-truth ruler
images, per class -- visual triangulation on top of the quantitative
Mann-Whitney result in ruler_bias_ground_truth_finalize_matched.py. If
the corrected CAV's attribution genuinely tracks ruler presence (not
just noise), these hot zones should land on the physical ruler marks,
not somewhere else in the image.

Picks the top N_TOP images by attribution from each class's _ruler group
in gt_attributions.csv (already scored by
ruler_bias_ground_truth_verification_matched.py), then reruns
LocalVisualTCAV on each to get the spatial heatmap -- the CSV only has
the scalar attribution, not the saved heatmap, so this recomputes it
(same cached CAV, same cached activations, cheap, no retraining).

post_relu only, via ruler_bias_common_matched.make_local_tcav -- the
same layer/CAV the ranking attribution itself came from, so the heatmap
you see is for the exact number these images were ranked by.

Caveat worth keeping in the writeup: for NV specifically, only 11 images
are eligible at all, and the finalize script's outlier check found one
of them accounts for 63.3% of the group's attribution -- that image will
almost certainly be NV's #1 here, so the NV heatmaps are illustrating one
strong case, not a broad pattern.

Author: Shruti Kakkar
"""

import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ruler_bias_common_matched import make_local_tcav
from ruler_bias_ground_truth_verification_matched import RESULTS_CSV as GT_CSV

N_TOP = 3
OUT_DIR = os.path.join(os.path.dirname(GT_CSV), "top_attribution_heatmaps")
os.makedirs(OUT_DIR, exist_ok=True)

with open(GT_CSV, newline='') as f:
    rows = list(csv.DictReader(f))
for r in rows:
    r['attribution'] = float(r['attribution'])
    r['has_ruler_gt'] = r['has_ruler_gt'] == 'True'

for cls in ["MEL", "NV"]:
    ruler_rows = sorted(
        [r for r in rows if r['class'] == cls and r['has_ruler_gt']],
        key=lambda r: r['attribution'], reverse=True
    )
    top = ruler_rows[:N_TOP]
    print(f"\n{cls} top {len(top)} ground-truth ruler images by attribution:")
    for rank, r in enumerate(top, start=1):
        print(f"  #{rank}: {r['filename']}  attribution={r['attribution']:.5f}")

    for rank, r in enumerate(top, start=1):
        fname = r['filename']
        rel_path = os.path.join(cls, fname)
        image_label = os.path.splitext(fname)[0]

        local_tcav = make_local_tcav(cls, rel_path)
        local_tcav.predict()
        local_tcav.explain(cache_cav=True, cache_random=True, n_cav_runs=20)

        def _save_instead_of_show(rank=rank, r=r, image_label=image_label, cls=cls):
            fname_out = os.path.join(
                OUT_DIR, f"top{rank}_{cls}_{image_label}_ruler_present.png"
            )
            fig = plt.gcf()
            fig.text(
                0.5, 1.0,
                f"{image_label}  |  true class: {cls}  |  ground-truth: ruler  |  "
                f"rank #{rank}  |  attribution={r['attribution']:.5f}",
                ha='center', va='top', fontsize=12, color='black', fontweight='bold',
                bbox=dict(facecolor='yellow', alpha=0.9, edgecolor='black', pad=4),
            )
            plt.savefig(fname_out, dpi=150, bbox_inches='tight')
            plt.close('all')
            print(f"    Saved: {fname_out}")
        plt.show = _save_instead_of_show

        local_tcav.plot()

print(f"\nAll done. Heatmaps in: {OUT_DIR}")
