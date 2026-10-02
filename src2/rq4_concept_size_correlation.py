"""
rq4_concept_size_correlation.py

RQ4 sensitivity-to-concept-set-composition check (Chapter 5, Section 5.6.2 of
the thesis): Spearman correlation between each concept's post_relu mean
attribution (averaged across the 8 classification classes) and its
positive-example count, nominal (absent-only) negative-example count, and
negative-to-positive ratio, across the 10 concepts.

Also runs the stated robustness check: the same correlation using the
augmented negative pool CAV training actually draws from for the 6
regularity-paired concepts (nominal negative + sibling variant's positive
examples as hard negatives, Section 4.3.1), restricted to those 6 concepts.

Written to recompute this analysis from the corrected, post-emblem-clip-fix
attribution results (outputs2/vtcav_ttest_proper_ig_padonly_seed2_hardneg_randomsample/
ttest_results_proper_ig_randomsample.json) -- no committed script computing
this analysis could be found anywhere in git history (checked every branch),
so this replaces what was originally an uncommitted, one-off computation.

Positive/negative example counts are the fixed values reported in
Table 5.1 of the thesis (chapters/results.tex, tab:concept_stats).

Author: Shruti Kakkar
"""

import json
import os

from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TTEST_RESULTS = os.path.join(
    PROJECT_ROOT, "outputs2",
    "vtcav_ttest_proper_ig_padonly_seed2_hardneg_randomsample",
    "ttest_results_proper_ig_randomsample.json",
)

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# Table 5.1 (chapters/results.tex, tab:concept_stats)
POSITIVE = {
    "pigment_network_typical": 160, "pigment_network_atypical": 93,
    "streaks_regular": 39, "streaks_irregular": 101,
    "pigmentation": 160, "regression_structures": 96,
    "dots_and_globules_regular": 156, "dots_and_globules_irregular": 173,
    "blue_whitish_veil": 74, "vascular_structures": 66,
}
NOMINAL_NEGATIVE = {
    "pigment_network_typical": 160, "pigment_network_atypical": 160,
    "streaks_regular": 273, "streaks_irregular": 273,
    "pigmentation": 253, "regression_structures": 317,
    "dots_and_globules_regular": 84, "dots_and_globules_irregular": 84,
    "blue_whitish_veil": 339, "vascular_structures": 347,
}
# Augmented negative pool for the 6 regularity-paired concepts (Table 5.1 footnote)
AUGMENTED_NEGATIVE = {
    "pigment_network_typical": 253, "pigment_network_atypical": 320,
    "streaks_regular": 374, "streaks_irregular": 312,
    "dots_and_globules_regular": 257, "dots_and_globules_irregular": 240,
}

with open(TTEST_RESULTS) as f:
    results = json.load(f)

concepts = list(POSITIVE.keys())
attribution = {}
for c in concepts:
    per_class_means = [results[c]["post_relu"][cls]["mean"] for cls in CLASSES]
    attribution[c] = sum(per_class_means) / len(per_class_means)

positive = [POSITIVE[c] for c in concepts]
nominal_negative = [NOMINAL_NEGATIVE[c] for c in concepts]
ratio = [NOMINAL_NEGATIVE[c] / POSITIVE[c] for c in concepts]
attr = [attribution[c] for c in concepts]

print("Per-concept post_relu mean attribution (averaged across 8 classes):")
for c in concepts:
    print(f"  {c:32s} {attribution[c]:.6f}")
print()

rho_pos, p_pos = stats.spearmanr(attr, positive)
rho_neg, p_neg = stats.spearmanr(attr, nominal_negative)
rho_ratio, p_ratio = stats.spearmanr(attr, ratio)

print(f"attribution vs positive-example count:        rho={rho_pos:+.3f}, p={p_pos:.3f}")
print(f"attribution vs nominal negative-example count: rho={rho_neg:+.3f}, p={p_neg:.3f}")
print(f"attribution vs negative-to-positive ratio:     rho={rho_ratio:+.3f}, p={p_ratio:.3f}")
print()

six = [c for c in concepts if c in AUGMENTED_NEGATIVE]
attr6 = [attribution[c] for c in six]
aug6 = [AUGMENTED_NEGATIVE[c] for c in six]
rho6, p6 = stats.spearmanr(attr6, aug6)
print(f"[Robustness check, n=6 regularity-paired concepts]")
print(f"attribution vs augmented (hard-negative-inclusive) negative pool: rho={rho6:+.3f}, p={p6:.3f}")
