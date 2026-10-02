"""
rq1_class_concentration_index.py

RQ1's secondary diagnostic (Chapter 4, Equation 4.7 / Chapter 5, Section
5.3 of the thesis): the class-concentration index, the coefficient of
variation across each concept's 8 per-class mean post_relu attribution
scores,

    CV_{l,C} = std_k(Attr_bar_{l,C,k}) / mean_k(Attr_bar_{l,C,k}), k = 1..8

computed per concept per layer, undefined whenever a concept's mean
attribution across all 8 classes is exactly zero, then averaged over the
concepts where it is defined at each layer.

No script computing this could be found anywhere in git history (checked
every branch, every commit, dangling commits, reflog) or among the
project's orphaned compiled bytecode -- this replaces what was originally
an uncommitted, one-off computation, using the same authoritative,
post-emblem-clip-fix, post-sampling-fix data source every current table
in the thesis draws from.

Uses numpy's default (population) standard deviation, matching a plain
`.std()` call with no ddof argument -- the natural default absent any
documented reason to prefer the sample (n-1) version.

Author: Shruti Kakkar
"""

import json
import os

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TTEST_RESULTS = os.path.join(
    PROJECT_ROOT, "outputs2",
    "vtcav_ttest_proper_ig_padonly_seed2_hardneg_randomsample",
    "ttest_results_proper_ig_randomsample.json",
)

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
LAYERS = ["conv5_block1_out", "conv5_block2_out", "conv5_block3_out", "post_relu"]

with open(TTEST_RESULTS) as f:
    results = json.load(f)

concepts = list(results.keys())

for layer in LAYERS:
    cvs = []
    undefined = []
    for c in concepts:
        means = np.array([results[c][layer][cls]["mean"] for cls in CLASSES])
        mean_of_means = means.mean()
        if mean_of_means == 0:
            undefined.append(c)
            continue
        cv = means.std() / mean_of_means
        cvs.append(cv)
    if cvs:
        print(
            f"{layer:20s} mean CV over {len(cvs)}/{len(concepts)} defined "
            f"concepts = {np.mean(cvs):.3f}  "
            f"(undefined: {undefined if undefined else 'none'})"
        )
    else:
        print(f"{layer:20s} undefined for every concept")
