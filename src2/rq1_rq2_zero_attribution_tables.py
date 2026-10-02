"""
rq1_rq2_zero_attribution_tables.py

PURPOSE: the RQ1/RQ2 summary numbers derived from the random-sample
proper-IG significance test:
  - Table 5.3 (tab:rq1_zero_cells): zero-attribution cells per layer,
    split by cause -- emblem-check failure (concept-level, zeroes all 8
    classes) vs. class-relative weight w_k(x)=0 (the remaining zeros, for
    concepts whose emblem check passes) -- plus the non-zero range at
    post_relu.
  - Table 5.4 (tab:rq2_sig_vs_zero): cells marked significant (p<0.05)
    per layer.
  - Section 5.4 text: direction of each significant call at
    conv5_block1_out (real mean <= random-CAV mean counts as inverse),
    and the regression_structures / streaks_regular comparison.

Reads ttest_results_proper_ig_randomsample.json (written by
VisualTCAV_derma_ttest_padonly_seed2_hardneg_randomsample.py) and each
concept's cached emblem (eps_plus, eps_minus) from the hard-negative
Global run's CAV cache (VisualTCAV_derma_global_padonly_seed2.py).
No model compute.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import json
import os
import numpy as np
from joblib import load

PROJECT_ROOT = os.path.expanduser(
    "~/scratch/dev-uos/projects/VTCAV_Dermatology"
)
MODEL_TAG = "padonly_seed2_hardneg"
CACHE_DIR = os.path.join(PROJECT_ROOT, "outputs2", f"vtcav_{MODEL_TAG}", "cache", "resnet50v2")
RESULTS_PATH = os.path.join(
    PROJECT_ROOT, "outputs2", f"vtcav_ttest_proper_ig_{MODEL_TAG}_randomsample",
    "ttest_results_proper_ig_randomsample.json",
)
OUT_PATH = os.path.join(PROJECT_ROOT, "outputs2", "rq1_rq2_zero_attribution_tables.json")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))  # joblib needs VisualTCAV's classes

LAYERS = ["conv5_block1_out", "conv5_block2_out", "conv5_block3_out", "post_relu"]
MAX_EXAMPLES = 200
N_CAV_RUNS = 20

# Hard-negative concepts: CAV trained with the sibling variant's positives
# as extra negatives (same mapping as VisualTCAV_derma_global_padonly_seed2.py)
EXTRA_NEGATIVES = {
    "pigment_network_typical":     "pigment_network_atypical",
    "pigment_network_atypical":    "pigment_network_typical",
    "streaks_regular":             "streaks_irregular",
    "streaks_irregular":           "streaks_regular",
    "dots_and_globules_regular":   "dots_and_globules_irregular",
    "dots_and_globules_irregular": "dots_and_globules_regular",
}


def emblem_passes(concept, layer_name):
    extra = f"_plus_{EXTRA_NEGATIVES[concept]}-positive" if concept in EXTRA_NEGATIVES else ""
    path = os.path.join(
        CACHE_DIR,
        f"cav_{concept}_positive{extra}_{MAX_EXAMPLES}_neg_{N_CAV_RUNS}runs_{layer_name}.joblib",
    )
    emblem = np.asarray(load(path).cav.concept_emblem)
    return float(emblem[0]) > float(emblem[1])  # eps_plus > eps_minus


with open(RESULTS_PATH) as f:
    results = json.load(f)
concepts = list(results)
classes = list(results[concepts[0]][LAYERS[0]])
n_cells = len(concepts) * len(classes)

out = {}
print(f"{'Layer':<18} {'zero':>7} {'emblem':>7} {'w_k=0':>7} {'sig':>7} {'inverse':>8} {'correct':>8}")
for layer_name in LAYERS:
    cells = [(c, k, results[c][layer_name][k]) for c in concepts for k in classes]
    failing = [c for c in concepts if not emblem_passes(c, layer_name)]
    zero = [(c, k) for c, k, v in cells if v["zero_attribution"]]
    via_emblem = sum(1 for c, k in zero if c in failing)
    significant = [(c, k, v) for c, k, v in cells if v["significant"]]
    inverse = [(c, k) for c, k, v in significant if v["mean"] <= v["rand_mean"]]
    nonzero = [v["mean"] for c, k, v in cells if not v["zero_attribution"]]

    out[layer_name] = {
        "n_cells": n_cells,
        "zero_cells": len(zero),
        "zero_via_emblem_check": via_emblem,
        "zero_via_class_weight": len(zero) - via_emblem,
        "emblem_check_fails": failing,
        "significant_cells": len(significant),
        "significant_inverse": len(inverse),
        "significant_correct": len(significant) - len(inverse),
        "nonzero_range": [min(nonzero), max(nonzero)] if nonzero else None,
    }
    print(f"{layer_name:<18} {len(zero):>3}/{n_cells} {via_emblem:>3}/{n_cells} "
          f"{len(zero) - via_emblem:>3}/{n_cells} {len(significant):>3}/{n_cells} "
          f"{len(inverse):>8} {len(significant) - len(inverse):>8}")
    print(f"{'':<18} emblem check fails: {failing or 'none'}"
          + (f"; non-zero range {min(nonzero):.4g}..{max(nonzero):.4g}" if nonzero else ""))

# Section 5.4: worst vs. best concept at conv5_block1_out
layer_name = "conv5_block1_out"
out["rq2_concept_comparison"] = {}
print(f"\nSignificant calls at {layer_name}:")
for concept in ["regression_structures", "streaks_regular"]:
    calls = [
        {"class": k, "mean": v["mean"], "rand_mean": v["rand_mean"],
         "inverse": v["mean"] <= v["rand_mean"]}
        for k, v in results[concept][layer_name].items() if v["significant"]
    ]
    out["rq2_concept_comparison"][concept] = calls
    print(f"  {concept}: {len(calls)}/{len(classes)} significant -- "
          + ", ".join(f"{c['class']} mean={c['mean']:.4g} ({'inverse' if c['inverse'] else 'correct'})"
                      for c in calls))

with open(OUT_PATH, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {OUT_PATH}")
