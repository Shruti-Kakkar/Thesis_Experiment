"""
rq3_negcontrol_effect_sizes.py

PURPOSE: the negative-control counterfactual table (tab:rq3_negcontrol,
Section 5.5.2): for each artifact and condition, the mean P(MEL) shift
over the 50 paired images, a one-sample t-test of the shifts against zero
(identical to the paired test the counterfactual scripts run), and the
paired effect size d_z = mean shift / SD of shifts.

Reads the counterfactual_results.csv files written by
ruler_bias_counterfactual_test.py, ruler_bias_counterfactual_negcontrol_test.py,
vignette_bias_counterfactual_test.py and
vignette_bias_counterfactual_negcontrol_test.py. No model compute.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import json
import os
import pandas as pd
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs2")

RUNS = {
    ("ruler_present", "real (tick pattern)"):            "ruler_bias_counterfactual_test_set",
    ("ruler_present", "control (plain bar)"):            "ruler_bias_counterfactual_negcontrol_test_set",
    ("vignette_present", "real (circular vignette)"):    "vignette_bias_counterfactual_test_set",
    ("vignette_present", "control (rectangular border)"): "vignette_bias_counterfactual_negcontrol_test_set",
}

OUT_PATH = os.path.join(OUTPUTS_DIR, "rq3_negcontrol_effect_sizes.json")

results = {}
print(f"{'Artifact':<18} {'Condition':<30} {'n':>3} {'mean':>8} {'t':>6} {'p':>10} {'d_z':>5}")
for (artifact, condition), results_dir in RUNS.items():
    shifts = pd.read_csv(os.path.join(OUTPUTS_DIR, results_dir, "counterfactual_results.csv"))["shift"]
    t_stat, p_value = stats.ttest_1samp(shifts, 0.0)
    d_z = shifts.mean() / shifts.std(ddof=1)
    results[f"{artifact} / {condition}"] = {
        "n": int(len(shifts)), "mean_shift": float(shifts.mean()),
        "t": float(t_stat), "df": int(len(shifts) - 1), "p": float(p_value), "d_z": float(d_z),
    }
    print(f"{artifact:<18} {condition:<30} {len(shifts):>3} {shifts.mean():>+8.4f} "
          f"{t_stat:>6.2f} {p_value:>10.3g} {d_z:>5.2f}")

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUT_PATH}")
