"""
BWV Post-hoc Add-on: Bootstrap CI + Misclassification-Association Test
=========================================================================
ONE-OFF script for the already-completed blue_whitish_veil calibration run.
Uses ONLY data already on disk (saved attributions CSV + meta.csv) --
does NOT touch the model, does NOT recompute any attribution. Safe to
delete after running once; the general concept_calibration.py script now
does both of these automatically for every future concept.

Adds two things to your existing BWV result:
  1. A 95% bootstrap confidence interval on the Youden's J threshold
     (how stable is 0.038, if we'd drawn a slightly different sample?)
  2. A Fisher's exact test on whether misclassification rate differs
     between BWV-present and BWV-absent images (did the
     RESTRICT_TO_CORRECT_PREDICTIONS filter quietly bias the 56/57 split?)
"""

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from sklearn.metrics import roc_curve

PROJECT_ROOT = "/home/student/s/skakkar/scratch/dev-uos/projects/VTCAV_Dermatology"
META_PATH = f"{PROJECT_ROOT}/datasets/release_v0/meta/meta.csv"
TRAIN_IDX_PATH = f"{PROJECT_ROOT}/datasets/release_v0/meta/train_indexes.csv"
ATTRIBUTIONS_CSV = f"{PROJECT_ROOT}/outputs2/bwv_calibration/bwv_calibration_attributions.csv"

N_BOOT = 2000
SEED = 42


# ---------------------------------------------------------------------------
# Recompute the original held-out group sizes (80 present / 82 absent),
# to derive misclassified counts by subtracting the correct-prediction
# counts already in the saved CSV. No model, no re-inference.
# ---------------------------------------------------------------------------
def get_original_heldout_counts():
    meta = pd.read_csv(META_PATH)
    train_idx = set(pd.read_csv(TRAIN_IDX_PATH)["indexes"])
    heldout = meta.loc[~meta.index.isin(train_idx)].copy()
    mel = heldout[heldout["diagnosis"].str.contains("melanoma", case=False)].copy()
    mel = mel[mel["blue_whitish_veil"].isin(["present", "absent"])].copy()
    n_present = (mel["blue_whitish_veil"] == "present").sum()
    n_absent = (mel["blue_whitish_veil"] == "absent").sum()
    return int(n_present), int(n_absent)


# ---------------------------------------------------------------------------
# 1. Bootstrap CI on the Youden's J-optimal threshold
# ---------------------------------------------------------------------------
def bootstrap_threshold_ci(present, absent, n_boot=N_BOOT, ci=0.95, seed=SEED):
    rng = np.random.default_rng(seed)
    n1, n2 = len(present), len(absent)
    thresholds = []

    for _ in range(n_boot):
        p_samp = rng.choice(present, size=n1, replace=True)
        a_samp = rng.choice(absent, size=n2, replace=True)
        y_true = np.concatenate([np.ones(n1), np.zeros(n2)])
        y_score = np.concatenate([p_samp, a_samp])
        fpr, tpr, thr = roc_curve(y_true, y_score)
        j = tpr - fpr
        thresholds.append(thr[np.argmax(j)])

    thresholds = np.array(thresholds)
    lower = np.percentile(thresholds, (1 - ci) / 2 * 100)
    upper = np.percentile(thresholds, (1 + ci) / 2 * 100)
    median = np.percentile(thresholds, 50)
    return median, lower, upper


# ---------------------------------------------------------------------------
# 2. Fisher's exact test: does misclassification rate depend on BWV label?
# ---------------------------------------------------------------------------
def misclassification_association_test(n_present_total, n_absent_total,
                                          n_present_correct, n_absent_correct):
    n_present_wrong = n_present_total - n_present_correct
    n_absent_wrong = n_absent_total - n_absent_correct

    # 2x2 table: rows = [present, absent], cols = [correct, misclassified]
    table = [[n_present_correct, n_present_wrong],
             [n_absent_correct, n_absent_wrong]]

    odds_ratio, p_value = fisher_exact(table)
    return table, odds_ratio, p_value


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(ATTRIBUTIONS_CSV)
    present = df.loc[df["bwv_label"] == 1, "attribution"].values
    absent = df.loc[df["bwv_label"] == 0, "attribution"].values
    n_present_correct, n_absent_correct = len(present), len(absent)

    print(f"Loaded {len(df)} correctly-classified images "
          f"(present={n_present_correct}, absent={n_absent_correct})")

    # --- 1. Bootstrap CI ---
    median_thr, ci_low, ci_high = bootstrap_threshold_ci(present, absent)
    print(f"\n[1] Bootstrap 95% CI on Youden's J threshold ({N_BOOT} resamples):")
    print(f"    median={median_thr:.4f}, 95% CI=[{ci_low:.4f}, {ci_high:.4f}]")
    print(f"    (point estimate from original run was 0.0382 -- should sit "
          f"near the median above)")

    # --- 2. Misclassification association test ---
    n_present_total, n_absent_total = get_original_heldout_counts()
    print(f"\nOriginal held-out counts: present={n_present_total}, absent={n_absent_total}")

    table, odds_ratio, p_value = misclassification_association_test(
        n_present_total, n_absent_total, n_present_correct, n_absent_correct
    )
    print(f"\n[2] Fisher's exact test -- does misclassification rate differ "
          f"by BWV label?")
    print(f"    2x2 table [rows=present/absent, cols=correct/misclassified]:")
    print(f"      present: correct={table[0][0]}, misclassified={table[0][1]}")
    print(f"      absent:  correct={table[1][0]}, misclassified={table[1][1]}")
    print(f"    odds ratio={odds_ratio:.3f}, p={p_value:.4f}")
    if p_value < 0.05:
        print(f"    -> SIGNIFICANT: misclassification rate differs by BWV "
              f"label -- the 56/57 split may be subtly biased. Worth a "
              f"caveat in the thesis.")
    else:
        print(f"    -> NOT significant: no evidence misclassification "
              f"depends on BWV label -- the present/absent filtering "
              f"appears unbiased.")

    # --- Save a short addendum file next to the original summary ---
    addendum_path = f"{PROJECT_ROOT}/outputs2/bwv_calibration/bwv_calibration_addendum.txt"
    with open(addendum_path, "w") as f:
        f.write(f"""BWV Calibration Addendum (bootstrap CI + misclassification check)
=====================================================================
Bootstrap 95% CI on Youden's J threshold ({N_BOOT} resamples):
  median={median_thr:.4f}, 95% CI=[{ci_low:.4f}, {ci_high:.4f}]
  (original point estimate: 0.0382)

Misclassification-association test (Fisher's exact):
  present: correct={table[0][0]}, misclassified={table[0][1]}
  absent:  correct={table[1][0]}, misclassified={table[1][1]}
  odds ratio={odds_ratio:.3f}, p={p_value:.4f}
""")
    print(f"\nSaved addendum to {addendum_path}")