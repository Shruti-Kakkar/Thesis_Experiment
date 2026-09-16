"""
Concept Attribution Calibration (general, reusable across concepts)
========================================================================
Determines an empirically-grounded VTCAV attribution threshold for a given
concept, by comparing per-image attribution scores between ground-truth
concept-present vs concept-absent images of a target class.

USAGE: set the six variables under "PER-RUN CONFIG" below for whichever
concept you're calibrating, then run. No separate script needed per concept.

Design:
  - Uses ONLY held-out (non-train-split) Derm7pt images, to avoid circularity
    (the CAV was fit using train-split labels; evaluating on those same
    images would trivially show separation).
  - Computes a per-image scalar attribution value using LocalVisualTCAV's
    own attribution output -- the same per-image quantity Global VTCAV
    averages across images to produce its class-level score.
  - Statistics, in order of what each answers:
      1. Mann-Whitney U           -- is there a real difference at all?
      2. Rank-biserial correlation -- how big is the difference?
      3. ROC / AUC + Youden's J    -- PRIMARY threshold (most rigorous)
      4. Bootstrap 95% CI on that threshold -- how stable is it?
      5. KDE crossing point        -- secondary/eyeball cross-check only
      6. Fisher's exact test       -- did the correct-prediction filter
                                       bias the present/absent split?

NOTE on multiple comparisons: once you've run this for ALL concepts, apply
a Benjamini-Hochberg correction across the resulting p-values before
reporting them as a set -- not needed for a single concept in isolation.
(ask for the correction snippet when you have the full set of results)

Validated end-to-end on voxel for blue_whitish_veil (2026-08-13):
  113/162 images survived correct-prediction filter (56 present/57 absent)
  U=2706.00, p=9.37e-11, rank-biserial=+0.695, AUC=0.848,
  Youden threshold=0.038, KDE crossing=0.041
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe for running on a remote server
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde, fisher_exact
from sklearn.metrics import roc_curve, auc

# ---------------------------------------------------------------------------
# PROJECT-WIDE CONFIG (should not need to change between concept runs)
# ---------------------------------------------------------------------------
PROJECT_ROOT = "/home/student/s/skakkar/scratch/dev-uos/projects/VTCAV_Dermatology"

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from VisualTCAV import Model, LocalVisualTCAV, preprocess_resnet_v2

META_PATH = f"{PROJECT_ROOT}/datasets/release_v0/meta/meta.csv"
TRAIN_IDX_PATH = f"{PROJECT_ROOT}/datasets/release_v0/meta/train_indexes.csv"
DERM7PT_IMAGE_ROOT = f"{PROJECT_ROOT}/datasets/release_v0/images"

VISUAL_TCAV_DIR = f"{PROJECT_ROOT}/outputs2/vtcav_padonly_seed2_hardneg"
MODELS_DIR = None
CACHE_DIR = None
CONCEPT_IMAGES_DIR = f"{PROJECT_ROOT}/concept_images"

MODEL_NAME = "resnet50v2"
GRAPH_PATH_FILENAME = "resnet50v2_isic2019_final.keras"
LABEL_PATH_FILENAME = "isic2019_classes.txt"

MAX_EXAMPLES = 200   # must match canonical CAV-training hyperparameters
N_CAV_RUNS = 20       # must match canonical CAV-training hyperparameters
M_STEPS = 50
RESIZE_MODE = "pad"

RESTRICT_TO_CORRECT_PREDICTIONS = True
N_BOOT = 2000          # bootstrap resamples for threshold CI
BOOT_SEED = 42

# ---------------------------------------------------------------------------
# PER-RUN CONFIG -- change these six for each concept
# ---------------------------------------------------------------------------
TARGET_CLASS = "MEL"                    # e.g. "MEL", "NV" -- must match isic2019_classes.txt exactly

CONCEPT = "pigmentation"           # concept folder root -- matches concept_images/{CONCEPT}/{positive,negative}
LAYER = "post_relu"

CONCEPT_META_COLUMN = "pigmentation"  # meta.csv column to read ground-truth label from
CONCEPT_POSITIVE_VALUES = ["diffuse irregular", "localized irregular", "diffuse regular", "localized regular"]      # values in that column counted as "concept present"
CONCEPT_NEGATIVE_VALUES = ["absent"]       # values counted as "concept absent"
# Examples for other concepts (uncomment/adjust as needed):
#   pigment_network_typical:
#     CONCEPT_META_COLUMN = "pigment_network"; CONCEPT_POSITIVE_VALUES = ["typical"]; CONCEPT_NEGATIVE_VALUES = ["absent"]
#   pigment_network_atypical:
#     CONCEPT_META_COLUMN = "pigment_network"; CONCEPT_POSITIVE_VALUES = ["atypical"]; CONCEPT_NEGATIVE_VALUES = ["absent"]
#   streaks_irregular:
#     CONCEPT_META_COLUMN = "streaks"; CONCEPT_POSITIVE_VALUES = ["irregular"]; CONCEPT_NEGATIVE_VALUES = ["absent"]
#   dots_and_globules_regular:
#     CONCEPT_META_COLUMN = "dots_and_globules"; CONCEPT_POSITIVE_VALUES = ["regular"]; CONCEPT_NEGATIVE_VALUES = ["absent"]

# Hard-negative pairing -- REQUIRED for pigment_network_*, streaks_*, and
# dots_and_globules_* (check your concept-definition table for exact
# sources); leave {} for concepts with simple absent-only negatives
# (blue_whitish_veil, regression_structures, pigmentation, vascular_structures).
EXTRA_NEGATIVE_CONCEPTS = {}

OUTPUT_DIR = f"{PROJECT_ROOT}/outputs2/{CONCEPT}_calibration"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# STEP 1 -- Load metadata, isolate held-out target-class images, label
# ---------------------------------------------------------------------------
def load_heldout_images():
    meta = pd.read_csv(META_PATH)
    train_idx = set(pd.read_csv(TRAIN_IDX_PATH)["indexes"])

    heldout = meta.loc[~meta.index.isin(train_idx)].copy()

    # TARGET_CLASS -> diagnosis substring match. Extend this mapping if you
    # calibrate against classes other than MEL (diagnosis strings in
    # meta.csv are the raw Derm7pt labels, e.g. "clark nevus", "basal cell
    # carcinoma" -- not the abbreviated ISIC class names).
    diag_substring = {"MEL": "melanoma", "NV": "nevus", "BCC": "basal cell",
                       "SK": "seborrheic"}.get(TARGET_CLASS)
    if diag_substring is None:
        raise ValueError(f"No diagnosis substring mapping for TARGET_CLASS={TARGET_CLASS}; "
                          f"add one to the diag_substring dict above.")

    sub = heldout[heldout["diagnosis"].str.contains(diag_substring, case=False)].copy()
    valid_values = CONCEPT_POSITIVE_VALUES + CONCEPT_NEGATIVE_VALUES
    sub = sub[sub[CONCEPT_META_COLUMN].isin(valid_values)].copy()
    sub["concept_label"] = sub[CONCEPT_META_COLUMN].isin(CONCEPT_POSITIVE_VALUES).astype(int)

    n_pos, n_neg = int(sub["concept_label"].sum()), int((sub["concept_label"] == 0).sum())
    print(f"[Step 1] Held-out {TARGET_CLASS} images with valid {CONCEPT} label: {len(sub)}")
    print(f"          present={n_pos}, absent={n_neg}")
    return sub


# ---------------------------------------------------------------------------
# STEP 2 -- Build a fresh Model per image and run Local VTCAV
# ---------------------------------------------------------------------------
def build_model():
    return Model(
        model_name=MODEL_NAME,
        graph_path_filename=GRAPH_PATH_FILENAME,
        label_path_filename=LABEL_PATH_FILENAME,
        preprocessing_function=preprocess_resnet_v2,
        binary_classification=False,
        max_examples=MAX_EXAMPLES,
        resize_mode=RESIZE_MODE,
    )


def compute_attribution_for_image(derm_relpath):
    """
    Returns (attribution: float or None, predicted_class: str).
    attribution is None when RESTRICT_TO_CORRECT_PREDICTIONS is True and
    the model's top-1 prediction isn't TARGET_CLASS -- explain() (CAV
    lookup + integrated gradients) is skipped in that case.
    """
    model = build_model()

    local = LocalVisualTCAV(
        test_image_filename=derm_relpath,
        target_class=TARGET_CLASS,
        m_steps=M_STEPS,
        model=model,
        visual_tcav_dir=VISUAL_TCAV_DIR,
        models_dir=MODELS_DIR,
        cache_dir=CACHE_DIR,
        concept_images_dir=CONCEPT_IMAGES_DIR,
        test_images_dir=DERM7PT_IMAGE_ROOT,
        extra_negative_concepts=EXTRA_NEGATIVE_CONCEPTS,
    )

    local.setConcepts([f"{CONCEPT}/positive"])
    local.setLayers([LAYER])

    local.predict()
    predicted_class = local.predictions[0][0].class_name

    if RESTRICT_TO_CORRECT_PREDICTIONS and predicted_class != TARGET_CLASS:
        return None, predicted_class

    local.explain(cache_cav=True, n_cav_runs=N_CAV_RUNS)
    attribution = float(local.computations[LAYER][f"{CONCEPT}/positive"].attributions[0])
    return attribution, predicted_class


# ---------------------------------------------------------------------------
# STEP 3 -- Run attribution computation across the held-out set
# ---------------------------------------------------------------------------
def run_attribution_pass(img_df):
    records = []
    skipped_wrong_pred_pos = 0   # concept-present images that got misclassified
    skipped_wrong_pred_neg = 0   # concept-absent images that got misclassified
    skipped_error = 0

    for i, (idx, row) in enumerate(img_df.iterrows()):
        try:
            attribution, predicted_class = compute_attribution_for_image(row["derm"])
        except Exception as e:
            print(f"  [WARN] failed on {row['derm']}: {e}")
            skipped_error += 1
            continue

        if attribution is None:
            if row["concept_label"] == 1:
                skipped_wrong_pred_pos += 1
            else:
                skipped_wrong_pred_neg += 1
            continue

        records.append({
            "case_num": row["case_num"],
            "derm": row["derm"],
            "concept_label": row["concept_label"],
            "attribution": attribution,
            "predicted_class": predicted_class,
        })

        if (i + 1) % 10 == 0:
            print(f"  processed {i + 1}/{len(img_df)}")

    print(f"[Step 3] Computed attribution for {len(records)} images "
          f"({skipped_wrong_pred_pos} present + {skipped_wrong_pred_neg} absent "
          f"skipped as misclassified, {skipped_error} skipped on error)")

    total_pos = int(img_df["concept_label"].sum())
    total_neg = int((img_df["concept_label"] == 0).sum())
    return pd.DataFrame(records), total_pos, total_neg


# ---------------------------------------------------------------------------
# STEP 4 -- Statistics
# ---------------------------------------------------------------------------
def bootstrap_threshold_ci(present, absent, n_boot=N_BOOT, ci=0.95, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    n1, n2 = len(present), len(absent)
    thresholds = []
    for _ in range(n_boot):
        p_samp = rng.choice(present, size=n1, replace=True)
        a_samp = rng.choice(absent, size=n2, replace=True)
        y_true = np.concatenate([np.ones(n1), np.zeros(n2)])
        y_score = np.concatenate([p_samp, a_samp])
        fpr, tpr, thr = roc_curve(y_true, y_score)
        thresholds.append(thr[np.argmax(tpr - fpr)])
    thresholds = np.array(thresholds)
    lower = np.percentile(thresholds, (1 - ci) / 2 * 100)
    upper = np.percentile(thresholds, (1 + ci) / 2 * 100)
    median = np.percentile(thresholds, 50)
    return median, lower, upper


def analyze(results_df, total_pos, total_neg):
    present = results_df.loc[results_df["concept_label"] == 1, "attribution"].values
    absent = results_df.loc[results_df["concept_label"] == 0, "attribution"].values
    n1, n2 = len(present), len(absent)

    print(f"\n[Step 4] Present: n={n1}, mean={present.mean():.4f}, "
          f"median={np.median(present):.4f}, std={present.std():.4f}")
    print(f"         Absent:  n={n2}, mean={absent.mean():.4f}, "
          f"median={np.median(absent):.4f}, std={absent.std():.4f}")

    # 1. Mann-Whitney U (non-parametric significance)
    u_stat, p_value = stats.mannwhitneyu(present, absent, alternative="greater")

    # 2. Rank-biserial correlation (effect size). Identity: 2*AUC - 1.
    rank_biserial = (2 * u_stat) / (n1 * n2) - 1
    print(f"\nMann-Whitney U: U={u_stat:.2f}, p={p_value:.6g}")
    print(f"Rank-biserial correlation (effect size): {rank_biserial:.3f}")

    # 3. ROC / AUC / Youden's J -- PRIMARY threshold
    y_true = results_df["concept_label"].values
    y_score = results_df["attribution"].values
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    best_threshold = thresholds[best_idx]
    best_sens, best_spec = tpr[best_idx], 1 - fpr[best_idx]
    print(f"\nROC AUC: {roc_auc:.3f}")
    print(f"Youden's J optimal threshold (PRIMARY): {best_threshold:.4f}")
    print(f"  Sensitivity: {best_sens:.3f}  Specificity: {best_spec:.3f}")

    # 4. Bootstrap 95% CI on that threshold
    boot_median, boot_low, boot_high = bootstrap_threshold_ci(present, absent)
    print(f"\nBootstrap 95% CI on threshold ({N_BOOT} resamples): "
          f"median={boot_median:.4f}, CI=[{boot_low:.4f}, {boot_high:.4f}]")

    # 5. KDE crossing point -- SECONDARY cross-check only
    x_min, x_max = min(present.min(), absent.min()), max(present.max(), absent.max())
    x_grid = np.linspace(x_min, x_max, 1000)
    kde_present, kde_absent = gaussian_kde(present), gaussian_kde(absent)
    d_present, d_absent = kde_present(x_grid), kde_absent(x_grid)
    diff = d_present - d_absent
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    crossing_points = x_grid[sign_changes]
    print(f"KDE crossing point(s) (secondary check): {crossing_points}")

    # 6. Fisher's exact test -- did the correct-prediction filter bias the split?
    n1_wrong, n2_wrong = total_pos - n1, total_neg - n2
    table = [[n1, n1_wrong], [n2, n2_wrong]]
    odds_ratio, fisher_p = fisher_exact(table)
    print(f"\nFisher's exact test (misclassification vs concept label):")
    print(f"  present: correct={n1}, misclassified={n1_wrong}")
    print(f"  absent:  correct={n2}, misclassified={n2_wrong}")
    print(f"  odds ratio={odds_ratio:.3f}, p={fisher_p:.4f}"
          + (" -- SIGNIFICANT: filter may have biased the split" if fisher_p < 0.05
             else " -- not significant: filter appears unbiased"))

    return {
        "present": present, "absent": absent, "n1": n1, "n2": n2,
        "u_stat": u_stat, "p_value": p_value, "rank_biserial": rank_biserial,
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
        "best_idx": best_idx, "best_threshold": best_threshold,
        "best_sens": best_sens, "best_spec": best_spec,
        "boot_median": boot_median, "boot_low": boot_low, "boot_high": boot_high,
        "x_grid": x_grid, "d_present": d_present, "d_absent": d_absent,
        "crossing_points": crossing_points,
        "fisher_table": table, "fisher_odds_ratio": odds_ratio, "fisher_p": fisher_p,
    }


# ---------------------------------------------------------------------------
# STEP 5 -- Plot + save summary
# ---------------------------------------------------------------------------
def plot_and_save(stats_out, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(stats_out["absent"], bins=20, alpha=0.5, density=True,
            label=f'{CONCEPT} Absent (n={stats_out["n2"]})', color="steelblue")
    ax.hist(stats_out["present"], bins=20, alpha=0.5, density=True,
            label=f'{CONCEPT} Present (n={stats_out["n1"]})', color="firebrick")
    ax.plot(stats_out["x_grid"], stats_out["d_absent"], color="steelblue", lw=2)
    ax.plot(stats_out["x_grid"], stats_out["d_present"], color="firebrick", lw=2)
    ax.axvspan(stats_out["boot_low"], stats_out["boot_high"], color="gray", alpha=0.15,
               label=f'95% CI [{stats_out["boot_low"]:.3f}, {stats_out["boot_high"]:.3f}]')
    for cp in stats_out["crossing_points"]:
        ax.axvline(cp, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(stats_out["best_threshold"], color="black", lw=2,
               label=f'Youden threshold (primary)={stats_out["best_threshold"]:.4f}')
    ax.set_xlabel(f"VTCAV Attribution ({CONCEPT})")
    ax.set_ylabel("Density")
    ax.set_title(f"Attribution Distribution: {CONCEPT} Present vs Absent\n({TARGET_CLASS}, held-out)")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(stats_out["fpr"], stats_out["tpr"], color="darkorange", lw=2,
             label=f'ROC (AUC={stats_out["roc_auc"]:.3f})')
    ax2.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax2.scatter(stats_out["fpr"][stats_out["best_idx"]], stats_out["tpr"][stats_out["best_idx"]],
                color="black", zorder=5, label="Youden J optimal")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title(f"ROC: Attribution as {CONCEPT}-presence classifier")
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"{CONCEPT}_calibration_plot.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\n[Step 5] Saved plot to {plot_path}")

    summary = f"""
{CONCEPT} Attribution Calibration Summary
=====================================
Sample: {stats_out['n1'] + stats_out['n2']} held-out {TARGET_CLASS} images (never used in CAV training)
  {CONCEPT} present: {stats_out['n1']}
  {CONCEPT} absent:  {stats_out['n2']}

Attribution stats:
  Present: mean={stats_out['present'].mean():.4f}, median={np.median(stats_out['present']):.4f}, std={stats_out['present'].std():.4f}
  Absent:  mean={stats_out['absent'].mean():.4f}, median={np.median(stats_out['absent']):.4f}, std={stats_out['absent'].std():.4f}

Mann-Whitney U test (present > absent): U={stats_out['u_stat']:.2f}, p={stats_out['p_value']:.6g}
Rank-biserial correlation (effect size): {stats_out['rank_biserial']:.3f}

ROC AUC: {stats_out['roc_auc']:.3f}
Youden's J optimal threshold (PRIMARY): {stats_out['best_threshold']:.4f}
  Sensitivity: {stats_out['best_sens']:.3f}
  Specificity: {stats_out['best_spec']:.3f}
Bootstrap 95% CI on threshold: [{stats_out['boot_low']:.4f}, {stats_out['boot_high']:.4f}] (median {stats_out['boot_median']:.4f})

KDE crossing point(s) (secondary cross-check): {stats_out['crossing_points']}

Fisher's exact test (did the correct-prediction filter bias the split?):
  table={stats_out['fisher_table']}, odds ratio={stats_out['fisher_odds_ratio']:.3f}, p={stats_out['fisher_p']:.4f}
"""
    print(summary)
    summary_path = os.path.join(out_dir, f"{CONCEPT}_calibration_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"[Step 5] Saved summary to {summary_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    img_df = load_heldout_images()

    print(f"\n[Sanity check] Finding a correctly-classified image to test full pipeline...")
    for i in range(min(10, len(img_df))):
        row = img_df.iloc[i]
        attr, pred = compute_attribution_for_image(row["derm"])
        if attr is not None:
            print(f"  {row['derm']}: attribution={attr:.4f}, predicted={pred}, "
                  f"ground_truth={CONCEPT}={row[CONCEPT_META_COLUMN]}")
            print("[Sanity check] Full explain()/CAV path exercised successfully.\n")
            break
        print(f"  {row['derm']}: misclassified as {pred}, skipping (cheap path OK)")
    else:
        print("[Sanity check] WARNING: no correctly-classified image found in first "
              "10 -- check TARGET_CLASS / model / paths before proceeding.\n")

    results_df, total_pos, total_neg = run_attribution_pass(img_df)
    results_df.to_csv(os.path.join(OUTPUT_DIR, f"{CONCEPT}_calibration_attributions.csv"), index=False)

    stats_out = analyze(results_df, total_pos, total_neg)
    plot_and_save(stats_out, OUTPUT_DIR)