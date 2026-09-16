"""
BWV Attribution Calibration Experiment
========================================
Goal: determine an empirically-grounded VTCAV attribution threshold for the
blue_whitish_veil (BWV) concept, by comparing per-image attribution scores
between ground-truth BWV-present vs BWV-absent melanoma images.

Design:
  - Uses ONLY held-out (non-train-split) Derm7pt MEL images, to avoid
    circularity (the BWV CAV was fit using train-split present/absent labels;
    evaluating on those same images would trivially show separation).
  - Computes a per-image scalar attribution value using LocalVisualTCAV's
    own attribution output (computations[layer][concept].attributions[0]) --
    the same per-image quantity your Global VTCAV pipeline averages across
    images to produce its class-level score.
  - Compares the two label groups via:
      1. Mann-Whitney U test (non-parametric, present > absent)
      2. ROC/AUC + Youden's J-optimal threshold (principled cutoff)
      3. KDE crossing point (the simple "eyeball" version, for comparison)

All path/filename/CAV-source config below has been verified against the
real filesystem (meta.csv, the cached CAVs in outputs2/, and the working
src2/VisualTCAV_derma_local_padonly_seed2_hardneg.py runner) rather than
guessed. See the inline comments on VISUAL_TCAV_DIR for the one genuine
judgment call (which of three non-identical cached BWV CAVs to calibrate
against) and how it was resolved.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe for running on a remote server
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve, auc

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PROJECT_ROOT = "/home/student/s/skakkar/scratch/dev-uos/projects/VTCAV_Dermatology"

# VisualTCAV.py lives in src/, this script lives in src2/, and it's not an
# installed package (verified `import VisualTCAV` fails without this) --
# same sys.path setup as src2/VisualTCAV_derma_local_padonly_seed2_hardneg.py.
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from VisualTCAV import Model, LocalVisualTCAV, preprocess_resnet_v2

META_PATH = f"{PROJECT_ROOT}/datasets/release_v0/meta/meta.csv"
TRAIN_IDX_PATH = f"{PROJECT_ROOT}/datasets/release_v0/meta/train_indexes.csv"

# Root that meta.csv's 'derm' column paths (e.g. "NEL/Nel026.jpg") are
# relative to. Verified against datasets/release_v0/images/NEL/Nel026.jpg.
DERM7PT_IMAGE_ROOT = f"{PROJECT_ROOT}/datasets/release_v0/images"

# VisualTCAV working directory holding the cached CAVs to calibrate against.
# This is the "padonly_seed2_hardneg" run -- the one your most recent
# (Aug 4) proper-IG significance test actually used for its headline
# blue_whitish_veil result (NV: mean=0.1653 p=0.0000 significant;
# MEL: mean=0.0097 p=0.0859 not significant). Two other cached BWV CAVs
# exist in outputs2/ (vtcav_padonly_seed2, and vtcav_padonly_seed2_hardneg_logistic
# which uses a different logistic-regression CAV direction from the
# hardneg-logistic branch) -- neither is byte-identical to this one, so
# don't swap VISUAL_TCAV_DIR without also re-confirming which result it's
# meant to calibrate.
VISUAL_TCAV_DIR = f"{PROJECT_ROOT}/outputs2/vtcav_padonly_seed2_hardneg"
MODELS_DIR = None           # None -> defaults to VISUAL_TCAV_DIR/models
CACHE_DIR = None            # None -> defaults to VISUAL_TCAV_DIR/cache

# NOT VISUAL_TCAV_DIR/concept_images -- that path doesn't exist (verified).
# Concept images live in a project-wide shared folder, same as CONCEPT_DIR
# in src2/VisualTCAV_derma_local_padonly_seed2_hardneg.py; every vtcav_*
# run points its CACHE at this same shared concept_images/ source.
CONCEPT_IMAGES_DIR = f"{PROJECT_ROOT}/concept_images"

# Exact filenames, verified against src2/VisualTCAV_derma_local_padonly_seed2_hardneg.py.
MODEL_NAME = "resnet50v2"                                 # architecture family (drives layer selection logic in _get_layer_tensors)
GRAPH_PATH_FILENAME = "resnet50v2_isic2019_final.keras"   # expected at {MODELS_DIR}/{MODEL_NAME}/{GRAPH_PATH_FILENAME}
LABEL_PATH_FILENAME = "isic2019_classes.txt"              # expected at {MODELS_DIR}/{MODEL_NAME}/{LABEL_PATH_FILENAME}

# Verified as the first line of isic2019_classes.txt.
TARGET_CLASS = "MEL"

CONCEPT = "blue_whitish_veil"   # concept root -- matches concept_images/blue_whitish_veil/{positive,negative}
LAYER = "post_relu"

# MUST match your canonical CAV-training hyperparameters, or _compute_cavs()
# will retrain a fresh CAV instead of loading your existing cached one.
MAX_EXAMPLES = 200
N_CAV_RUNS = 20
M_STEPS = 50
RESIZE_MODE = "pad"

# blue_whitish_veil has no typical/atypical pairing (unlike pigment_network
# or streaks), so it uses absent-only negatives. Verified: it is absent from
# the EXTRA_NEGATIVES dict in src2/VisualTCAV_derma_local_padonly_seed2_hardneg.py,
# which lists extra sources only for pigment_network_typical/atypical,
# streaks_regular/irregular, and dots_and_globules_regular/irregular.
EXTRA_NEGATIVE_CONCEPTS = {}

RESTRICT_TO_CORRECT_PREDICTIONS = True  # only include images the model itself predicts as MEL

OUTPUT_DIR = f"{PROJECT_ROOT}/outputs2/bwv_calibration"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# STEP 1 -- Load metadata, isolate held-out MEL images, label by BWV status
# ---------------------------------------------------------------------------
def load_heldout_mel_images():
    meta = pd.read_csv(META_PATH)
    train_idx = set(pd.read_csv(TRAIN_IDX_PATH)["indexes"])

    heldout = meta.loc[~meta.index.isin(train_idx)].copy()
    mel = heldout[heldout["diagnosis"].str.contains("melanoma", case=False)].copy()
    mel = mel[mel["blue_whitish_veil"].isin(["present", "absent"])].copy()
    mel["bwv_label"] = (mel["blue_whitish_veil"] == "present").astype(int)

    print(f"[Step 1] Held-out MEL images with valid BWV label: {len(mel)}")
    print(f"          present={mel['bwv_label'].sum()}, "
          f"absent={(mel['bwv_label'] == 0).sum()}")
    return mel


# ---------------------------------------------------------------------------
# STEP 2 -- Build a fresh Model per image and run Local VTCAV
# ---------------------------------------------------------------------------
def build_model():
    return Model(
        model_name=MODEL_NAME,
        graph_path_filename=GRAPH_PATH_FILENAME,
        label_path_filename=LABEL_PATH_FILENAME,
        # VisualTCAV.py's preprocess_resnet_v2 is (confusingly) bound to
        # inception_resnet_v2.preprocess_input, but that and the real
        # resnet_v2.preprocess_input both just call imagenet_utils
        # .preprocess_input(mode="tf") -- verified identical, so this is
        # the correct preprocessing regardless of the misleading name.
        preprocessing_function=preprocess_resnet_v2,
        binary_classification=False,
        max_examples=MAX_EXAMPLES,
        resize_mode=RESIZE_MODE,
    )


def compute_attribution_for_image(derm_relpath):
    """
    Builds a fresh Model + LocalVisualTCAV for a single image, runs
    prediction + explanation, and returns (attribution: float or None, predicted_class: str).

    attribution is None when RESTRICT_TO_CORRECT_PREDICTIONS is True and the
    model's top-1 prediction isn't TARGET_CLASS -- explain() (CAV lookup +
    a 51-step integrated-gradients pass at M_STEPS=50) is skipped in that
    case, since the image would be discarded by run_attribution_pass anyway.

    A fresh Model object is built per image (cheap -- Model.__init__ does no
    I/O). The underlying Keras model IS reloaded from disk on each call, via
    LocalVisualTCAV -> VisualTCAV._bindModel -> KerasModelWrapper.__init__,
    since VisualTCAV's binding logic does not support reusing an
    already-bound wrapper across separate LocalVisualTCAV instances (the
    second call would try to call an already-instantiated wrapper object as
    if it were still the class). This matches the same per-image cost your
    existing Local VTCAV runs already pay.

    CAV training itself is NOT repeated per image, as long as MAX_EXAMPLES
    and N_CAV_RUNS above match your original training run -- _compute_cavs()
    will load the cached ConceptLayer/CAV file instead of retraining.
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

    attribution = float(
        local.computations[LAYER][f"{CONCEPT}/positive"].attributions[0]
    )

    return attribution, predicted_class


# ---------------------------------------------------------------------------
# STEP 3 -- Run attribution computation across the held-out MEL set
# ---------------------------------------------------------------------------
def run_attribution_pass(mel_df):
    records = []
    skipped_wrong_pred = 0
    skipped_error = 0

    for i, (idx, row) in enumerate(mel_df.iterrows()):
        try:
            attribution, predicted_class = compute_attribution_for_image(row["derm"])
        except Exception as e:
            print(f"  [WARN] failed on {row['derm']}: {e}")
            skipped_error += 1
            continue

        if attribution is None:
            skipped_wrong_pred += 1
            continue

        records.append({
            "case_num": row["case_num"],
            "derm": row["derm"],
            "bwv_label": row["bwv_label"],
            "attribution": attribution,
            "predicted_class": predicted_class,
        })

        if (i + 1) % 10 == 0:
            print(f"  processed {i + 1}/{len(mel_df)}")

    print(f"[Step 3] Computed attribution for {len(records)} images "
          f"({skipped_wrong_pred} skipped as misclassified, "
          f"{skipped_error} skipped on error)")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# STEP 4 -- Statistics: Mann-Whitney U, ROC/AUC, Youden's J, KDE crossing
# ---------------------------------------------------------------------------
def analyze(results_df):
    present = results_df.loc[results_df["bwv_label"] == 1, "attribution"].values
    absent = results_df.loc[results_df["bwv_label"] == 0, "attribution"].values
    n1, n2 = len(present), len(absent)

    print(f"\n[Step 4] Present: n={n1}, mean={present.mean():.4f}, "
          f"median={np.median(present):.4f}, std={present.std():.4f}")
    print(f"         Absent:  n={n2}, mean={absent.mean():.4f}, "
          f"median={np.median(absent):.4f}, std={absent.std():.4f}")

    u_stat, p_value = stats.mannwhitneyu(present, absent, alternative="greater")
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
    print(f"\nMann-Whitney U: U={u_stat:.2f}, p={p_value:.6g}")
    print(f"Rank-biserial correlation (effect size): {rank_biserial:.3f}")

    y_true = results_df["bwv_label"].values
    y_score = results_df["attribution"].values
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    best_threshold = thresholds[best_idx]
    best_sens, best_spec = tpr[best_idx], 1 - fpr[best_idx]

    print(f"\nROC AUC: {roc_auc:.3f}")
    print(f"Youden's J optimal threshold: {best_threshold:.4f}")
    print(f"  Sensitivity: {best_sens:.3f}  Specificity: {best_spec:.3f}")

    x_min, x_max = min(present.min(), absent.min()), max(present.max(), absent.max())
    x_grid = np.linspace(x_min, x_max, 1000)
    kde_present, kde_absent = gaussian_kde(present), gaussian_kde(absent)
    d_present, d_absent = kde_present(x_grid), kde_absent(x_grid)
    diff = d_present - d_absent
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    crossing_points = x_grid[sign_changes]
    print(f"\nKDE crossing point(s): {crossing_points}")

    return {
        "present": present, "absent": absent, "n1": n1, "n2": n2,
        "u_stat": u_stat, "p_value": p_value, "rank_biserial": rank_biserial,
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
        "best_idx": best_idx, "best_threshold": best_threshold,
        "best_sens": best_sens, "best_spec": best_spec,
        "x_grid": x_grid, "d_present": d_present, "d_absent": d_absent,
        "crossing_points": crossing_points,
    }


# ---------------------------------------------------------------------------
# STEP 5 -- Plot + save summary
# ---------------------------------------------------------------------------
def plot_and_save(stats_out, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(stats_out["absent"], bins=20, alpha=0.5, density=True,
            label=f'BWV Absent (n={stats_out["n2"]})', color="steelblue")
    ax.hist(stats_out["present"], bins=20, alpha=0.5, density=True,
            label=f'BWV Present (n={stats_out["n1"]})', color="firebrick")
    ax.plot(stats_out["x_grid"], stats_out["d_absent"], color="steelblue", lw=2)
    ax.plot(stats_out["x_grid"], stats_out["d_present"], color="firebrick", lw=2)
    for cp in stats_out["crossing_points"]:
        ax.axvline(cp, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(stats_out["best_threshold"], color="black", lw=2,
               label=f'Youden threshold={stats_out["best_threshold"]:.4f}')
    ax.set_xlabel("VTCAV Attribution (blue_whitish_veil)")
    ax.set_ylabel("Density")
    ax.set_title("Attribution Distribution: BWV Present vs Absent\n(held-out MEL images)")
    ax.legend()

    ax2 = axes[1]
    ax2.plot(stats_out["fpr"], stats_out["tpr"], color="darkorange", lw=2,
             label=f'ROC (AUC={stats_out["roc_auc"]:.3f})')
    ax2.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax2.scatter(stats_out["fpr"][stats_out["best_idx"]],
                stats_out["tpr"][stats_out["best_idx"]],
                color="black", zorder=5, label="Youden J optimal")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC: Attribution as BWV-presence classifier")
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "bwv_calibration_plot.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\n[Step 5] Saved plot to {plot_path}")

    summary = f"""
BWV Attribution Calibration Summary
=====================================
Sample: {stats_out['n1'] + stats_out['n2']} held-out MEL images (never used in CAV training)
  BWV present: {stats_out['n1']}
  BWV absent:  {stats_out['n2']}

Attribution stats:
  Present: mean={stats_out['present'].mean():.4f}, median={np.median(stats_out['present']):.4f}, std={stats_out['present'].std():.4f}
  Absent:  mean={stats_out['absent'].mean():.4f}, median={np.median(stats_out['absent']):.4f}, std={stats_out['absent'].std():.4f}

Mann-Whitney U test (present > absent): U={stats_out['u_stat']:.2f}, p={stats_out['p_value']:.6g}
Rank-biserial correlation (effect size): {stats_out['rank_biserial']:.3f}

ROC AUC: {stats_out['roc_auc']:.3f}
Youden's J optimal threshold: {stats_out['best_threshold']:.4f}
  Sensitivity: {stats_out['best_sens']:.3f}
  Specificity: {stats_out['best_spec']:.3f}

KDE crossing point(s): {stats_out['crossing_points']}
"""
    print(summary)
    summary_path = os.path.join(out_dir, "bwv_calibration_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"[Step 5] Saved summary to {summary_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mel_df = load_heldout_mel_images()

    # Quick sanity check on a single image before running the full 162 --
    # catches path/filename issues early instead of failing at image 80.
    print("\n[Sanity check] Running attribution on first image only...")
    first_row = mel_df.iloc[0]
    attr, pred = compute_attribution_for_image(first_row["derm"])
    attr_str = f"{attr:.4f}" if attr is not None else "SKIPPED (misclassified)"
    print(f"  {first_row['derm']}: attribution={attr_str}, predicted={pred}, "
          f"ground_truth_bwv={first_row['blue_whitish_veil']}")
    print("[Sanity check] OK -- proceeding to full run.\n")

    results_df = run_attribution_pass(mel_df)
    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "bwv_calibration_attributions.csv"), index=False
    )

    stats_out = analyze(results_df)
    plot_and_save(stats_out, OUTPUT_DIR)
    