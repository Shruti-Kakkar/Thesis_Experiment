"""
VisualTCAV_derma_ttest_padonly_seed2.py
Statistical significance testing for Visual-TCAV results -- padding-only
(seed 2) model variant.

ADAPTED FROM VisualTCAV_derma_ttest.py (Run 1 / Run 2) TO:
  - Point at the padding-only seed2 model's cache (isolated from Run 1/2 cache)
  - Write results to a separate TTEST_DIR

Must be run AFTER VisualTCAV_derma_global_padonly_seed2.py, since it reuses
that run's MODEL_TAG-namespaced cache dir (for the model wrapper location and
negative-activation cache) -- same dependency structure as the original script.

Follows Lucieri et al. (2020) Section IV-C:
  - 20 real CAVs per concept per layer (different random splits)
  - 50 random CAVs per concept per layer for t-test baseline
  - Two-sided Welch t-test: real vs random attribution distributions
  - p < 0.05 → significant; p >= 0.05 → marked with asterisk

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import os
import json
import numpy as np
from scipy import stats
from joblib import load, dump
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prettytable import PrettyTable

# ─────────────────────────────────────────────
# 0. MODEL VARIANT TAG — must match the global script's tag exactly,
# so this reads the SAME isolated cache that script populated.
# ─────────────────────────────────────────────
MODEL_TAG = "padonly_seed2"

# ─────────────────────────────────────────────
# 1. PATHS
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser(
    "~/scratch/dev-uos/projects/VTCAV_Dermatology"
)
VTCAV_DIR    = os.path.join(PROJECT_ROOT, "outputs2", f"vtcav_{MODEL_TAG}")
MODELS_DIR   = os.path.join(VTCAV_DIR, "models")
CACHE_DIR    = os.path.join(VTCAV_DIR, "cache", "resnet50v2")
TEST_DIR     = os.path.join(PROJECT_ROOT, "datasets", "test_images_by_class")  # shared
CONCEPT_DIR  = os.path.join(PROJECT_ROOT, "concept_images")                     # shared
TTEST_DIR    = os.path.join(PROJECT_ROOT, "outputs2", f"vtcav_ttest_{MODEL_TAG}")

os.makedirs(TTEST_DIR, exist_ok=True)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from VisualTCAV import KerasModelWrapper, ImageActivationGenerator
from tensorflow.keras.applications.resnet_v2 import (
    preprocess_input as preprocess_resnet_v2
)
import tensorflow as tf

# ─────────────────────────────────────────────
# 2. CONSTANTS
# ─────────────────────────────────────────────
CLASSES       = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
LAYERS        = ["conv5_block1_out", "conv5_block2_out",
                 "conv5_block3_out", "post_relu"]
N_RANDOM_CAVS = 50
MAX_EXAMPLES  = 200
ALPHA         = 0.05
N_CAV_RUNS    = 20

CONCEPTS = {
    "pigment_network_typical":      "pigment_network_typical/positive",
    "pigment_network_atypical":     "pigment_network_atypical/positive",
    "streaks_regular":              "streaks_regular/positive",
    "streaks_irregular":            "streaks_irregular/positive",
    "pigmentation":                 "pigmentation/positive",
    "regression_structures":        "regression_structures/positive",
    "dots_and_globules_regular":    "dots_and_globules_regular/positive",
    "dots_and_globules_irregular":  "dots_and_globules_irregular/positive",
    "blue_whitish_veil":            "blue_whitish_veil/positive",
    "vascular_structures":          "vascular_structures/positive",
}

print("=" * 60)
print(f"Visual-TCAV Statistical Significance Testing [{MODEL_TAG}]")
print(f"N random CAVs: {N_RANDOM_CAVS} | Alpha: {ALPHA}")
print("=" * 60)

# ─────────────────────────────────────────────
# 3. LOAD MODEL WRAPPER
# ─────────────────────────────────────────────
print("\nLoading model wrapper...")
model_path = os.path.join(MODELS_DIR, "resnet50v2", "resnet50v2_isic2019_final.keras")
if not os.path.exists(model_path):
    raise RuntimeError(
        f"Model symlink not found at {model_path}. "
        f"Run VisualTCAV_derma_global_padonly_seed2.py first -- it creates "
        f"this symlink and populates the cache this script depends on."
    )

wrapper = KerasModelWrapper(
    model_path,
    os.path.join(MODELS_DIR, "resnet50v2", "isic2019_classes.txt"),
    batch_size=20,
    # CRITICAL: this script calls KerasModelWrapper directly, bypassing
    # Model/_bindModel, so model_name must be passed explicitly here too --
    # otherwise _get_layer_tensors falls back to the loaded model's raw
    # (unreliable) .name attribute. See VisualTCAV.py fix notes.
    model_name="resnet50v2",
)
print(f"  Model: {wrapper.model_name}")
print(f"  Path : {model_path} -> {os.path.realpath(model_path)}")

# ─────────────────────────────────────────────
# 4. FAST ATTRIBUTION: dot product only
# ─────────────────────────────────────────────
def fast_attribution(test_fmaps_pooled, direction):
    direction_np = direction.numpy() if hasattr(direction, 'numpy') else direction
    scores = test_fmaps_pooled @ direction_np
    return scores.tolist()


def load_or_compute_test_fmaps(class_name, layer_name):
    cache_path = os.path.join(
        CACHE_DIR, f"test_fmaps_{class_name}_{layer_name}.joblib"
    )
    if os.path.exists(cache_path):
        return load(cache_path)

    gen = ImageActivationGenerator(
        model_wrapper=wrapper,
        concept_images_dir=TEST_DIR,
        cache_dir=CACHE_DIR,
        preprocessing_function=preprocess_resnet_v2,
        max_examples=MAX_EXAMPLES,
        # CRITICAL: must match training preprocessing -- see notes in
        # VisualTCAV_derma_global_padonly_seed2.py
        resize_mode='pad',
    )
    fmaps = gen.get_feature_maps_for_concept(class_name, layer_name)
    dump(fmaps, cache_path, compress=3)
    return fmaps


# ─────────────────────────────────────────────
# 5. MAIN LOOP
# ─────────────────────────────────────────────
ttest_results = {}

for concept_name, concept_folder in CONCEPTS.items():

    ttest_results[concept_name] = {}
    concept_root = concept_folder.split('/')[0]
    safe_name    = concept_name.replace('/', '_')

    print(f"\n{'─'*60}")
    print(f"Concept: {concept_name}")
    print(f"{'─'*60}")

    for layer_name in LAYERS:

        ttest_results[concept_name][layer_name] = {}

        neg_cache = os.path.join(
            CACHE_DIR,
            f"neg_acts_{concept_root}_{MAX_EXAMPLES}_{layer_name}.joblib"
        )
        if not os.path.exists(neg_cache):
            print(f"  [SKIP] Negative cache not found: {neg_cache}")
            continue

        neg_acts = load(neg_cache)
        pooled_neg = tf.reduce_mean(neg_acts, axis=(1, 2)).numpy()
        n_neg = len(pooled_neg)

        gen = ImageActivationGenerator(
            model_wrapper=wrapper,
            concept_images_dir=CONCEPT_DIR,
            cache_dir=CACHE_DIR,
            preprocessing_function=preprocess_resnet_v2,
            max_examples=MAX_EXAMPLES,
            resize_mode='pad',
        )
        pos_acts = gen.get_feature_maps_for_concept(concept_folder, layer_name)
        pooled_pos = tf.reduce_mean(pos_acts, axis=(1, 2)).numpy()
        n_pos = len(pooled_pos)
        n_min = min(n_pos, n_neg)

        print(f"  Layer {layer_name}: pos={n_pos} neg={n_neg}")

        real_directions = []
        for run in range(N_CAV_RUNS):
            rng = np.random.default_rng(42 + run)
            pos_idx = rng.permutation(n_pos)[:n_min]
            neg_idx = rng.permutation(n_neg)[:n_min]
            n_train = int(n_min * 0.8)
            c0 = np.mean(pooled_pos[pos_idx[:n_train]], axis=0)
            c1 = np.mean(pooled_neg[neg_idx[:n_train]], axis=0)
            real_directions.append(c0 - c1)

        mean_real_direction = np.mean(real_directions, axis=0)

        random_directions = []
        for rand_idx in range(N_RANDOM_CAVS):
            rng = np.random.default_rng(rand_idx * 1000)
            idx = rng.permutation(n_neg)
            half = len(idx) // 2
            rc0 = np.mean(pooled_neg[idx[:half]], axis=0)
            rc1 = np.mean(pooled_neg[idx[half:]], axis=0)
            random_directions.append(rc0 - rc1)

        for class_name in CLASSES:

            try:
                test_fmaps = load_or_compute_test_fmaps(class_name, layer_name)
            except Exception as e:
                print(f"    [SKIP] {class_name}: {e}")
                continue

            test_pooled = tf.reduce_mean(test_fmaps, axis=(1, 2)).numpy()

            real_scores = fast_attribution(test_pooled, mean_real_direction)

            random_mean_scores = []
            for rand_dir in random_directions:
                rand_scores = fast_attribution(test_pooled, rand_dir)
                random_mean_scores.append(float(np.mean(rand_scores)))

            if np.std(real_scores) == 0 and np.std(random_mean_scores) == 0:
                p_value = float('nan')
                t_stat  = float('nan')
            else:
                t_stat, p_value = stats.ttest_ind(
                    real_scores, random_mean_scores, equal_var=False
                )

            significant = (not np.isnan(p_value)) and (p_value < ALPHA)

            ttest_results[concept_name][layer_name][class_name] = {
                'mean':        float(np.mean(real_scores)),
                'std':         float(np.std(real_scores)),
                'rand_mean':   float(np.mean(random_mean_scores)),
                'rand_std':    float(np.std(random_mean_scores)),
                't_statistic': float(t_stat) if not np.isnan(t_stat) else None,
                'p_value':     float(p_value) if not np.isnan(p_value) else None,
                'significant': bool(significant),
            }

            sig_str = "✓ significant" if significant else "* NOT significant"
            p_str   = f"{p_value:.4f}" if not np.isnan(p_value) else "nan"
            print(f"    {class_name}: mean={np.mean(real_scores):.5f} "
                  f"p={p_str} {sig_str}")

# ─────────────────────────────────────────────
# 6. SAVE RESULTS
# ─────────────────────────────────────────────
results_path = os.path.join(TTEST_DIR, "ttest_results.json")
with open(results_path, 'w') as f:
    json.dump(ttest_results, f, indent=2)
print(f"\nResults saved: {results_path}")

# ─────────────────────────────────────────────
# 7. SUMMARY TABLES
# ─────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"SUMMARY — post_relu layer [{MODEL_TAG}]")
print(f"{'='*70}")

for focus_class in CLASSES:
    print(f"\nClass: {focus_class}")
    table = PrettyTable(field_names=[
        "Concept", "Mean", "p-value", "Significant"
    ])
    for concept_name in CONCEPTS:
        r = ttest_results.get(concept_name, {}).get(
            "post_relu", {}
        ).get(focus_class, {})
        mean = r.get('mean', 0)
        pval = r.get('p_value', None)
        sig  = r.get('significant', False)
        p_str = f"{pval:.4f}" if pval is not None else "nan"
        table.add_row([
            concept_name,
            f"{mean:.5f}",
            p_str,
            "✓" if sig else "* NS"
        ])
    print(table)

# ─────────────────────────────────────────────
# 8. PLOTS
# ─────────────────────────────────────────────
concept_list    = list(CONCEPTS.keys())
concept_display = [
    "PN Typ", "PN Atyp", "ST Reg", "ST Irreg",
    "Pigment", "Regress", "DG Reg", "DG Irreg",
    "BWV", "Vascular"
]

for focus_class in CLASSES:
    fig, ax = plt.subplots(figsize=(13, 5))
    x     = np.arange(len(concept_list))
    width = 0.18
    cmap  = plt.cm.viridis(np.linspace(0.1, 0.9, len(LAYERS)))

    for i, layer_name in enumerate(LAYERS):
        means = []
        errs  = []
        sigs  = []
        for concept_name in concept_list:
            r = ttest_results.get(concept_name, {}).get(
                layer_name, {}
            ).get(focus_class, {})
            means.append(r.get('mean', 0))
            errs.append(r.get('std', 0) / np.sqrt(MAX_EXAMPLES) * 2)
            sigs.append(r.get('significant', False))

        pos  = x + (i - len(LAYERS)/2) * width + width/2
        bars = ax.bar(
            pos, means, width=width, yerr=errs,
            label=layer_name.replace('_', ' '),
            color=cmap[i], capsize=3, zorder=2,
        )
        for bar, sig in zip(bars, sigs):
            if not sig:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0001,
                    '*', ha='center', va='bottom',
                    color='red', fontsize=14, fontweight='bold'
                )

    ax.set_ylabel('Attribution mean (2σ error bars)')
    ax.set_title(
        f'ResNet50V2 [{MODEL_TAG}] — {focus_class}\n'
        f'* = not significant (p ≥ {ALPHA}, Welch t-test, 50 random CAVs)',
        fontsize=10
    )
    ax.set_xticks(x)
    ax.set_xticklabels(concept_display, rotation=15, ha='right', fontsize=9)
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', fontsize=8)
    ax.grid(linewidth=0.3, zorder=1)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=-0.5, right=len(concept_list) - 0.5)
    plt.tight_layout()

    plot_path = os.path.join(TTEST_DIR, f"ttest_plot_{focus_class}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")

print(f"\nAll done. Results in: {TTEST_DIR}")