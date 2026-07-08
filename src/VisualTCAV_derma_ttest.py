"""
VisualTCAV_derma_ttest.py
Statistical significance testing for Visual-TCAV results.

Follows Lucieri et al. (2020) Section IV-C exactly:
  - 20 real CAVs per concept per layer (different random splits)
    already computed by VisualTCAV_derma_global.py
  - 50 random CAVs per concept per layer for t-test baseline
    Random CAVs: randomly shuffle labels within the absent image set
    (same source as real CAV negatives — Option A)
  - Two-sided t-test: real attribution distribution vs random
  - p < 0.05 → significant
  - p >= 0.05 → marked with asterisk (not significant)

Author: Shruti Kakkar
References:
  - Lucieri et al. (2020) Section IV-C
  - Kim et al. (2018) TCAV statistical testing
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
# 1. PATHS
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser(
    "~/scratch/dev-uos/projects/VTCAV_Dermatology"
)
VTCAV_DIR       = os.path.join(PROJECT_ROOT, "outputs", "vtcav")
MODELS_DIR      = os.path.join(VTCAV_DIR, "models")
CACHE_DIR       = os.path.join(VTCAV_DIR, "cache", "resnet50v2")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "datasets", "test_images_by_class")
CONCEPT_DIR     = os.path.join(PROJECT_ROOT, "concept_images")
RESULTS_DIR     = os.path.join(PROJECT_ROOT, "outputs", "vtcav_results")
TTEST_DIR       = os.path.join(PROJECT_ROOT, "outputs", "vtcav_ttest")

os.makedirs(TTEST_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from VisualTCAV import (
    KerasModelWrapper, ImageActivationGenerator,
    ConceptLayer, Cav, Stat, contraharmonic_mean
)
from tensorflow.keras.applications.resnet_v2 import (
    preprocess_input as preprocess_resnet_v2
)
import tensorflow as tf
import tensorflow_probability as tfp

# ─────────────────────────────────────────────
# 2. CONSTANTS
# ─────────────────────────────────────────────
CLASSES       = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]
LAYERS        = ["conv5_block1_out", "conv5_block2_out",
                 "conv5_block3_out", "post_relu"]
N_RANDOM_CAVS = 50    # following Lucieri et al.
MAX_EXAMPLES  = 200
ALPHA         = 0.05
N_CAV_RUNS    = 20    # must match VisualTCAV_derma_global.py

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
print("Visual-TCAV Statistical Significance Testing")
print(f"N random CAVs: {N_RANDOM_CAVS} | Alpha: {ALPHA}")
print("Random CAV source: Derm7pt absent images (Option A)")
print("=" * 60)

# ─────────────────────────────────────────────
# 3. LOAD MODEL WRAPPER
# ─────────────────────────────────────────────
print("\nLoading model wrapper...")
wrapper = KerasModelWrapper(
    os.path.join(MODELS_DIR, "resnet50v2", "resnet50v2_isic2019_final.keras"),
    os.path.join(MODELS_DIR, "resnet50v2", "isic2019_classes.txt"),
    batch_size=20,
)
print(f"  Model: {wrapper.model_name}")
print(f"  Layers: {list(wrapper.layer_tensors.keys())}")

# ─────────────────────────────────────────────
# 4. ACTIVATION GENERATOR
# ─────────────────────────────────────────────
activation_gen = ImageActivationGenerator(
    model_wrapper=wrapper,
    concept_images_dir=CONCEPT_DIR,
    cache_dir=CACHE_DIR,
    preprocessing_function=preprocess_resnet_v2,
    max_examples=MAX_EXAMPLES,
)

# ─────────────────────────────────────────────
# 5. HELPER: compute CAV direction from pooled acts
# ─────────────────────────────────────────────
def compute_direction_and_emblem(pos_acts, neg_acts):
    """
    Compute CAV direction = centroid(pos) - centroid(neg)
    and concept emblem from spatial feature maps.
    """
    pooled_pos = tf.reduce_mean(pos_acts, axis=(1, 2))
    pooled_neg = tf.reduce_mean(neg_acts, axis=(1, 2))

    c0 = tf.reduce_mean(pooled_pos, axis=0)
    c1 = tf.reduce_mean(pooled_neg, axis=0)
    direction = tf.subtract(c0, c1)

    emblems = contraharmonic_mean(
        tf.nn.relu(
            tf.reduce_sum(
                tf.multiply(direction[None, None, None, :], pos_acts),
                axis=3
            )
        ),
        axis=(1, 2)
    )
    negative_emblems = contraharmonic_mean(
        tf.nn.relu(
            tf.reduce_sum(
                tf.multiply(direction[None, None, None, :], neg_acts),
                axis=3
            )
        ),
        axis=(1, 2)
    )
    concept_emblem = tf.cast(
        (tfp.stats.percentile(emblems, 50.0),
         tfp.stats.percentile(negative_emblems, 50.0)),
        tf.float32
    )
    return direction, concept_emblem


def compute_attribution_for_fmap(feature_maps, layer_name, class_idx,
                                  direction, concept_emblem, m_steps=50):
    """
    Compute Visual-TCAV attribution for one image's feature maps.
    Exactly as GlobalVisualTCAV.explain() core loop.
    """
    # Logits
    logits = wrapper.get_logits(
        np.expand_dims(feature_maps, axis=0), layer_name
    )[0]
    logits_baseline = wrapper.get_logits(
        np.expand_dims(tf.zeros(shape=feature_maps.shape), axis=0),
        layer_name
    )[0]

    ig_expected = tf.nn.relu(tf.subtract(logits, logits_baseline))
    ig_max = tf.reduce_max(ig_expected)
    if ig_max > 0:
        ig_expected_norm = tf.divide(ig_expected, ig_max)
    else:
        ig_expected_norm = ig_expected
    ig_expected_class = ig_expected_norm[class_idx]

    # Integrated gradients
    alphas = tf.linspace(0.0, 1.0, m_steps + 1)
    baseline = tf.zeros(shape=feature_maps.shape)
    image = tf.image.convert_image_dtype(feature_maps, tf.float32)
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = tf.subtract(input_x, baseline_x)
    interpolated = tf.add(baseline_x, tf.multiply(alphas_x, delta))

    grads = wrapper.get_gradient_of_score(interpolated, layer_name, class_idx)
    ig = tf.math.reduce_mean(
        (np.array(grads)[:-1] + np.array(grads)[1:]) / tf.constant(2.0),
        axis=0,
    )

    # Attributions
    attributions = tf.nn.relu(tf.multiply(ig, feature_maps))
    attributions = tf.multiply(
        tf.divide(attributions,
                  tf.add(tf.reduce_sum(attributions),
                         tf.keras.backend.epsilon())),
        ig_expected_class
    )

    # Concept map
    concept_map = tf.nn.relu(
        tf.math.reduce_sum(
            tf.multiply(direction[None, None, :], feature_maps),
            axis=2
        )
    )

    # Normalize concept map
    if concept_emblem[0] > concept_emblem[1]:
        concept_map = tf.where(concept_map > concept_emblem[0], concept_emblem[0], concept_map)
        concept_map = tf.where(concept_map < concept_emblem[1], concept_emblem[1], concept_map)
        concept_map = (concept_map - concept_emblem[1]) / (concept_emblem[0] - concept_emblem[1])
    else:
        concept_map = tf.multiply(concept_map, 0)

    # Masked attributions
    pooled_masked = tf.reduce_sum(
        tf.multiply(attributions, concept_map[:, :, None]),
        axis=(0, 1)
    )

    # Normalized CAV
    if tf.reduce_min(feature_maps) < 0:
        pooled_cav_norm = tf.nn.relu(
            tf.multiply(
                direction,
                tf.where(
                    tf.reduce_sum(
                        tf.multiply(feature_maps, concept_map[:, :, None]),
                        axis=(0, 1)
                    ) < 0, -1.0, 1.0
                )
            )
        )
    else:
        pooled_cav_norm = tf.nn.relu(direction)

    max_cav = tf.reduce_max(pooled_cav_norm)
    if max_cav > 0:
        pooled_cav_norm = tf.divide(pooled_cav_norm, max_cav)

    return float(tf.tensordot(pooled_cav_norm, pooled_masked, axes=1))


# ─────────────────────────────────────────────
# 6. LOAD TEST IMAGE FEATURE MAPS (cached per class/layer)
# ─────────────────────────────────────────────
def get_test_fmaps(class_name, layer_name):
    cache_path = os.path.join(
        CACHE_DIR, f"test_fmaps_{class_name}_{layer_name}.joblib"
    )
    if os.path.exists(cache_path):
        return load(cache_path)
    # Load via activation generator
    activation_gen.concept_images_dir = TEST_IMAGES_DIR
    fmaps = activation_gen.get_feature_maps_for_concept(class_name, layer_name)
    activation_gen.concept_images_dir = CONCEPT_DIR
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    dump(fmaps, cache_path, compress=3)
    return fmaps


# ─────────────────────────────────────────────
# 7. MAIN LOOP
# For each concept:
#   a) Load cached negative (absent) activations
#   b) Compute real attribution scores using cached CAV directions
#   c) Compute 50 random CAV attributions from shuffled absent images
#   d) T-test real vs random
# ─────────────────────────────────────────────
ttest_results = {}

for concept_name, concept_folder in CONCEPTS.items():

    ttest_results[concept_name] = {}
    concept_root = concept_folder.split('/')[0]  # e.g. "pigment_network_typical"
    safe_name    = concept_name.replace('/', '_')

    print(f"\n{'─'*60}")
    print(f"Concept: {concept_name}")
    print(f"{'─'*60}")

    for layer_name in LAYERS:

        ttest_results[concept_name][layer_name] = {}

        # ── Load cached negative (absent) activations ─────────────
        neg_cache = os.path.join(
            CACHE_DIR,
            f"neg_acts_{concept_root}_{MAX_EXAMPLES}_{layer_name}.joblib"
        )
        if not os.path.exists(neg_cache):
            print(f"  [SKIP] Negative cache not found: {neg_cache}")
            continue
        neg_acts = load(neg_cache)   # (N_neg, 7, 7, 2048)
        print(f"  Layer {layer_name}: negative acts {neg_acts.shape}")

        # ── Load cached positive activations ──────────────────────
        pos_cache = os.path.join(
            CACHE_DIR,
            f"cav_{safe_name}_{MAX_EXAMPLES}_neg_{N_CAV_RUNS}runs_{layer_name}.joblib"
        )
        if not os.path.exists(pos_cache):
            # Try to load feature maps directly
            activation_gen.concept_images_dir = CONCEPT_DIR
            pos_acts = activation_gen.get_feature_maps_for_concept(
                concept_folder, layer_name
            )
        else:
            # Load ConceptLayer from cache — we need original pos acts
            # Re-extract them fresh for the t-test
            activation_gen.concept_images_dir = CONCEPT_DIR
            pos_acts = activation_gen.get_feature_maps_for_concept(
                concept_folder, layer_name
            )

        print(f"  Positive acts: {pos_acts.shape}")

        # Pool for CAV computation
        pooled_pos = tf.reduce_mean(pos_acts, axis=(1, 2)).numpy()
        pooled_neg = tf.reduce_mean(neg_acts, axis=(1, 2)).numpy()
        n_pos, n_neg = len(pooled_pos), len(pooled_neg)
        n_min = min(n_pos, n_neg)

        # ── Compute real attribution scores per class ──────────────
        # Use mean direction from N_CAV_RUNS splits
        real_directions = []
        for run in range(N_CAV_RUNS):
            rng = np.random.default_rng(42 + run)
            pos_idx = rng.permutation(n_pos)[:n_min]
            neg_idx = rng.permutation(n_neg)[:n_min]
            n_train = int(n_min * 0.8)
            c0 = tf.reduce_mean(pooled_pos[pos_idx][:n_train], axis=0)
            c1 = tf.reduce_mean(pooled_neg[neg_idx][:n_train], axis=0)
            real_directions.append(tf.subtract(c0, c1))

        mean_real_direction = tf.reduce_mean(real_directions, axis=0)
        _, mean_real_emblem = compute_direction_and_emblem(pos_acts, neg_acts)

        for class_name in CLASSES:
            class_idx = wrapper.labels.index(class_name)

            # Load test feature maps
            try:
                test_fmaps = get_test_fmaps(class_name, layer_name)
            except Exception as e:
                print(f"    [SKIP] {class_name}: {e}")
                continue

            # Real attribution scores (one per test image)
            real_scores = []
            for fmap in test_fmaps[:MAX_EXAMPLES]:
                score = compute_attribution_for_fmap(
                    fmap, layer_name, class_idx,
                    mean_real_direction, mean_real_emblem
                )
                real_scores.append(score)

            # ── 50 random CAVs from shuffled absent images ─────────
            # Following Option A: randomly relabel absent images
            # 50% → "positive", 50% → "negative"
            # Both groups are absent images — no real concept signal
            random_mean_scores = []

            for rand_idx in range(N_RANDOM_CAVS):
                rng = np.random.default_rng(rand_idx * 1000)
                idx = rng.permutation(len(pooled_neg))
                half = len(idx) // 2

                # Random "positive" = first half of shuffled absent acts
                rand_pos_pooled = pooled_neg[idx[:half]]
                # Random "negative" = second half of shuffled absent acts
                rand_neg_pooled = pooled_neg[idx[half:]]
                rand_pos_acts   = neg_acts[idx[:half]]
                rand_neg_acts   = neg_acts[idx[half:]]

                # Random CAV direction
                rc0 = tf.reduce_mean(rand_pos_pooled, axis=0)
                rc1 = tf.reduce_mean(rand_neg_pooled, axis=0)
                rand_direction = tf.subtract(rc0, rc1)

                # Random emblem
                _, rand_emblem = compute_direction_and_emblem(
                    rand_pos_acts, rand_neg_acts
                )

                # Mean attribution for this random CAV on test images
                rand_scores_run = []
                for fmap in test_fmaps[:MAX_EXAMPLES]:
                    score = compute_attribution_for_fmap(
                        fmap, layer_name, class_idx,
                        rand_direction, rand_emblem
                    )
                    rand_scores_run.append(score)

                random_mean_scores.append(float(np.mean(rand_scores_run)))

            # ── Two-sided t-test ───────────────────────────────────
            t_stat, p_value = stats.ttest_ind(
                real_scores, random_mean_scores,
                equal_var=False  # Welch's t-test
            )
            significant = p_value < ALPHA

            ttest_results[concept_name][layer_name][class_name] = {
                'mean':        float(np.mean(real_scores)),
                'std':         float(np.std(real_scores)),
                'rand_mean':   float(np.mean(random_mean_scores)),
                'rand_std':    float(np.std(random_mean_scores)),
                't_statistic': float(t_stat),
                'p_value':     float(p_value),
                'significant': bool(significant),
                'n_real':      len(real_scores),
                'n_random':    len(random_mean_scores),
            }

            sig_str = "✓ significant" if significant else "* NOT significant"
            print(f"    {class_name}: mean={np.mean(real_scores):.5f} "
                  f"p={p_value:.4f} {sig_str}")

# ─────────────────────────────────────────────
# 8. SAVE RESULTS
# ─────────────────────────────────────────────
results_path = os.path.join(TTEST_DIR, "ttest_results.json")
with open(results_path, 'w') as f:
    json.dump(ttest_results, f, indent=2)
print(f"\nResults saved: {results_path}")

# ─────────────────────────────────────────────
# 9. SUMMARY TABLES — MEL and NV (primary classes)
# ─────────────────────────────────────────────
for focus_class in ["MEL", "NV"]:

    print(f"\n{'='*70}")
    print(f"Class: {focus_class} — post_relu layer")
    print(f"{'='*70}")

    table = PrettyTable(field_names=[
        "Concept", "Mean attrib.", "Rand mean", "p-value", "Significant"
    ])
    table.title = f"Class: {focus_class} | Layer: post_relu | α={ALPHA}"

    for concept_name in CONCEPTS:
        r = ttest_results.get(concept_name, {}).get(
            "post_relu", {}
        ).get(focus_class, {})

        mean  = r.get('mean', 0)
        rmean = r.get('rand_mean', 0)
        pval  = r.get('p_value', 1.0)
        sig   = r.get('significant', False)

        table.add_row([
            concept_name,
            f"{mean:.5f}",
            f"{rmean:.5f}",
            f"{pval:.4f}",
            "✓" if sig else "* NS"
        ])

    print(table)

# ─────────────────────────────────────────────
# 10. PLOTS — following Lucieri et al. Figure 4
#     Red asterisks on non-significant bars
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
            errs.append(r.get('std', 0) / np.sqrt(max(r.get('n_real', 1), 1)) * 2)
            sigs.append(r.get('significant', False))

        pos = x + (i - len(LAYERS)/2) * width + width/2
        bars = ax.bar(
            pos, means,
            width=width,
            yerr=errs,
            label=layer_name.replace('_', ' '),
            color=cmap[i],
            capsize=3,
            zorder=2,
        )

        # Red asterisk on non-significant bars (Lucieri Fig 4 style)
        for bar, sig in zip(bars, sigs):
            if not sig:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(errs) * 0.1 + 0.0001,
                    '*',
                    ha='center', va='bottom',
                    color='red', fontsize=14, fontweight='bold'
                )

    ax.set_ylabel('Attribution (2σ error)')
    ax.set_title(
        f'ResNet50V2 — {focus_class} target class\n'
        f'* = not significant (p ≥ {ALPHA}, Welch t-test vs 50 random CAVs)',
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

print(f"\nAll done. T-test results in: {TTEST_DIR}")
