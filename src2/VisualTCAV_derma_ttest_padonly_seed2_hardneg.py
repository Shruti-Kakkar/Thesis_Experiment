"""
VisualTCAV_derma_ttest_proper_ig.py

NOTE ON FILENAME: despite the name, this file no longer runs the fast
dot-product shortcut -- it was replaced with proper Integrated-Gradients-
based significance testing (see below). Filename kept for continuity
with the hard-negative Global run it reads CAVs from; ~2.5 hours to run,
not the few minutes the old shortcut version took.

"Proper" significance testing -- uses real Integrated-Gradients-based
causal attribution (matching what GlobalVisualTCAV actually computes),
instead of the earlier fast raw-dot-product shortcut.

TWO KEY EFFICIENCY IDEAS, both verified before building this for real:
  1. IG only depends on (image, class, layer) -- NOT on which concept/
     direction you're scoring. Computed ONCE per (image, class, layer),
     then reused across the real CAV direction and all 50 random
     directions for every concept at that layer. This is what makes the
     difference between ~2.5 hours and the "overnight" runtime the
     original fast-shortcut was built to avoid.
  2. The 51-direction application step (real + 50 random) is fully
     vectorized as batched tensor ops rather than a Python for-loop --
     verified numerically equivalent to the loop version before use.

Loads CACHED CAV directions from your existing Global run rather than
refitting them -- so this works regardless of which cav_method (centroid
or logistic) produced them; it just reads whatever's on disk.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import os
import json
import time
import numpy as np
from scipy import stats
from joblib import load, dump
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import tensorflow as tf

# ─────────────────────────────────────────────
# 0. MODEL VARIANT TAG — which Global run's cache to read from.
# Change this to "padonly_seed2_hardneg_logistic" once that run has
# completed, if you want to test the logistic-regression CAVs instead.
# ─────────────────────────────────────────────
MODEL_TAG = "padonly_seed2_hardneg"

# ─────────────────────────────────────────────
# 1. PATHS
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser(
    "~/scratch/dev-uos/projects/VTCAV_Dermatology"
)
VTCAV_DIR    = os.path.join(PROJECT_ROOT, "outputs2", f"vtcav_{MODEL_TAG}")
MODELS_DIR   = os.path.join(VTCAV_DIR, "models")
CACHE_DIR    = os.path.join(VTCAV_DIR, "cache", "resnet50v2")
TEST_DIR     = os.path.join(PROJECT_ROOT, "datasets", "test_images_by_class")
CONCEPT_DIR  = os.path.join(PROJECT_ROOT, "concept_images")
TTEST_DIR    = os.path.join(PROJECT_ROOT, "outputs2", f"vtcav_ttest_proper_ig_{MODEL_TAG}")
IG_CACHE_DIR = os.path.join(TTEST_DIR, "ig_cache")  # separate from CAV cache -- new artifact type

os.makedirs(TTEST_DIR, exist_ok=True)
os.makedirs(IG_CACHE_DIR, exist_ok=True)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from VisualTCAV import KerasModelWrapper, ImageActivationGenerator
from tensorflow.keras.applications.resnet_v2 import (
    preprocess_input as preprocess_resnet_v2
)

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
M_STEPS       = 50  # Integrated Gradients interpolation steps

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

# MUST match whatever the Global run actually used, or cached CAV/negative
# files won't be found by the cache-key logic below.
EXTRA_NEGATIVES = {
    "pigment_network_typical":  ["pigment_network_atypical/positive"],
    "pigment_network_atypical": ["pigment_network_typical/positive"],
    "streaks_regular":          ["streaks_irregular/positive"],
    "streaks_irregular":        ["streaks_regular/positive"],
    "dots_and_globules_regular":   ["dots_and_globules_irregular/positive"],
    "dots_and_globules_irregular": ["dots_and_globules_regular/positive"],
}

print("=" * 60)
print(f"Visual-TCAV PROPER (IG-based) Significance Testing [{MODEL_TAG}]")
print(f"N random CAVs: {N_RANDOM_CAVS} | Alpha: {ALPHA}")
print("=" * 60)

# ─────────────────────────────────────────────
# 3. LOAD MODEL WRAPPER
# ─────────────────────────────────────────────
wrapper = KerasModelWrapper(
    os.path.join(MODELS_DIR, "resnet50v2", "resnet50v2_isic2019_final.keras"),
    os.path.join(MODELS_DIR, "resnet50v2", "isic2019_classes.txt"),
    batch_size=20,
    model_name="resnet50v2",
)
print(f"Model loaded: {wrapper.model_name}\n")


# ─────────────────────────────────────────────
# 4. CACHE-KEY HELPERS — mirror VisualTCAV.py's _compute_cavs /
# _compute_negative_activations EXACTLY, so we correctly locate whatever
# the Global run already cached.
# ─────────────────────────────────────────────
def cav_cache_path(concept_name, layer_name):
    concept_root = concept_name.split('/')[0]
    safe_name = concept_name.replace('/', '_')
    extra_folders = EXTRA_NEGATIVES.get(concept_root, [])
    extra_tag = ""
    if extra_folders:
        extra_safe = "_".join(sorted(f.replace('/', '-') for f in extra_folders))
        extra_tag = f"_plus_{extra_safe}"
    return os.path.join(
        CACHE_DIR,
        f'cav_{safe_name}{extra_tag}_{MAX_EXAMPLES}_neg_{N_CAV_RUNS}runs_{layer_name}.joblib'
    )

def neg_acts_cache_path(concept_root, layer_name):
    extra_folders = EXTRA_NEGATIVES.get(concept_root, [])
    extra_tag = ""
    if extra_folders:
        extra_safe = "_".join(sorted(f.replace('/', '-') for f in extra_folders))
        extra_tag = f"_plus_{extra_safe}"
    return os.path.join(
        CACHE_DIR,
        f'neg_acts_{concept_root}{extra_tag}_{MAX_EXAMPLES}_{layer_name}.joblib'
    )

def load_real_direction(concept_name, layer_name):
    path = cav_cache_path(concept_name, layer_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No cached CAV found at {path}. Run the Global script for "
            f"MODEL_TAG='{MODEL_TAG}' first."
        )
    concept_layer = load(path)
    direction = concept_layer.cav.direction
    return direction.numpy() if hasattr(direction, 'numpy') else np.asarray(direction)

def build_random_directions(concept_name, layer_name, seed_base=1000):
    concept_root = concept_name.split('/')[0]
    path = neg_acts_cache_path(concept_root, layer_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No cached negative activations at {path}")
    neg_acts = load(path)
    pooled_neg = tf.reduce_mean(neg_acts, axis=(1, 2)).numpy()
    n_neg = len(pooled_neg)

    directions = []
    for rand_idx in range(N_RANDOM_CAVS):
        rng = np.random.default_rng(rand_idx * seed_base)
        idx = rng.permutation(n_neg)
        half = len(idx) // 2
        rc0 = np.mean(pooled_neg[idx[:half]], axis=0)
        rc1 = np.mean(pooled_neg[idx[half:]], axis=0)
        directions.append(rc0 - rc1)
    return directions


# ─────────────────────────────────────────────
# 5. INTEGRATED GRADIENTS + SHARED ATTRIBUTIONS (computed ONCE per
# image/class/layer, reused across all 10 concepts)
# ─────────────────────────────────────────────
def compute_integrated_gradients(feature_maps, layer_name, class_idx):
    alphas = tf.linspace(start=0.0, stop=1.0, num=M_STEPS + 1)
    baseline = tf.zeros(shape=feature_maps.shape)
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(feature_maps, axis=0)
    delta = tf.subtract(input_x, baseline_x)
    interpolated = tf.add(baseline_x, tf.multiply(alphas_x, delta))
    grads = wrapper.get_gradient_of_score(interpolated, layer_name, class_idx)
    return tf.math.reduce_mean(
        (np.array(grads)[:-1] + np.array(grads)[1:]) / 2.0, axis=0,
    )

def compute_shared_attributions(feature_maps, layer_name, class_idx):
    """Returns the IG-based attribution tensor (H,W,C) for one image,
    shared across every concept -- the expensive step, done once."""
    logits = wrapper.get_logits(np.expand_dims(feature_maps, axis=0), layer_name)[0]
    logits_baseline = wrapper.get_logits(
        np.expand_dims(tf.zeros(shape=feature_maps.shape), axis=0), layer_name
    )[0]
    ig_expected = tf.nn.relu(tf.subtract(logits, logits_baseline))
    max_val = tf.reduce_max(ig_expected)
    ig_expected_norm = ig_expected / max_val if max_val > 0 else ig_expected
    ig_expected_class = ig_expected_norm[class_idx]

    ig = compute_integrated_gradients(feature_maps, layer_name, class_idx)
    attributions = tf.nn.relu(tf.multiply(ig, feature_maps))
    attributions = tf.multiply(
        tf.divide(attributions, tf.add(tf.reduce_sum(attributions), 1e-10)),
        ig_expected_class
    )
    return attributions

def get_or_compute_class_layer_attributions(class_name, layer_name, test_fmaps):
    """Cached: (200, H, W, C) attribution tensors for every test image of
    this class at this layer -- the artifact every concept reuses."""
    cache_path = os.path.join(IG_CACHE_DIR, f"attrib_{class_name}_{layer_name}.joblib")
    if os.path.exists(cache_path):
        return load(cache_path)

    class_idx = wrapper.label_to_id(class_name)
    all_attributions = []
    t0 = time.time()
    for i, fm in enumerate(test_fmaps):
        fm_tf = tf.constant(fm, dtype=tf.float32)
        attributions = compute_shared_attributions(fm_tf, layer_name, class_idx)
        all_attributions.append(attributions.numpy())
        if (i + 1) % 50 == 0:
            print(f"    IG progress: {i+1}/{len(test_fmaps)} "
                  f"({time.time()-t0:.1f}s elapsed)")
    all_attributions = np.stack(all_attributions, axis=0)
    dump(all_attributions, cache_path, compress=3)
    return all_attributions


# ─────────────────────────────────────────────
# 6. VECTORIZED 51-DIRECTION APPLICATION (batched, not a Python loop)
# Verified numerically equivalent to the per-direction loop version.
# ─────────────────────────────────────────────
def apply_directions_batched(attributions, feature_maps, directions_stacked):
    """
    attributions:       (H,W,C) shared IG-based attribution map
    feature_maps:        (H,W,C) raw activations
    directions_stacked:  (51, C) real + 50 random directions
    Returns: (51,) attribution scores, one per direction
    """
    concept_maps = tf.nn.relu(
        tf.tensordot(directions_stacked, feature_maps, axes=[[1], [2]])
    )  # (51,H,W)

    masked = attributions[None, :, :, :] * concept_maps[:, :, :, None]  # (51,H,W,C)
    pooled_masked_attributions = tf.reduce_sum(masked, axis=(1, 2))  # (51,C)

    pooled_cav_norm = tf.nn.relu(directions_stacked)  # (51,C)
    max_per_direction = tf.reduce_max(pooled_cav_norm, axis=1, keepdims=True)
    max_per_direction = tf.where(
        max_per_direction > 0, max_per_direction, tf.ones_like(max_per_direction)
    )
    pooled_cav_norm = pooled_cav_norm / max_per_direction

    scores = tf.reduce_sum(pooled_cav_norm * pooled_masked_attributions, axis=1)  # (51,)
    return scores.numpy()


# ─────────────────────────────────────────────
# 7. TEST FEATURE MAPS (raw activations, reused across the whole run)
# ─────────────────────────────────────────────
def load_test_fmaps(class_name, layer_name):
    cache_path = os.path.join(CACHE_DIR, f"test_fmaps_{class_name}_{layer_name}.joblib")
    if os.path.exists(cache_path):
        return load(cache_path)
    gen = ImageActivationGenerator(
        model_wrapper=wrapper,
        concept_images_dir=TEST_DIR,
        cache_dir=CACHE_DIR,
        preprocessing_function=preprocess_resnet_v2,
        max_examples=MAX_EXAMPLES,
        resize_mode='pad',
    )
    fmaps = gen.get_feature_maps_for_concept(class_name, layer_name)
    dump(fmaps, cache_path, compress=3)
    return fmaps


# ─────────────────────────────────────────────
# 8. MAIN LOOP — layer outer, class next (IG shared across concepts),
# concept innermost (reuses that layer/class's IG results)
# ─────────────────────────────────────────────
ttest_results = {concept: {layer: {} for layer in LAYERS} for concept in CONCEPTS}

run_start = time.time()

for layer_name in LAYERS:
    print(f"\n{'='*60}\nLayer: {layer_name}\n{'='*60}")

    for class_name in CLASSES:
        print(f"\n  Class: {class_name}")
        test_fmaps = load_test_fmaps(class_name, layer_name)

        print(f"  Computing/loading shared IG attributions "
              f"({len(test_fmaps)} images)...")
        attributions_all = get_or_compute_class_layer_attributions(
            class_name, layer_name, test_fmaps
        )

        for concept_name, concept_folder in CONCEPTS.items():
            try:
                real_direction = load_real_direction(concept_folder, layer_name)
                random_dirs = build_random_directions(concept_folder, layer_name)
            except FileNotFoundError as e:
                print(f"    [SKIP] {concept_name}: {e}")
                continue

            directions_stacked = tf.constant(
                np.stack([real_direction] + random_dirs, axis=0), dtype=tf.float32
            )  # (51, C)

            all_scores = np.zeros((len(test_fmaps), N_RANDOM_CAVS + 1))
            for i in range(len(test_fmaps)):
                fm_tf = tf.constant(test_fmaps[i], dtype=tf.float32)
                attrib_tf = tf.constant(attributions_all[i], dtype=tf.float32)
                all_scores[i] = apply_directions_batched(attrib_tf, fm_tf, directions_stacked)

            real_scores = all_scores[:, 0]
            random_mean_scores = all_scores[:, 1:].mean(axis=0)  # (50,) -- mean per random direction

            if np.std(real_scores) == 0 and np.std(random_mean_scores) == 0:
                t_stat, p_value = float('nan'), float('nan')
            else:
                t_stat, p_value = stats.ttest_ind(
                    real_scores, random_mean_scores, equal_var=False
                )
            significant = (not np.isnan(p_value)) and (p_value < ALPHA)

            ttest_results[concept_name][layer_name][class_name] = {
                'mean': float(np.mean(real_scores)),
                'std': float(np.std(real_scores)),
                'rand_mean': float(np.mean(random_mean_scores)),
                'rand_std': float(np.std(random_mean_scores)),
                't_statistic': float(t_stat) if not np.isnan(t_stat) else None,
                'p_value': float(p_value) if not np.isnan(p_value) else None,
                'significant': bool(significant),
            }

            sig_str = "significant" if significant else "NOT significant"
            print(f"    {concept_name}: mean={np.mean(real_scores):.4f} "
                  f"p={p_value:.4f} {sig_str}")

print(f"\nTotal run time: {(time.time()-run_start)/3600:.2f} hours")

# ─────────────────────────────────────────────
# 9. SAVE RESULTS
# ─────────────────────────────────────────────
results_path = os.path.join(TTEST_DIR, "ttest_results_proper_ig.json")
with open(results_path, 'w') as f:
    json.dump(ttest_results, f, indent=2)
print(f"\nResults saved: {results_path}")

# ─────────────────────────────────────────────
# 10. SUMMARY TABLES
# ─────────────────────────────────────────────
for focus_class in CLASSES:
    print(f"\nClass: {focus_class}")
    table = PrettyTable(field_names=["Concept", "Mean", "p-value", "Significant"])
    for concept_name in CONCEPTS:
        r = ttest_results.get(concept_name, {}).get("post_relu", {}).get(focus_class, {})
        mean = r.get('mean', 0)
        pval = r.get('p_value', None)
        sig = r.get('significant', False)
        p_str = f"{pval:.4f}" if pval is not None else "nan"
        table.add_row([concept_name, f"{mean:.5f}", p_str, "yes" if sig else "no"])
    print(table)

print(f"\nAll done. Results in: {TTEST_DIR}")