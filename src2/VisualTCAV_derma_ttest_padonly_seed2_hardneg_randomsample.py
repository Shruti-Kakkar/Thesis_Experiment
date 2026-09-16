"""
VisualTCAV_derma_ttest_padonly_seed2_hardneg_randomsample.py

Full re-run of VisualTCAV_derma_ttest_padonly_seed2_hardneg.py (the
script that produced the thesis's current Table 5.4 numbers), with the one
deliberate change validated by VisualTCAV_derma_ttest_padonly_seed2_hardneg_
randomsample_5class_post_relu_check.py first: test-image selection uses a seeded true-random sample of
200 images per class, in place of ImageActivationGenerator's un-shuffled
`filenames[:max_imgs]` -- the same non-random-sampling bug already found
and fixed in the local-screening pipeline (Section 5.6.1).

The faster post_relu-only, 5-class check already showed this is not a
no-op: 0/50 zero-attribution cells either way (Table 5.3's RQ1 pattern is
robust), but Table 5.4's finer-grained numbers do move -- 44/50 to 41/50
significant cells at post_relu alone, 5 outright significance flips, and
mean-attribution shifts up to 0.40 on some concept/class pairs. This
script covers all 8 classes and all 4 layers to produce the full,
corrected replacement for Table 5.4 (and, downstream, Tables 5.6/5.7).

For DF/SCC/VASC (each under 200 total test images), a "random sample of
200" is the full available set either way, so results there are expected
to match the original run's up to floating-point/ordering noise; only
AK/BCC/BKL/MEL/NV can genuinely differ.

Estimated runtime: comparable to the original script, ~2.5 hours.

CAVs themselves are NOT recomputed here, for the same reason as the
check script: they are built from each concept's own training images,
not the test set this script resamples, so the existing cached CAVs from
the hardneg Global run remain valid to reuse as-is.

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
# 0. MODEL VARIANT TAG — which Global run's cached CAVs to read from.
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

# Separate output/cache tree from both the original filenames[:200] run and
# the smaller post_relu-only check, so none of the three overwrite each other.
TTEST_DIR    = os.path.join(PROJECT_ROOT, "outputs2", f"vtcav_ttest_proper_ig_{MODEL_TAG}_randomsample")
IG_CACHE_DIR = os.path.join(TTEST_DIR, "ig_cache")
FMAP_CACHE_DIR = os.path.join(TTEST_DIR, "fmap_cache")

os.makedirs(TTEST_DIR, exist_ok=True)
os.makedirs(IG_CACHE_DIR, exist_ok=True)
os.makedirs(FMAP_CACHE_DIR, exist_ok=True)

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

# Fixed per-class seeds (not position-based) so they stay stable regardless
# of CLASSES' order. AK/BCC/BKL/MEL/NV match the seeds used by
# VisualTCAV_derma_ttest_padonly_seed2_hardneg_randomsample_5class_post_relu_check.py.
SEED_BY_CLASS = {
    'AK': 42, 'BCC': 43, 'BKL': 44, 'MEL': 45, 'NV': 46,
    'DF': 47, 'VASC': 48, 'SCC': 49,
}

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
print(f"Visual-TCAV PROPER (IG-based) Significance Testing, TRUE-RANDOM SAMPLE [{MODEL_TAG}]")
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
# _compute_negative_activations EXACTLY, so we correctly locate the
# already-cached CAVs from the hardneg Global run. CAVs are built from
# concept TRAINING images, not the test set this script resamples, so
# reusing them as-is is valid.
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
# image/class/layer, reused across all 10 concepts) — unchanged from the
# authoritative script.
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
    """Cached: (N, H, W, C) attribution tensors for every resampled test
    image of this class at this layer -- the artifact every concept reuses."""
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
# 6. VECTORIZED 51-DIRECTION APPLICATION — unchanged.
# ─────────────────────────────────────────────
def apply_directions_batched(attributions, feature_maps, directions_stacked):
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
# 7. TEST FEATURE MAPS — THE ONE DELIBERATE CHANGE FROM THE AUTHORITATIVE
# SCRIPT. Instead of ImageActivationGenerator.get_images_for_concept()'s
# `tf.io.gfile.listdir()` -> `filenames[:max_imgs]` (no shuffle), this
# lists every test image for the class, shuffles with a seeded RNG, and
# only then takes the first MAX_EXAMPLES -- a true random sample instead
# of whatever order the filesystem happens to return.
# ─────────────────────────────────────────────
def get_test_images_random(gen, class_name, seed):
    def is_image(filename):
        for ext in ["jpg", "jpeg", "png", "gif", "bmp"]:
            if filename.lower().endswith(ext):
                return True
        return False

    class_dir = os.path.join(TEST_DIR, class_name)
    img_paths = [os.path.join(class_dir, d) for d in tf.io.gfile.listdir(class_dir) if is_image(d)]
    rng = np.random.default_rng(seed)
    shuffled = list(img_paths)
    rng.shuffle(shuffled)
    print(f"    [{class_name}] {len(img_paths)} total test images available; "
          f"true-random seed={seed} sample of {min(MAX_EXAMPLES, len(shuffled))} drawn")
    imgs = gen._load_images_from_files(
        shuffled,
        MAX_EXAMPLES,
        shape=gen.model_wrapper.get_image_shape()[:2],
        preprocess=True,
    )
    return imgs

def load_test_fmaps_random(class_name, layer_name):
    seed = SEED_BY_CLASS[class_name]
    cache_path = os.path.join(FMAP_CACHE_DIR, f"test_fmaps_randomseed{seed}_{class_name}_{layer_name}.joblib")
    if os.path.exists(cache_path):
        return load(cache_path)
    gen = ImageActivationGenerator(
        model_wrapper=wrapper,
        concept_images_dir=TEST_DIR,
        cache_dir=FMAP_CACHE_DIR,
        preprocessing_function=preprocess_resnet_v2,
        max_examples=MAX_EXAMPLES,
        resize_mode='pad',
    )
    imgs = get_test_images_random(gen, class_name, seed)
    fmaps = wrapper.get_feature_maps(imgs, layer_name)
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
        print(f"\n  Class: {class_name} (sample seed={SEED_BY_CLASS[class_name]})")
        test_fmaps = load_test_fmaps_random(class_name, layer_name)

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
                'zero_attribution': bool(np.mean(real_scores) == 0.0),
            }

            sig_str = "significant" if significant else "NOT significant"
            print(f"    {concept_name}: mean={np.mean(real_scores):.4f} "
                  f"p={p_value:.4f} {sig_str}")

print(f"\nTotal run time: {(time.time()-run_start)/3600:.2f} hours")

# ─────────────────────────────────────────────
# 9. SAVE RESULTS
# ─────────────────────────────────────────────
results_path = os.path.join(TTEST_DIR, "ttest_results_proper_ig_randomsample.json")
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
