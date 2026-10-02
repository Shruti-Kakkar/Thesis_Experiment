"""
check_cav_negative_pool_sampling.py

PURPOSE: 6 of the 10 real derm7pt concepts have a "negative" (concept-
absent) training-image folder larger than MAX_EXAMPLES=200: streaks_irregular
(273), streaks_regular (273), vascular_structures (347), blue_whitish_veil
(339), pigmentation (253), regression_structures (317). CAV construction
loads these through ImageActivationGenerator.get_images_for_concept(), the
same un-shuffled `filenames[:200]` selection already found and fixed for
test images (Tables 5.3/5.4) and for the ruler_present/vignette_present
CAV pools (Limitations, "Source Coverage of the Artifact Concept Images").
This script checks whether it matters here too: not just which images get
*scored* (the earlier fix), but which images the CAV *direction itself* is
built from.

SCOPE: only post_relu, the one validated layer (Section 5.6.1's RQ1
result). CAVs are per-layer; the three degenerate layers' attribution is
already near-fully zero for structural reasons unrelated to CAV quality,
so a shifted CAV direction there would not change any reported finding.

Each concept's positive folder, and each hard-negative concept's extra
negative source (the sibling regularity variant's positive folder, for
streaks_regular/streaks_irregular), are all under 200 images already
(verified separately) -- filenames[:200] there already is the full set,
so only the default "absent" folder needs correcting here.

METHOD:
  1. Load the existing cached CAV direction (built from the biased
     filenames[:200] negative sample) for each of the 6 concepts.
  2. Rebuild the CAV using a true-random, seeded 200-image draw from the
     full negative pool instead, replicating VisualTCAV._compute_cavs's
     exact construction (centroid-difference, 20 runs, 80/20 split,
     seed=42) so the only variable that changes is which images the
     negative centroid is computed from.
  3. Compare the two directions by cosine similarity.
  4. Score both directions against the SAME already-cached, correctly
     true-random-sampled test-image IG attributions (from the Table
     5.3/5.4 fix) to see whether mean attribution toward each concept's
     Table 5.8 target class actually shifts downstream.

CAVs are NOT recomputed for the positive side, and the biased CAV
directions are read from the existing Global run's cache, not rebuilt --
only the negative side of the corrected CAV is freshly computed.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import os
import numpy as np
from joblib import load, dump
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_TAG = "padonly_seed2_hardneg"
VTCAV_DIR = os.path.join(PROJECT_ROOT, "outputs2", f"vtcav_{MODEL_TAG}")
MODELS_DIR = os.path.join(VTCAV_DIR, "models")
CACHE_DIR = os.path.join(VTCAV_DIR, "cache", "resnet50v2")
CONCEPT_DIR = os.path.join(PROJECT_ROOT, "concept_images")

RANDOMSAMPLE_TTEST_DIR = os.path.join(
    PROJECT_ROOT, "outputs2", f"vtcav_ttest_proper_ig_{MODEL_TAG}_randomsample"
)
IG_CACHE_DIR = os.path.join(RANDOMSAMPLE_TTEST_DIR, "ig_cache")
FMAP_CACHE_DIR = os.path.join(RANDOMSAMPLE_TTEST_DIR, "fmap_cache")

CHECK_DIR = os.path.join(PROJECT_ROOT, "outputs2", "cav_negative_pool_sampling_check")
os.makedirs(CHECK_DIR, exist_ok=True)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from VisualTCAV import KerasModelWrapper, ImageActivationGenerator
from tensorflow.keras.applications.resnet_v2 import (
    preprocess_input as preprocess_resnet_v2
)

MAX_EXAMPLES = 200
N_CAV_RUNS = 20
CAV_SEED = 42
LAYER = "post_relu"
CLASS_SEED_MAP = {"MEL": 45, "NV": 46}  # match randomsample re-run's SEED_BY_CLASS

# The 6 at-risk concepts, their Table 5.8 target class(es) to check
# downstream impact on, and whether they use hard-negative augmentation.
CONCEPTS = {
    "streaks_regular":       {"target_classes": ["NV"],  "extra_negatives": ["streaks_irregular/positive"]},
    "streaks_irregular":     {"target_classes": ["MEL"], "extra_negatives": ["streaks_regular/positive"]},
    "blue_whitish_veil":     {"target_classes": ["MEL"], "extra_negatives": []},
    "pigmentation":          {"target_classes": ["MEL"], "extra_negatives": []},
    "regression_structures": {"target_classes": ["MEL"], "extra_negatives": []},
    "vascular_structures":   {"target_classes": ["MEL", "NV"], "extra_negatives": []},
}

print("=" * 60)
print(f"CAV negative-pool sampling check [{MODEL_TAG}], layer={LAYER}")
print("=" * 60)

wrapper = KerasModelWrapper(
    os.path.join(MODELS_DIR, "resnet50v2", "resnet50v2_isic2019_final.keras"),
    os.path.join(MODELS_DIR, "resnet50v2", "isic2019_classes.txt"),
    batch_size=20,
    model_name="resnet50v2",
)
print(f"Model loaded: {wrapper.model_name}\n")


def cav_cache_path(concept_root, extra_negatives, layer_name):
    extra_tag = ""
    if extra_negatives:
        extra_safe = "_".join(sorted(f.replace('/', '-') for f in extra_negatives))
        extra_tag = f"_plus_{extra_safe}"
    return os.path.join(
        CACHE_DIR,
        f'cav_{concept_root}_positive{extra_tag}_{MAX_EXAMPLES}_neg_{N_CAV_RUNS}runs_{layer_name}.joblib'
    )


def load_biased_direction(concept_root, extra_negatives, layer_name):
    path = cav_cache_path(concept_root, extra_negatives, layer_name)
    concept_layer = load(path)
    direction = concept_layer.cav.direction
    emblem = concept_layer.cav.concept_emblem
    emblem = emblem.numpy() if hasattr(emblem, 'numpy') else np.asarray(emblem)
    return (direction.numpy() if hasattr(direction, 'numpy') else np.asarray(direction)), emblem


def get_random_negative_activations(concept_root, extra_negatives, layer_name, seed):
    """Mirrors VisualTCAV._compute_negative_activations's concatenation of
    the default 'absent' folder with any extra hard-negative sources, but
    draws the default folder via a true-random shuffle instead of
    filenames[:max_imgs]. Extra sources are all under MAX_EXAMPLES already
    (verified separately), so they are loaded unchanged via the existing
    (unbiased in this case) pipeline."""

    def is_image(filename):
        for ext in ["jpg", "jpeg", "png", "gif", "bmp"]:
            if filename.lower().endswith(ext):
                return True
        return False

    negative_folder = f"{concept_root}/negative"
    concept_dir = os.path.join(CONCEPT_DIR, negative_folder)
    img_paths = [os.path.join(concept_dir, d) for d in tf.io.gfile.listdir(concept_dir) if is_image(d)]
    rng = np.random.default_rng(seed)
    shuffled = list(img_paths)
    rng.shuffle(shuffled)
    print(f"    [{concept_root}] {len(img_paths)} total negative images available; "
          f"true-random seed={seed} sample of {min(MAX_EXAMPLES, len(shuffled))} drawn")

    gen = ImageActivationGenerator(
        model_wrapper=wrapper,
        concept_images_dir=CONCEPT_DIR,
        cache_dir=None,
        preprocessing_function=preprocess_resnet_v2,
        max_examples=MAX_EXAMPLES,
        resize_mode='pad',
    )
    imgs = gen._load_images_from_files(
        shuffled, MAX_EXAMPLES, shape=wrapper.get_image_shape()[:2], preprocess=True,
    )
    negative_acts = wrapper.get_feature_maps(imgs, layer_name)

    for extra_folder in extra_negatives:
        extra_acts = gen.get_feature_maps_for_concept(extra_folder, layer_name)
        negative_acts = np.concatenate([negative_acts, extra_acts], axis=0)

    return negative_acts


def build_corrected_cav(concept_root, extra_negatives, layer_name):
    """Replicates VisualTCAV._compute_cavs's exact math (centroid
    difference, N_CAV_RUNS runs, 80/20 split, seed=CAV_SEED), with only
    the negative-activation source swapped for a true-random sample."""
    gen = ImageActivationGenerator(
        model_wrapper=wrapper,
        concept_images_dir=CONCEPT_DIR,
        cache_dir=None,
        preprocessing_function=preprocess_resnet_v2,
        max_examples=MAX_EXAMPLES,
        resize_mode='pad',
    )
    concept_acts = gen.get_feature_maps_for_concept(f"{concept_root}/positive", layer_name)
    negative_acts = get_random_negative_activations(concept_root, extra_negatives, layer_name, seed=CAV_SEED)

    pooled_pos = tf.reduce_mean(concept_acts, axis=(1, 2)).numpy()
    pooled_neg = tf.reduce_mean(negative_acts, axis=(1, 2)).numpy()

    n_pos, n_neg = len(pooled_pos), len(pooled_neg)
    n_min = min(n_pos, n_neg)

    directions = []
    for run in range(N_CAV_RUNS):
        run_rng = np.random.default_rng(CAV_SEED + run)
        pos_idx = run_rng.permutation(n_pos)[:n_min]
        neg_idx = run_rng.permutation(n_neg)[:n_min]
        pos_run = pooled_pos[pos_idx]
        neg_run = pooled_neg[neg_idx]

        n_train = int(n_min * 0.8)
        pos_train = pos_run[:n_train]
        neg_train = neg_run[:n_train]

        c0 = tf.reduce_mean(pos_train, axis=0)
        c1 = tf.reduce_mean(neg_train, axis=0)
        direction = tf.subtract(c0, c1)
        directions.append(direction)

    mean_direction = tf.reduce_mean(directions, axis=0).numpy()
    return mean_direction


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def apply_direction(attributions, feature_maps, direction, eps_plus, eps_minus):
    """Same math as apply_directions_batched in the ttest scripts (including
    the emblem-check clip-and-rescale fix), for a single direction instead
    of a stacked batch of 51. eps_plus/eps_minus come from the biased CAV's
    own concept_emblem, applied to both the biased and the corrected
    direction so the comparison stays apples-to-apples (same rationale as
    the ttest script's fix)."""
    concept_map = tf.nn.relu(tf.tensordot(direction, feature_maps, axes=[[0], [2]]))  # (H,W)
    if eps_plus > eps_minus:
        concept_map = tf.clip_by_value(concept_map, eps_minus, eps_plus)
        concept_map = (concept_map - eps_minus) / (eps_plus - eps_minus)
    else:
        concept_map = tf.zeros_like(concept_map)
    masked = attributions * concept_map[:, :, None]  # (H,W,C)
    pooled_masked = tf.reduce_sum(masked, axis=(0, 1))  # (C,)

    pooled_cav_norm = tf.nn.relu(direction)
    max_val = tf.reduce_max(pooled_cav_norm)
    if max_val > 0:
        pooled_cav_norm = pooled_cav_norm / max_val

    score = tf.reduce_sum(pooled_cav_norm * pooled_masked)
    return float(score.numpy())


def mean_attribution_for_class(direction, target_class, eps_plus, eps_minus):
    attrib_path = os.path.join(IG_CACHE_DIR, f"attrib_{target_class}_{LAYER}.joblib")
    seed = CLASS_SEED_MAP[target_class]
    fmap_path = os.path.join(FMAP_CACHE_DIR, f"test_fmaps_randomseed{seed}_{target_class}_{LAYER}.joblib")
    attributions_all = load(attrib_path)
    test_fmaps = load(fmap_path)

    direction_tf = tf.constant(direction, dtype=tf.float32)
    scores = []
    for i in range(len(test_fmaps)):
        fm_tf = tf.constant(test_fmaps[i], dtype=tf.float32)
        attrib_tf = tf.constant(attributions_all[i], dtype=tf.float32)
        scores.append(apply_direction(attrib_tf, fm_tf, direction_tf, eps_plus, eps_minus))
    return float(np.mean(scores))


# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
results = {}

for concept_root, cfg in CONCEPTS.items():
    print(f"\n{'='*60}\nConcept: {concept_root}\n{'='*60}")

    biased_dir, emblem = load_biased_direction(concept_root, cfg["extra_negatives"], LAYER)
    eps_plus, eps_minus = float(emblem[0]), float(emblem[1])
    corrected_dir = build_corrected_cav(concept_root, cfg["extra_negatives"], LAYER)

    sim = cosine_sim(biased_dir, corrected_dir)
    print(f"  Cosine similarity (biased CAV vs. random-negative-sample CAV): {sim:.4f}")

    class_results = {}
    for target_class in cfg["target_classes"]:
        mean_biased = mean_attribution_for_class(biased_dir, target_class, eps_plus, eps_minus)
        mean_corrected = mean_attribution_for_class(corrected_dir, target_class, eps_plus, eps_minus)
        pct_change = 100 * (mean_corrected - mean_biased) / mean_biased if mean_biased != 0 else float('nan')
        print(f"  [{target_class}] mean attribution: biased={mean_biased:.5f} "
              f"corrected={mean_corrected:.5f} ({pct_change:+.1f}%)")
        class_results[target_class] = {
            "mean_biased": mean_biased,
            "mean_corrected": mean_corrected,
            "pct_change": pct_change,
        }

    results[concept_root] = {
        "cosine_similarity": sim,
        "class_results": class_results,
    }

dump(results, os.path.join(CHECK_DIR, "results.joblib"))
import json
with open(os.path.join(CHECK_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
for concept_root, r in results.items():
    print(f"{concept_root:28s} cosine_sim={r['cosine_similarity']:.4f}")
    for target_class, cr in r["class_results"].items():
        print(f"    [{target_class}] {cr['mean_biased']:.5f} -> {cr['mean_corrected']:.5f} ({cr['pct_change']:+.1f}%)")

print(f"\nAll done. Results in: {CHECK_DIR}")
