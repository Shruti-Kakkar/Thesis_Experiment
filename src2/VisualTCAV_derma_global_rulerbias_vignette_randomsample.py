"""
VisualTCAV_derma_global_rulerbias_vignette_randomsample.py

PURPOSE: Tables 5.6 and 5.7 report GlobalVisualTCAV mean attribution
(with 95.45% CI) for ruler_present (pooled and 4 per-style CAVs) and
vignette_present, captioned as using the "full MEL and NV test sets"
(1327 and 2495 images respectively). That caption is false as written:
each producing script (VisualTCAV_derma_global_ruler_bias_matched.py,
VisualTCAV_derma_global_ruler_bias_by_category.py,
VisualTCAV_derma_global_vignette_matched.py) instantiates GlobalVisualTCAV
with max_examples=200, and GlobalVisualTCAV.computeFeatureMaps() loads
test images via ImageActivationGenerator.get_images_for_concept(), the
same un-shuffled filenames[:200] selection already found and corrected
for Tables 5.3/5.4 and the CAV negative-pool check. This re-runs the
same six GlobalVisualTCAV configurations with a true-random 200-image
test sample instead.

WHAT STAYS THE SAME: every CAV direction. All six MODEL_TAGs below have
an existing cached CAV (built from the ORIGINAL, not-yet-corrected
concept-image pool -- see Limitations, "Negative-Pool Sampling in CAV
Construction" and "Source Coverage of the Artifact Concept Images", both
knowingly left uncorrected). explain()'s cache_cav=True means
_compute_cavs() hits that existing cache and returns immediately,
without touching the monkey-patched image loader at all -- confirmed by
inspection: the cache path this script constructs matches the existing
cache file on disk for all six configurations. Only the TEST-image
selection step is corrected here.

HOW: ImageActivationGenerator._load_images_from_files is monkey-patched
on each run's own activation_generator INSTANCE only (never on the
class), to shuffle the file list with a seeded RNG before slicing to
max_examples. Since CAV construction never reaches this call (cache hit,
see above), the patch only ever affects the test-image loading path in
GlobalVisualTCAV.computeFeatureMaps().

SCOPE: post_relu only, matching Tables 5.6/5.7 (they report no other
layer), and MEL/NV only, matching the two classes those tables use.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import os
import types
import json
import numpy as np
from joblib import dump

PROJECT_ROOT = os.path.expanduser(
    "~/scratch/dev-uos/projects/VTCAV_Dermatology"
)
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "datasets", "test_images_by_class")

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from VisualTCAV import GlobalVisualTCAV, Model
from tensorflow.keras.applications.resnet_v2 import (
    preprocess_input as preprocess_resnet_v2
)

LAYERS = ["post_relu"]
CLASSES = ["MEL", "NV"]
CLASS_SEED = {"MEL": 45, "NV": 46}  # matches SEED_BY_CLASS elsewhere in this investigation

GRAPH_FILENAME = "resnet50v2_isic2019_final.keras"
LABELS_FILENAME = "isic2019_classes.txt"

# The six existing GlobalVisualTCAV configurations Tables 5.6/5.7 draw from.
RUNS = [
    {"label": "ruler_present (pooled)",     "model_tag": "padonly_seed2_rulerbias_matched",
     "concept_dir": "concept_images_ruler_matched",              "concept": "ruler_present/positive"},
    {"label": "ruler_present (thick_edge)", "model_tag": "padonly_seed2_rulerbias_matched_thick_edge",
     "concept_dir": "concept_images_ruler_matched_thick_edge",   "concept": "ruler_present/positive"},
    {"label": "ruler_present (lesion_side)","model_tag": "padonly_seed2_rulerbias_matched_lesion_side",
     "concept_dir": "concept_images_ruler_matched_lesion_side",  "concept": "ruler_present/positive"},
    {"label": "ruler_present (ticks_edge)", "model_tag": "padonly_seed2_rulerbias_matched_ticks_edge",
     "concept_dir": "concept_images_ruler_matched_ticks_edge",   "concept": "ruler_present/positive"},
    {"label": "ruler_present (short_ruler)","model_tag": "padonly_seed2_rulerbias_matched_short_ruler",
     "concept_dir": "concept_images_ruler_matched_short_ruler",  "concept": "ruler_present/positive"},
    {"label": "vignette_present",           "model_tag": "padonly_seed2_vignette_matched",
     "concept_dir": "concept_images_vignette_matched",           "concept": "vignette_present/positive"},
]

CHECK_DIR = os.path.join(PROJECT_ROOT, "outputs2", "global_attribution_randomsample_check")
os.makedirs(CHECK_DIR, exist_ok=True)


def make_random_loader(seed):
    """Returns a replacement for ImageActivationGenerator._load_images_from_files
    that shuffles filenames with a seeded RNG before slicing to max_imgs,
    in place of the original's un-shuffled filenames[:max_imgs]."""
    def _load_images_from_files_random(self, filenames, max_imgs=500, shape=(224, 224), preprocess=True):
        rng = np.random.default_rng(seed)
        shuffled = list(filenames)
        rng.shuffle(shuffled)
        print(f"    [random loader] {len(filenames)} files available; "
              f"true-random seed={seed} sample of {min(max_imgs, len(shuffled))} drawn")
        return self.__class__._load_images_from_files_original(self, shuffled, max_imgs, shape, preprocess)
    return _load_images_from_files_random


results = {}

for run in RUNS:
    label = run["label"]
    model_tag = run["model_tag"]
    print(f"\n{'='*60}\n{label}  [{model_tag}]\n{'='*60}")

    vtcav_dir = os.path.join(PROJECT_ROOT, "outputs2", f"vtcav_{model_tag}")
    models_dir = os.path.join(vtcav_dir, "models")
    cache_dir = os.path.join(vtcav_dir, "cache")
    concept_dir = os.path.join(PROJECT_ROOT, run["concept_dir"])

    results[label] = {}

    for target_class in CLASSES:
        global_visual_tcav = GlobalVisualTCAV(
            test_images_folder=target_class,
            target_class=target_class,
            m_steps=50,
            batch_size=20,
            n_cav_runs=20,
            model=Model(
                model_name="resnet50v2",
                graph_path_filename=GRAPH_FILENAME,
                label_path_filename=LABELS_FILENAME,
                preprocessing_function=preprocess_resnet_v2,
                max_examples=200,
                resize_mode='pad',
            ),
            models_dir=models_dir,
            cache_dir=cache_dir,
            test_images_dir=TEST_IMAGES_DIR,
            concept_images_dir=concept_dir,
            negative_suffix="negative",
        )

        global_visual_tcav.setLayers(layer_names=LAYERS)
        global_visual_tcav.setConcepts(concept_names=[run["concept"]])

        # Monkey-patch the true-random loader onto THIS instance's activation
        # generator only. CAV construction hits the existing cache before
        # ever calling this (verified: cache_cav=True, cache path matches
        # what's already on disk), so this only affects test-image loading.
        gen = global_visual_tcav.model.activation_generator
        gen.__class__._load_images_from_files_original = gen.__class__._load_images_from_files
        gen._load_images_from_files = types.MethodType(
            make_random_loader(CLASS_SEED[target_class]), gen
        )

        global_visual_tcav.explain(cache_cav=True, cache_random=False)

        stat = global_visual_tcav.stats["post_relu"][run["concept"]]
        mean, begin, end = float(stat.mean), float(stat.begin), float(stat.end)
        print(f"  [{target_class}] mean={mean:.4e}  95.45% CI=[{begin:.4e}, {end:.4e}]")

        results[label][target_class] = {"mean": mean, "ci_begin": begin, "ci_end": end}

with open(os.path.join(CHECK_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
dump(results, os.path.join(CHECK_DIR, "results.joblib"))

print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
for label, r in results.items():
    print(f"{label}")
    for target_class, cr in r.items():
        print(f"    [{target_class}] mean={cr['mean']:.4e}  CI=[{cr['ci_begin']:.4e}, {cr['ci_end']:.4e}]")

print(f"\nAll done. Results in: {CHECK_DIR}")
