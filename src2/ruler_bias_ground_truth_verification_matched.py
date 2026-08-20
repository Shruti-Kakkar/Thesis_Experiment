"""
ruler_bias_ground_truth_verification_matched.py
Scoring stage: verifies the corrected 'ruler_present' CAV's per-image
attribution against hand-sorted ground truth -- does attribution
actually track whether an image has a physical ruler in it? Run
ruler_bias_ground_truth_finalize_matched.py afterwards for the
group comparison and Mann-Whitney U test.

GlobalVisualTCAV only reports an aggregate mean per class -- no per-image
number to check against ground truth. This uses the same per-image
machinery as ruler_bias_local_screening_matched.py, but scores
datasets/ruler_sorted_test/{MEL,NV}_test_{clean,ruler}/ (400 images:
145 MEL clean, 55 MEL ruler, 189 NV clean, 11 NV ruler) instead of an
unlabeled sample -- a held-out, hand-labeled subset of the same
test_images_by_class images the Global/Local matched scripts explain,
confirmed by filename overlap.

Reuses the SAME CAV cache as VisualTCAV_derma_global_ruler_bias_matched.py
(MODEL_TAG=padonly_seed2_rulerbias_matched, n_cav_runs=20,
max_examples=200) -- does not retrain it, and test_images_dir stays
TEST_IMAGES_DIR (test_images_by_class) since that's where these same
files live for the model wrapper.

Each image reloads the full ResNet50V2 model, and that GPU memory isn't
released between images -- across all 400 images this will exhaust a
shared GPU partway through. Run in disjoint chunks, same pattern as
ruler_bias_local_screening.py:

    python ruler_bias_ground_truth_verification_matched.py --start 0   --end 100
    python ruler_bias_ground_truth_verification_matched.py --start 100 --end 200
    python ruler_bias_ground_truth_verification_matched.py --start 200 --end 300
    python ruler_bias_ground_truth_verification_matched.py --start 300 --end 400

Each image's row is appended to RESULTS_CSV as soon as it's computed, so
a chunk that crashes partway through keeps what it already finished --
rerun that chunk's remaining range to fill the gap. Keep --start/--end
ranges disjoint across runs to avoid duplicate rows (the finalize script
does not deduplicate).

post_relu ONLY, same layer as the local screening pipeline.

Author: Shruti Kakkar
"""

import argparse
import csv
import os

from ruler_bias_common_matched import make_local_tcav, PROJECT_ROOT

RULER_GT_DIR = os.path.join(PROJECT_ROOT, "datasets", "ruler_sorted_test")
GT_FOLDERS = [
    ("MEL", "MEL_test_clean", False),
    ("MEL", "MEL_test_ruler", True),
    ("NV",  "NV_test_clean",  False),
    ("NV",  "NV_test_ruler",  True),
]

RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs2", "ruler_bias_ground_truth_verification_matched")
RESULTS_CSV = os.path.join(RESULTS_DIR, "gt_attributions.csv")
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_gt_targets():
    targets = []
    for true_class, subfolder, has_ruler in GT_FOLDERS:
        folder = os.path.join(RULER_GT_DIR, subfolder)
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                targets.append((true_class, fname, has_ruler))
    return targets


print("VisualTCAV imported successfully.\n")

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=None,
                     help="exclusive; defaults to the full target list")
args = parser.parse_args()

targets = build_gt_targets()
end = args.end if args.end is not None else len(targets)
chunk = targets[args.start:end]

print(f"Ground-truth verification [{args.start}:{end}] of {len(targets)} total "
      f"({sum(1 for _, _, r in targets if r)} ruler, "
      f"{sum(1 for _, _, r in targets if not r)} clean), post_relu only\n")

write_header = not os.path.exists(RESULTS_CSV)
with open(RESULTS_CSV, 'a', newline='') as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(['class', 'filename', 'has_ruler_gt', 'attribution'])

    for i, (true_class, fname, has_ruler) in enumerate(chunk):
        # same underlying file, addressed via test_images_by_class/<class>/
        rel_path = os.path.join(true_class, fname)

        local_tcav = make_local_tcav(true_class, rel_path)
        local_tcav.predict()
        local_tcav.explain(cache_cav=True, cache_random=True, n_cav_runs=20)

        attribution = float(
            local_tcav.computations["post_relu"]["ruler_present/positive"].attributions[0]
        )
        writer.writerow([true_class, fname, has_ruler, f"{attribution:.6f}"])
        f.flush()
        print(f"  [{args.start + i + 1}/{len(targets)}] {true_class}/{fname} "
              f"(ruler_gt={has_ruler}): attribution={attribution:.5f}")

print(f"\nChunk done. Results appended to: {RESULTS_CSV}")
