"""
vignette_padding_confound_test.py
Quick diagnostic: does the 'vignette_present' CAV's local attribution on
real test images correlate with whether the image needed letterbox
padding (resize_mode='pad') rather than with any real photographic
vignette in the photo?

The vast majority of test images are 600x450 (non-square -> padded to
square, landing black bars in all four corners -- visually similar to a
synthetic vignette). A small minority (~2%) are naturally 1024x1024
(square -> no padding needed). This scores a handful of the rare square
images and compares them to already-scored non-square images from
vignette_bias_local_screening_matched.py's attributions.csv.

Caveat (see conversation): square images are rare enough that they may
come from a different acquisition source than the 600x450 majority, so
a difference found here could reflect that too, not padding alone. This
is a first-pass diagnostic, not a fully clean causal test.

Chunked like vignette_bias_local_screening_matched.py for the same
reason (per-image model reload leaks GPU memory).

    python vignette_padding_confound_test.py --start 0 --end 5
    python vignette_padding_confound_test.py --start 5 --end 10

Author: Shruti Kakkar
"""

import argparse
import csv
import os

from vignette_bias_common_matched import TEST_IMAGES_DIR, make_local_tcav

SQUARE_TARGETS = [
    ("MEL", "ISIC_0053460.jpg"), ("MEL", "ISIC_0053481.jpg"),
    ("MEL", "ISIC_0053489.jpg"), ("MEL", "ISIC_0053495.jpg"),
    ("MEL", "ISIC_0053512.jpg"),
    ("NV", "ISIC_0053453.jpg"), ("NV", "ISIC_0053458.jpg"),
    ("NV", "ISIC_0053477.jpg"), ("NV", "ISIC_0053480.jpg"),
    ("NV", "ISIC_0053490.jpg"),
]

RESULTS_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "outputs2", "vignette_bias_local_screening_matched",
    "padding_confound_square_attributions.csv",
)
RESULTS_CSV = os.path.normpath(RESULTS_CSV)

print("VisualTCAV imported successfully.\n")

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=None)
args = parser.parse_args()

end = args.end if args.end is not None else len(SQUARE_TARGETS)
chunk = SQUARE_TARGETS[args.start:end]

print(f"Scoring square (no-padding) images [{args.start}:{end}] of "
      f"{len(SQUARE_TARGETS)} total\n")

write_header = not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0
with open(RESULTS_CSV, 'a', newline='') as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(['class', 'filename', 'attribution'])

    for i, (true_class, fname) in enumerate(chunk):
        rel_path = os.path.join(true_class, fname)

        local_tcav = make_local_tcav(true_class, rel_path)
        local_tcav.predict()
        local_tcav.explain(cache_cav=True, cache_random=True, n_cav_runs=20)

        attribution = float(
            local_tcav.computations["post_relu"]["vignette_present/positive"].attributions[0]
        )
        writer.writerow([true_class, fname, f"{attribution:.6f}"])
        f.flush()
        print(f"  [{args.start + i + 1}/{len(SQUARE_TARGETS)}] {true_class}/{fname}: "
              f"attribution={attribution:.5f}")

print(f"\nChunk done. Results appended to: {RESULTS_CSV}")
