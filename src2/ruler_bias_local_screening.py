"""
ruler_bias_local_screening.py
Scoring stage of the ruler-bias local screen. Run ruler_bias_finalize.py
afterwards for the summary statistics and top-outlier heatmaps.

Each image reloads the full ResNet50V2 model, and that GPU memory isn't
released between images -- across the full 50-image sweep this reliably
exhausts a shared GPU partway through. Run this in disjoint index chunks
instead of all at once so each process exits (and releases its GPU
memory) before the next chunk starts, e.g.:

    python ruler_bias_local_screening.py --start 0  --end 10
    python ruler_bias_local_screening.py --start 10 --end 20
    python ruler_bias_local_screening.py --start 20 --end 30
    python ruler_bias_local_screening.py --start 30 --end 40
    python ruler_bias_local_screening.py --start 40 --end 50

Each image's attribution is appended to RESULTS_CSV as soon as it's
computed, so a chunk that crashes partway through still keeps the rows
it already finished -- rerun that chunk's remaining range to fill the
gap. Keep --start/--end ranges disjoint across runs to avoid duplicate
rows (ruler_bias_finalize.py does not deduplicate).

post_relu ONLY (the layer with real signal, per the Global/Local vignette-
confound findings) -- keeps this fast.

Author: Shruti Kakkar
"""

import argparse
import csv
import os

from ruler_bias_common import build_targets, make_local_tcav, RESULTS_CSV

print("VisualTCAV imported successfully.\n")

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=None,
                     help="exclusive; defaults to the full target list")
args = parser.parse_args()

targets = build_targets()
end = args.end if args.end is not None else len(targets)
chunk = targets[args.start:end]

print(f"Screening images [{args.start}:{end}] of {len(targets)} total, "
      f"post_relu only\n")

write_header = not os.path.exists(RESULTS_CSV)
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
            local_tcav.computations["post_relu"]["ruler_present/positive"].attributions[0]
        )
        writer.writerow([true_class, fname, f"{attribution:.6f}"])
        f.flush()
        print(f"  [{args.start + i + 1}/{len(targets)}] {true_class}/{fname}: "
              f"attribution={attribution:.5f}")

print(f"\nChunk done. Results appended to: {RESULTS_CSV}")
