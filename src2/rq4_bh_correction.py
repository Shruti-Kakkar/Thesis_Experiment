"""
rq4_bh_correction.py

PURPOSE: the Benjamini-Hochberg correction in the RQ4 calibration table
(tab:rq4_bh, Section 5.6.3): BH-adjusted p-values (alpha=0.05) across the
nine concepts with a well-defined present/absent split
(vascular_structures excluded).

Reads each concept's raw one-sided Mann-Whitney p-value from its
calibration summary (concept_calibration.py for eight concepts,
bwv_calibration.py for blue_whitish_veil) and writes
outputs2/concept_calibration_bh_correction.csv. The table's AUC,
threshold/bootstrap CI, rank-biserial and Fisher columns come straight
from those same summaries. No model compute.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import os
import re
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs2")
ALPHA = 0.05

# concept -> (summary file, target class)
SUMMARIES = {
    "streaks_regular":             ("streaks_regular_calibration/streaks_regular_calibration_summary.txt", "NV"),
    "blue_whitish_veil":           ("bwv_calibration/bwv_calibration_summary.txt", "MEL"),
    "pigment_network_typical":     ("pigment_network_typical_calibration/pigment_network_typical_calibration_summary.txt", "NV"),
    "streaks_irregular":           ("streaks_irregular_calibration/streaks_irregular_calibration_summary.txt", "MEL"),
    "pigmentation":                ("pigmentation_calibration/pigmentation_calibration_summary.txt", "MEL"),
    "dots_and_globules_regular":   ("dots_and_globules_regular_calibration/dots_and_globules_regular_calibration_summary.txt", "NV"),
    "pigment_network_atypical":    ("pigment_network_atypical_calibration/pigment_network_atypical_calibration_summary.txt", "MEL"),
    "regression_structures":       ("regression_structures_calibration/regression_structures_calibration_summary.txt", "MEL"),
    "dots_and_globules_irregular": ("dots_and_globules_irregular_calibration/dots_and_globules_irregular_calibration_summary.txt", "MEL"),
}

OUT_PATH = os.path.join(OUTPUTS_DIR, "concept_calibration_bh_correction.csv")


def raw_p(summary_path):
    text = open(os.path.join(OUTPUTS_DIR, summary_path)).read()
    return float(re.search(r"Mann-Whitney U test.*?p=([0-9.eE+-]+)", text).group(1))


rows = [{"concept": c, "target_class": cls, "p_raw": raw_p(path)}
        for c, (path, cls) in SUMMARIES.items()]
df = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
m = len(df)
df.insert(0, "rank", range(1, m + 1))
df["bh_critical"] = df["rank"] / m * ALPHA

# Step-up adjusted p-values: p_adj(i) = min_{k >= i} p(k) * m / k, capped at 1
adjusted = (df["p_raw"] * m / df["rank"]).tolist()
for i in range(m - 2, -1, -1):
    adjusted[i] = min(adjusted[i], adjusted[i + 1])
df["p_adj_bh"] = [min(p, 1.0) for p in adjusted]
df["significant_after_bh"] = df["p_adj_bh"] < ALPHA

print(df.to_string(index=False))
df.to_csv(OUT_PATH, index=False)
print(f"\n{int(df['significant_after_bh'].sum())}/{m} concepts survive BH correction")
print(f"Saved: {OUT_PATH}")
