"""
plot_confusion_matrix.py

PURPOSE: the row-normalised confusion-matrix figure in the thesis
(fig:confusion_matrix, Section 5.2, confusion_matrix_seed2.png) for the
padding-preserving seed-2 model.

train_new_padonly_multiseed.py computes this matrix but only saves it as
a raw-count image (outputs2/confusion_matrix_PadOnly_seed2.png), not as
data. The counts below were transcribed from that image. Before
plotting, they are checked against the saved classification report
(outputs2/classification_report_PadOnly_seed2.txt): every class's
support, precision, recall and F1, plus overall accuracy, must match the
report to its printed two decimals, otherwise the script stops.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_PATH = os.path.join(PROJECT_ROOT, "outputs2", "classification_report_PadOnly_seed2.txt")
OUT_PATH = os.path.join(PROJECT_ROOT, "outputs2", "confusion_matrix_seed2.png")

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# Rows = true class, columns = predicted class, same order as CLASS_NAMES
cm = np.array([
    [862, 185,  75,  84,  88,  3, 12, 18],
    [305, 1851, 141, 44, 113,  9, 15, 17],
    [ 64,  29, 650, 147,  36,  9, 14, 26],
    [ 28,   8, 130, 150,  46,  1,  4,  7],
    [131,  63,  80,  83, 263,  2,  8, 30],
    [  4,  14,  22,   6,   6, 35,  0,  4],
    [ 17,  15,  23,   6,   4,  2, 34,  3],
    [  7,   3,  47,  31,  14,  1,  0, 62],
])

# ── Check the transcribed counts against the saved classification report ──
report = open(REPORT_PATH).read()
tp = np.diag(cm)
precision = tp / cm.sum(axis=0)
recall = tp / cm.sum(axis=1)
f1 = 2 * precision * recall / (precision + recall)
for i, name in enumerate(CLASS_NAMES):
    p, r, f, s = re.search(rf"^\s*{name}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", report, re.M).groups()
    assert int(s) == cm[i].sum(), f"{name}: support {cm[i].sum()} != report {s}"
    for label, ours, theirs in [("precision", precision[i], p), ("recall", recall[i], r), ("F1", f1[i], f)]:
        assert f"{ours:.2f}" == theirs, f"{name}: {label} {ours:.2f} != report {theirs}"
accuracy = re.search(r"accuracy\s+([\d.]+)", report).group(1)
assert f"{tp.sum() / cm.sum():.2f}" == accuracy, "overall accuracy mismatch"
print(f"Counts match the classification report (accuracy {accuracy}, n={cm.sum()}).")

# ── Plot: shading and percentages row-normalised, raw count above each ──
pct = cm / cm.sum(axis=1, keepdims=True) * 100
annot = np.empty_like(cm, dtype=object)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot[i, j] = f"{cm[i, j]}\n({pct[i, j]:.0f}%)"

plt.figure(figsize=(10, 8))
sns.heatmap(pct, annot=annot, fmt='', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            vmin=0, vmax=100, cbar_kws={'label': 'Row-normalized (%)'})
plt.title('Confusion Matrix — ResNet50V2, padding-only (seed 2)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"Saved: {OUT_PATH}")
