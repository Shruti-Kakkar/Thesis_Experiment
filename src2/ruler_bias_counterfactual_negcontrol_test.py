"""
ruler_bias_counterfactual_negcontrol_test.py
Negative control for ruler_bias_counterfactual_test.py: same design,
same 50 true-NV images, same paired t-test, same edge-cycling logic --
but the injected artifact is a plain SOLID GRAY BAR (add_generic_edge_mark,
no tick marks) instead of add_synthetic_ruler()'s measurement ticks.

WHY: "add ruler ticks -> P(MEL) rises" could in principle be explained by
"add ANY dark mark near the lesion edge -> P(MEL) rises," not specifically
the ruler's measurement-tick pattern. This isolates the ruler-PATTERN-
specific effect from a generic edge artifact -- this negative control was
proposed early in the project but never implemented until now.

The gray bar occupies the same gap/span/edge as add_synthetic_ruler()'s
tick zone (same lesion-relative placement, same edge-cycling order), so
placement and footprint are matched -- only the presence of an actual
tick PATTERN differs.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import os
import numpy as np
import pandas as pd
from scipy import stats
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser(
    "~/scratch/dev-uos/projects/VTCAV_Dermatology"
)
SOURCE_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models2", "resnet50v2_isic2019_final_padonly_seed2.keras"
)
CLEAN_DIR = os.path.join(PROJECT_ROOT, "datasets", "ruler_sorted_test", "NV_test_clean")
GT_CSV = os.path.join(PROJECT_ROOT, "datasets", "ISIC_2019_Test_GroundTruth.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs2", "ruler_bias_counterfactual_negcontrol_test_set")
EXAMPLES_DIR = os.path.join(RESULTS_DIR, "example_images")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EXAMPLES_DIR, exist_ok=True)

N_IMAGES = 50
N_EXAMPLES_TO_SAVE = 6
IMG_SIZE = 224
RANDOM_SEED = 42  # same seed as the real ruler counterfactual test, so the
                   # edge-cycling order is identical -- placement is matched

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from VisualTCAV import KerasModelWrapper
from tensorflow.keras.applications.resnet_v2 import (
    preprocess_input as preprocess_resnet_v2
)

print("=" * 60)
print("Ruler Bias -- NEGATIVE CONTROL Counterfactual Test (plain bar, no ticks)")
print(f"N images: {N_IMAGES} | Class tested: NV (checking P(MEL) shift)")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. LESION LOCALIZATION -- identical to ruler_bias_counterfactual_test.py
# ─────────────────────────────────────────────
from scipy import ndimage

def find_lesion_bbox(pil_img):
    img = np.array(pil_img.convert('RGB')).astype(np.float32)
    h, w, _ = img.shape
    gray = img.mean(axis=2)

    mask = gray < np.percentile(gray, 35)
    mask = ndimage.binary_opening(mask, structure=np.ones((5, 5)))
    labeled, n_labels = ndimage.label(mask)

    if n_labels == 0:
        return (w // 4, h // 4, 3 * w // 4, 3 * h // 4)

    center_label = labeled[h // 2, w // 2]
    if center_label != 0:
        best_label = center_label
    else:
        sizes = ndimage.sum(mask, labeled, range(1, n_labels + 1))
        best_label = int(np.argmax(sizes)) + 1

    y_slice, x_slice = ndimage.find_objects(labeled == best_label)[0]
    return (x_slice.start, y_slice.start, x_slice.stop, y_slice.stop)

# ─────────────────────────────────────────────
# 2. GENERIC EDGE MARK -- a plain solid gray bar, NO tick marks.
# Same gap/span/edge placement as add_synthetic_ruler()'s tick zone, and
# the bar's width matches tick_len_major's scale, so footprint is
# comparable -- only the tick PATTERN is removed, replaced by a uniform
# fill using the average of the real ruler's tick_color/edge_color tones.
# ─────────────────────────────────────────────
def add_generic_edge_mark(pil_img, edge='right'):
    base = pil_img.copy().convert('RGBA')
    w, h = base.size
    x0, y0, x1, y1 = find_lesion_bbox(pil_img)

    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    gap = max(2, min(w, h) // 60)
    bar_width = max(6, min(w, h) // 18)  # matches tick_len_major's scale
    fill_color = (42, 42, 42, 160)  # average tone of tick_color/edge_color

    if edge in ('left', 'right'):
        y_start, y_end = max(0, y0), min(h, y1)
        if y_end <= y_start:
            y_start, y_end = 0, h
        if edge == 'right':
            rx0 = min(w - 1, x1 + gap)
            rx1 = min(w, rx0 + bar_width)
        else:
            rx1 = max(0, x0 - gap)
            rx0 = max(0, rx1 - bar_width)
        draw.rectangle([rx0, y_start, rx1, y_end], fill=fill_color)
    else:
        x_start, x_end = max(0, x0), min(w, x1)
        if x_end <= x_start:
            x_start, x_end = 0, w
        if edge == 'bottom':
            ry0 = min(h - 1, y1 + gap)
            ry1 = min(h, ry0 + bar_width)
        else:
            ry1 = max(0, y0 - gap)
            ry0 = max(0, ry1 - bar_width)
        draw.rectangle([x_start, ry0, x_end, ry1], fill=fill_color)

    img = Image.alpha_composite(base, overlay).convert('RGB')
    return img

# ─────────────────────────────────────────────
# 3. LOAD MODEL
# ─────────────────────────────────────────────
MODEL_DIR = os.path.join(RESULTS_DIR, "model")
MODEL_SUBDIR = os.path.join(MODEL_DIR, "resnet50v2")
os.makedirs(MODEL_SUBDIR, exist_ok=True)
graph_dest = os.path.join(MODEL_SUBDIR, "resnet50v2_isic2019_final.keras")
if not os.path.exists(graph_dest):
    os.symlink(SOURCE_MODEL_PATH, graph_dest)
labels_dest = os.path.join(MODEL_SUBDIR, "isic2019_classes.txt")
if not os.path.exists(labels_dest):
    with open(labels_dest, 'w') as f:
        for cls in CLASS_NAMES:
            f.write(cls + "\n")

wrapper = KerasModelWrapper(graph_dest, labels_dest, batch_size=20, model_name="resnet50v2")
mel_index = wrapper.label_to_id("MEL")
print(f"Model loaded. MEL class index: {mel_index}\n")

# ─────────────────────────────────────────────
# 4. GROUND-TRUTH ID -> CLASS LOOKUP
# ─────────────────────────────────────────────
df = pd.read_csv(GT_CSV)
df['label'] = df[CLASS_NAMES].values.argmax(axis=1)
df['class'] = df['label'].apply(lambda i: CLASS_NAMES[i])

def normalize_id(filename):
    base = os.path.splitext(filename)[0]
    return base.replace('_downsampled', '')

id_to_class = dict(zip(df['image'].apply(normalize_id), df['class']))

# ─────────────────────────────────────────────
# 5. PICK N CLEAN NV IMAGES -- IDENTICAL selection to the real ruler test
# ─────────────────────────────────────────────
nv_clean_files = []
for fname in sorted(os.listdir(CLEAN_DIR)):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    if id_to_class.get(normalize_id(fname)) == "NV":
        nv_clean_files.append(fname)
    if len(nv_clean_files) >= N_IMAGES:
        break

print(f"Found {len(nv_clean_files)} clean, confirmed-NV images to test\n")

# ─────────────────────────────────────────────
# 6. PREDICTION HELPER
# ─────────────────────────────────────────────
def resize_with_pad_pil(img, shape):
    target_w, target_h = shape
    orig_w, orig_h = img.size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = max(1, round(orig_w * scale)), max(1, round(orig_h * scale))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new('RGB', (target_w, target_h), (0, 0, 0))
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas

def get_mel_probability(pil_img):
    resized = resize_with_pad_pil(pil_img, (IMG_SIZE, IMG_SIZE))
    arr = np.array(resized).astype(np.float32)
    arr = preprocess_resnet_v2(arr)
    batch = np.expand_dims(arr, axis=0)
    predictions = wrapper.get_predictions(tf.constant(batch, dtype=tf.float32))
    return float(predictions[0][mel_index])

# ─────────────────────────────────────────────
# 7. RUN THE COUNTERFACTUAL TEST -- same edge-cycling as the real ruler test
# ─────────────────────────────────────────────
edges = ['left', 'right', 'top', 'bottom']

results = []
for i, fname in enumerate(nv_clean_files):
    img_path = os.path.join(CLEAN_DIR, fname)
    pil_img = Image.open(img_path).convert('RGB')

    p_mel_clean = get_mel_probability(pil_img)

    edge = edges[i % len(edges)]
    mark_img = add_generic_edge_mark(pil_img, edge=edge)
    p_mel_mark = get_mel_probability(mark_img)

    shift = p_mel_mark - p_mel_clean
    results.append({
        'filename': fname,
        'edge': edge,
        'p_mel_clean': p_mel_clean,
        'p_mel_mark': p_mel_mark,
        'shift': shift,
    })

    print(f"  [{i+1}/{len(nv_clean_files)}] {fname}: "
          f"P(MEL) clean={p_mel_clean:.4f} -> mark={p_mel_mark:.4f} "
          f"(shift={shift:+.4f})")

    if i < N_EXAMPLES_TO_SAVE:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(pil_img)
        axes[0].set_title(f"Clean\nP(MEL)={p_mel_clean:.3f}")
        axes[0].axis('off')
        axes[1].imshow(mark_img)
        axes[1].set_title(f"+ Generic edge mark ({edge})\nP(MEL)={p_mel_mark:.3f}")
        axes[1].axis('off')
        fig.suptitle(f"{normalize_id(fname)} (true class: NV)")
        plt.tight_layout()
        plt.savefig(os.path.join(EXAMPLES_DIR, f"pair_{normalize_id(fname)}.png"), dpi=150)
        plt.close(fig)

# ─────────────────────────────────────────────
# 8. AGGREGATE STATISTICS
# ─────────────────────────────────────────────
shifts = np.array([r['shift'] for r in results])
clean_probs = np.array([r['p_mel_clean'] for r in results])
mark_probs = np.array([r['p_mel_mark'] for r in results])

t_stat, p_value = stats.ttest_rel(mark_probs, clean_probs)

n_flipped_to_mel = np.sum(
    (clean_probs < 0.5) & (mark_probs >= 0.5)
)

print(f"\n{'='*60}")
print("RESULTS (NEGATIVE CONTROL -- plain bar, no tick marks)")
print(f"{'='*60}")
print(f"Mean P(MEL), clean:      {clean_probs.mean():.4f}")
print(f"Mean P(MEL), mark added: {mark_probs.mean():.4f}")
print(f"Mean shift:              {shifts.mean():+.4f}  (std={shifts.std():.4f})")
print(f"Paired t-test:           t={t_stat:.3f}, p={p_value:.6f}")
print(f"Significant at α=0.05?   {'YES' if p_value < 0.05 else 'no'}")
print(f"Images that crossed the MEL decision boundary "
      f"(P<0.5 -> P>=0.5) purely from the overlay: {n_flipped_to_mel}/{len(results)}")
print(f"\nCompare to the real ruler counterfactual test's mean shift "
      f"(+0.037, p=0.0102) -- a smaller/non-significant shift here would "
      f"confirm the real effect is tick-PATTERN-specific, not a generic "
      f"edge artifact.")

# ─────────────────────────────────────────────
# 9. SAVE FULL RESULTS
# ─────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(RESULTS_DIR, "counterfactual_results.csv"), index=False)

fig, ax = plt.subplots(figsize=(6, 5))
for r in results:
    ax.plot([0, 1], [r['p_mel_clean'], r['p_mel_mark']], 'o-', color='gray', alpha=0.4)
ax.axhline(0.5, color='red', linestyle='--', linewidth=1, label='Decision boundary')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Clean', '+ Generic edge mark'])
ax.set_ylabel('P(MEL)')
ax.set_title(f'NEGATIVE CONTROL -- P(MEL) shift on {len(results)} true-NV images\n'
             f'mean shift={shifts.mean():+.4f}, paired t-test p={p_value:.4f}')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "counterfactual_shift_plot.png"), dpi=150)
plt.close(fig)

print(f"\nResults saved to: {RESULTS_DIR}")
print(f"Example before/after image pairs: {EXAMPLES_DIR}")
