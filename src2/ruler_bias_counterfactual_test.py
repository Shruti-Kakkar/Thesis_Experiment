"""
ruler_bias_counterfactual_test.py
Layer 3 of the bias-detection experiment: a direct causal test, not a
correlational one -- mirrors Winkler et al. (2019, JAMA Dermatology),
who compared the SAME lesions imaged with and without a surgical
marking/ruler and found specificity collapsed (84.1% -> 45.8%) once
marked, even though sensitivity looked "better."

DESIGN:
  - Take N genuinely ruler-free NV images from the held-out ISIC 2019 test
    set (NV_test_clean -- true class NV per the official test ground truth,
    confirmed no ruler, and never seen during training)
  - For each: get the model's baseline P(MEL) on the clean image
  - Digitally add a synthetic ruler overlay (same image, edge placement
    randomized per image) -- following Winkler et al.'s own precedent of
    using an "electronically added mark" as one of their three conditions
  - Get P(MEL) again on the SAME image with the ruler added
  - shift = P(MEL | ruler added) - P(MEL | clean)

WHY THIS IS STRONGER EVIDENCE than the Global/Local attribution work:
this never depends on knowing what's in any test set, and it's a genuine
paired causal comparison (same image, only the artifact changes) --
immune to the "do we actually know which images have rulers" gap that
motivated this whole redesign.

STATISTICAL TEST: paired (not independent) t-test, since each image is
compared to itself before/after -- correct test for this design, unlike
the independent-samples Welch test used elsewhere in this project.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import os
import random
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
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs2", "ruler_bias_counterfactual_test_set")
EXAMPLES_DIR = os.path.join(RESULTS_DIR, "example_images")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EXAMPLES_DIR, exist_ok=True)

N_IMAGES = 50          # how many clean NV images to test -- increase if you want more power
N_EXAMPLES_TO_SAVE = 6  # how many before/after image pairs to save for the thesis
IMG_SIZE = 224
RANDOM_SEED = 42

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
print("Ruler Bias -- Counterfactual Overlay Test")
print(f"N images: {N_IMAGES} | Class tested: NV (checking P(MEL) shift)")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. LESION LOCALIZATION (rough, for ruler placement only)
# ISIC lesion images are framed with the lesion near center. We find the
# darker/browner blob nearest the image center, remove thin hair-like
# structures with a binary opening, and return its bounding box.
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
# 2. SYNTHETIC RULER OVERLAY
# A translucent ruler (tick marks only, no filled bar) placed flush against
# one side of the lesion's bounding box, as if laid on the skin to measure
# it -- rather than a solid strip pasted at the image edge.
# Visually verified before building this script -- see conversation.
# ─────────────────────────────────────────────
def add_synthetic_ruler(pil_img, edge='right'):
    base = pil_img.copy().convert('RGBA')
    w, h = base.size
    x0, y0, x1, y1 = find_lesion_bbox(pil_img)

    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    gap = max(2, min(w, h) // 60)
    tick_len_major = max(6, min(w, h) // 18)
    tick_len_minor = tick_len_major // 2
    tick_color = (25, 25, 25, 190)
    edge_color = (60, 60, 60, 130)

    if edge in ('left', 'right'):
        y_start, y_end = max(0, y0), min(h, y1)
        if y_end <= y_start:
            y_start, y_end = 0, h
        rx = min(w - 1, x1 + gap) if edge == 'right' else max(0, x0 - gap)
        sign = -1 if edge == 'right' else 1  # ticks point back toward the lesion
        draw.line([rx, y_start, rx, y_end], fill=edge_color, width=2)
        tick_spacing = max(6, (y_end - y_start) // 15)
        for i, y in enumerate(range(y_start, y_end, tick_spacing)):
            tick_len = tick_len_major if i % 5 == 0 else tick_len_minor
            draw.line([rx, y, rx + sign * tick_len, y], fill=tick_color, width=2)
    else:
        x_start, x_end = max(0, x0), min(w, x1)
        if x_end <= x_start:
            x_start, x_end = 0, w
        ry = min(h - 1, y1 + gap) if edge == 'bottom' else max(0, y0 - gap)
        sign = -1 if edge == 'bottom' else 1  # ticks point back toward the lesion
        draw.line([x_start, ry, x_end, ry], fill=edge_color, width=2)
        tick_spacing = max(6, (x_end - x_start) // 15)
        for i, x in enumerate(range(x_start, x_end, tick_spacing)):
            tick_len = tick_len_major if i % 5 == 0 else tick_len_minor
            draw.line([x, ry, x, ry + sign * tick_len], fill=tick_color, width=2)

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
# 4. GROUND-TRUTH ID -> CLASS LOOKUP (same as elsewhere in this project)
# ─────────────────────────────────────────────
df = pd.read_csv(GT_CSV)
df['label'] = df[CLASS_NAMES].values.argmax(axis=1)
df['class'] = df['label'].apply(lambda i: CLASS_NAMES[i])

def normalize_id(filename):
    base = os.path.splitext(filename)[0]
    return base.replace('_downsampled', '')

id_to_class = dict(zip(df['image'].apply(normalize_id), df['class']))

# ─────────────────────────────────────────────
# 5. PICK N CLEAN NV IMAGES
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
# 6. PREDICTION HELPER — aspect-preserving pad resize (matches training),
# preprocess, run through model, return P(MEL)
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
# 7. RUN THE COUNTERFACTUAL TEST
# ─────────────────────────────────────────────
rng = random.Random(RANDOM_SEED)
edges = ['left', 'right', 'top', 'bottom']

results = []
for i, fname in enumerate(nv_clean_files):
    img_path = os.path.join(CLEAN_DIR, fname)
    pil_img = Image.open(img_path).convert('RGB')

    p_mel_clean = get_mel_probability(pil_img)

    edge = edges[i % len(edges)]  # cycle through edges for placement diversity
    ruler_img = add_synthetic_ruler(pil_img, edge=edge)
    p_mel_ruler = get_mel_probability(ruler_img)

    shift = p_mel_ruler - p_mel_clean
    results.append({
        'filename': fname,
        'edge': edge,
        'p_mel_clean': p_mel_clean,
        'p_mel_ruler': p_mel_ruler,
        'shift': shift,
    })

    print(f"  [{i+1}/{len(nv_clean_files)}] {fname}: "
          f"P(MEL) clean={p_mel_clean:.4f} -> ruler={p_mel_ruler:.4f} "
          f"(shift={shift:+.4f})")

    if i < N_EXAMPLES_TO_SAVE:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(pil_img)
        axes[0].set_title(f"Clean\nP(MEL)={p_mel_clean:.3f}")
        axes[0].axis('off')
        axes[1].imshow(ruler_img)
        axes[1].set_title(f"+ Synthetic ruler ({edge})\nP(MEL)={p_mel_ruler:.3f}")
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
ruler_probs = np.array([r['p_mel_ruler'] for r in results])

t_stat, p_value = stats.ttest_rel(ruler_probs, clean_probs)  # PAIRED test -- same images

n_flipped_to_mel = np.sum(
    (clean_probs < 0.5) & (ruler_probs >= 0.5)
)  # count of images that crossed the MEL decision boundary purely from the overlay

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Mean P(MEL), clean:        {clean_probs.mean():.4f}")
print(f"Mean P(MEL), ruler added:  {ruler_probs.mean():.4f}")
print(f"Mean shift:                {shifts.mean():+.4f}  (std={shifts.std():.4f})")
print(f"Paired t-test:             t={t_stat:.3f}, p={p_value:.6f}")
print(f"Significant at α=0.05?     {'YES' if p_value < 0.05 else 'no'}")
print(f"Images that crossed the MEL decision boundary "
      f"(P<0.5 -> P>=0.5) purely from the overlay: {n_flipped_to_mel}/{len(results)}")

# ─────────────────────────────────────────────
# 9. SAVE FULL RESULTS
# ─────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(RESULTS_DIR, "counterfactual_results.csv"), index=False)

fig, ax = plt.subplots(figsize=(6, 5))
for r in results:
    ax.plot([0, 1], [r['p_mel_clean'], r['p_mel_ruler']], 'o-', color='gray', alpha=0.4)
ax.axhline(0.5, color='red', linestyle='--', linewidth=1, label='Decision boundary')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Clean', '+ Synthetic ruler'])
ax.set_ylabel('P(MEL)')
ax.set_title(f'Counterfactual ruler overlay -- P(MEL) shift on {len(results)} true-NV images\n'
             f'mean shift={shifts.mean():+.4f}, paired t-test p={p_value:.4f}')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "counterfactual_shift_plot.png"), dpi=150)
plt.close(fig)

print(f"\nResults saved to: {RESULTS_DIR}")
print(f"Example before/after image pairs: {EXAMPLES_DIR}")