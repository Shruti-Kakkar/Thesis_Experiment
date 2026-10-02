"""
vignette_bias_counterfactual_negcontrol_test.py
Negative control for vignette_bias_counterfactual_test.py: same design,
same 50 true-NV images, same paired t-test -- but the injected artifact
is add_generic_border() (a plain RECTANGULAR dark border, area-matched
to the real vignette) instead of add_vignette() (a circular vignette).

WHY: "add vignette -> P(MEL) rises" could in principle be explained by
"add ANY dark border -> P(MEL) rises," not specifically a circular
vignette. This isolates vignette-SHAPE-specific effect from generic
edge-darkening by showing whether a shape-matched-in-darkened-area but
differently-shaped artifact produces a smaller (or absent) effect.

Author: Shruti Kakkar
"""

import sys
sys.dont_write_bytecode = True

import os
import numpy as np
import pandas as pd
from scipy import stats
from PIL import Image
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
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs2", "vignette_bias_counterfactual_negcontrol_test_set")
EXAMPLES_DIR = os.path.join(RESULTS_DIR, "example_images")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EXAMPLES_DIR, exist_ok=True)

N_IMAGES = 50
N_EXAMPLES_TO_SAVE = 6
IMG_SIZE = 224

# Same radius_frac/feather_frac/dark_level as the real vignette test, so
# the border is area-matched -- only the shape (rectangular, not
# circular) differs.
BORDER_RADIUS_FRAC = 0.58
BORDER_FEATHER_FRAC = 0.08
BORDER_DARK_LEVEL = 4

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from VisualTCAV import KerasModelWrapper
from tensorflow.keras.applications.resnet_v2 import (
    preprocess_input as preprocess_resnet_v2
)
from vignette_overlay import add_generic_border

print("=" * 60)
print("Vignette Bias -- NEGATIVE CONTROL Counterfactual Test")
print(f"N images: {N_IMAGES} | Class tested: NV (checking P(MEL) shift)")
print(f"Generic border: area-matched to radius_frac={BORDER_RADIUS_FRAC} vignette, "
      f"feather_frac={BORDER_FEATHER_FRAC} (fixed, deterministic, RECTANGULAR not circular)")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. LOAD MODEL
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
# 2. GROUND-TRUTH ID -> CLASS LOOKUP
# ─────────────────────────────────────────────
df = pd.read_csv(GT_CSV)
df['label'] = df[CLASS_NAMES].values.argmax(axis=1)
df['class'] = df['label'].apply(lambda i: CLASS_NAMES[i])

def normalize_id(filename):
    base = os.path.splitext(filename)[0]
    return base.replace('_downsampled', '')

id_to_class = dict(zip(df['image'].apply(normalize_id), df['class']))

# ─────────────────────────────────────────────
# 3. PICK N CLEAN NV IMAGES -- IDENTICAL selection to the real vignette test
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
# 4. PREDICTION HELPER
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
# 5. RUN THE COUNTERFACTUAL TEST
# ─────────────────────────────────────────────
results = []
for i, fname in enumerate(nv_clean_files):
    img_path = os.path.join(CLEAN_DIR, fname)
    pil_img = Image.open(img_path).convert('RGB')

    p_mel_clean = get_mel_probability(pil_img)

    border_img = add_generic_border(
        pil_img,
        radius_frac=BORDER_RADIUS_FRAC,
        feather_frac=BORDER_FEATHER_FRAC,
        dark_level=BORDER_DARK_LEVEL,
    )
    p_mel_border = get_mel_probability(border_img)

    shift = p_mel_border - p_mel_clean
    results.append({
        'filename': fname,
        'p_mel_clean': p_mel_clean,
        'p_mel_border': p_mel_border,
        'shift': shift,
    })

    print(f"  [{i+1}/{len(nv_clean_files)}] {fname}: "
          f"P(MEL) clean={p_mel_clean:.4f} -> border={p_mel_border:.4f} "
          f"(shift={shift:+.4f})")

    if i < N_EXAMPLES_TO_SAVE:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(pil_img)
        axes[0].set_title(f"Clean\nP(MEL)={p_mel_clean:.3f}")
        axes[0].axis('off')
        axes[1].imshow(border_img)
        axes[1].set_title(f"+ Generic rectangular border\nP(MEL)={p_mel_border:.3f}")
        axes[1].axis('off')
        fig.suptitle(f"{normalize_id(fname)} (true class: NV)")
        plt.tight_layout()
        plt.savefig(os.path.join(EXAMPLES_DIR, f"pair_{normalize_id(fname)}.png"), dpi=150)
        plt.close(fig)

# ─────────────────────────────────────────────
# 6. AGGREGATE STATISTICS
# ─────────────────────────────────────────────
shifts = np.array([r['shift'] for r in results])
clean_probs = np.array([r['p_mel_clean'] for r in results])
border_probs = np.array([r['p_mel_border'] for r in results])

t_stat, p_value = stats.ttest_rel(border_probs, clean_probs)

n_flipped_to_mel = np.sum(
    (clean_probs < 0.5) & (border_probs >= 0.5)
)

print(f"\n{'='*60}")
print("RESULTS (NEGATIVE CONTROL -- generic rectangular border)")
print(f"{'='*60}")
print(f"Mean P(MEL), clean:        {clean_probs.mean():.4f}")
print(f"Mean P(MEL), border added: {border_probs.mean():.4f}")
print(f"Mean shift:                {shifts.mean():+.4f}  (std={shifts.std():.4f})")
print(f"Paired t-test:             t={t_stat:.3f}, p={p_value:.6f}")
print(f"Significant at α=0.05?     {'YES' if p_value < 0.05 else 'no'}")
print(f"Images that crossed the MEL decision boundary "
      f"(P<0.5 -> P>=0.5) purely from the overlay: {n_flipped_to_mel}/{len(results)}")
print(f"\nCompare to the real vignette counterfactual test's mean shift "
      f"(+0.1042, p<0.000001) -- a smaller/non-significant shift here "
      f"would confirm the real effect is vignette-SHAPE-specific, not "
      f"generic edge-darkening.")

# ─────────────────────────────────────────────
# 7. SAVE FULL RESULTS
# ─────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(RESULTS_DIR, "counterfactual_results.csv"), index=False)

fig, ax = plt.subplots(figsize=(6, 5))
for r in results:
    ax.plot([0, 1], [r['p_mel_clean'], r['p_mel_border']], 'o-', color='gray', alpha=0.4)
ax.axhline(0.5, color='red', linestyle='--', linewidth=1, label='Decision boundary')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Clean', '+ Generic rectangular border'])
ax.set_ylabel('P(MEL)')
ax.set_title(f'NEGATIVE CONTROL -- P(MEL) shift on {len(results)} true-NV images\n'
             f'mean shift={shifts.mean():+.4f}, paired t-test p={p_value:.4f}')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "counterfactual_shift_plot.png"), dpi=150)
plt.close(fig)

print(f"\nResults saved to: {RESULTS_DIR}")
print(f"Example before/after image pairs: {EXAMPLES_DIR}")
