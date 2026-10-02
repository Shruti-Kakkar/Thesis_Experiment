"""
vignette_bias_counterfactual_test.py
Layer 3 of the vignette bias-detection experiment: a direct causal test,
mirroring ruler_bias_counterfactual_test.py exactly (see that script's
docstring for the Winkler et al. 2019 precedent this design follows).

DESIGN:
  - Take N genuinely vignette-free NV images from the SAME held-out ISIC
    2019 test pool the ruler counterfactual test used (NV_test_clean --
    true class NV per the official test ground truth, confirmed ruler-free
    and rectangular/non-vignetted, never seen during training). Using the
    identical base pool makes the two counterfactual tests directly
    comparable.
  - For each: get the model's baseline P(MEL) on the clean image
  - Digitally add a synthetic circular vignette overlay (same image,
    add_vignette() from vignette_overlay.py)
  - Get P(MEL) again on the SAME image with the vignette added
  - shift = P(MEL | vignette added) - P(MEL | clean)

Vignette parameters are FIXED across all images (radius_frac=0.58,
feather_frac=0.08, center_jitter=0 -- perfectly centered, no per-image
randomization) per the deterministic-artifact convention established for
the ruler counterfactual test: randomizing appearance adds variance to
the paired-shift estimate without adding causal insight, since the goal
is isolating presence-vs-absence of the artifact, not generalizing
across vignette designs.

STATISTICAL TEST: paired t-test, same as the ruler counterfactual test.

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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models2", "resnet50v2_isic2019_final_padonly_seed2.keras"
)
CLEAN_DIR = os.path.join(PROJECT_ROOT, "datasets", "ruler_sorted_test", "NV_test_clean")
GT_CSV = os.path.join(PROJECT_ROOT, "datasets", "ISIC_2019_Test_GroundTruth.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs2", "vignette_bias_counterfactual_test_set")
EXAMPLES_DIR = os.path.join(RESULTS_DIR, "example_images")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EXAMPLES_DIR, exist_ok=True)

N_IMAGES = 50          # matches the ruler counterfactual test's N
N_EXAMPLES_TO_SAVE = 6
IMG_SIZE = 224

# Fixed, deterministic vignette parameters -- same for every image
VIGNETTE_RADIUS_FRAC = 0.58   # matches the corrected vignette_present CAV build
VIGNETTE_FEATHER_FRAC = 0.08
VIGNETTE_CENTER_JITTER = 0.0  # perfectly centered -- no per-image randomization
VIGNETTE_DARK_LEVEL = 4

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from VisualTCAV import KerasModelWrapper
from tensorflow.keras.applications.resnet_v2 import (
    preprocess_input as preprocess_resnet_v2
)
from vignette_overlay import add_vignette

print("=" * 60)
print("Vignette Bias -- Counterfactual Overlay Test")
print(f"N images: {N_IMAGES} | Class tested: NV (checking P(MEL) shift)")
print(f"Vignette: radius_frac={VIGNETTE_RADIUS_FRAC}, "
      f"feather_frac={VIGNETTE_FEATHER_FRAC} (fixed, deterministic)")
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
# 2. GROUND-TRUTH ID -> CLASS LOOKUP (same as ruler_bias_counterfactual_test.py)
# ─────────────────────────────────────────────
df = pd.read_csv(GT_CSV)
df['label'] = df[CLASS_NAMES].values.argmax(axis=1)
df['class'] = df['label'].apply(lambda i: CLASS_NAMES[i])

def normalize_id(filename):
    base = os.path.splitext(filename)[0]
    return base.replace('_downsampled', '')

id_to_class = dict(zip(df['image'].apply(normalize_id), df['class']))

# ─────────────────────────────────────────────
# 3. PICK N CLEAN NV IMAGES -- same pool/order logic as the ruler test
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
# 4. PREDICTION HELPER — aspect-preserving pad resize (matches training),
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
# 5. RUN THE COUNTERFACTUAL TEST
# ─────────────────────────────────────────────
results = []
for i, fname in enumerate(nv_clean_files):
    img_path = os.path.join(CLEAN_DIR, fname)
    pil_img = Image.open(img_path).convert('RGB')

    p_mel_clean = get_mel_probability(pil_img)

    vign_img = add_vignette(
        pil_img,
        radius_frac=VIGNETTE_RADIUS_FRAC,
        feather_frac=VIGNETTE_FEATHER_FRAC,
        center_jitter=VIGNETTE_CENTER_JITTER,
        dark_level=VIGNETTE_DARK_LEVEL,
        seed=0,  # fixed seed -- center_jitter=0 makes this a no-op anyway
    )
    p_mel_vign = get_mel_probability(vign_img)

    shift = p_mel_vign - p_mel_clean
    results.append({
        'filename': fname,
        'p_mel_clean': p_mel_clean,
        'p_mel_vignette': p_mel_vign,
        'shift': shift,
    })

    print(f"  [{i+1}/{len(nv_clean_files)}] {fname}: "
          f"P(MEL) clean={p_mel_clean:.4f} -> vignette={p_mel_vign:.4f} "
          f"(shift={shift:+.4f})")

    if i < N_EXAMPLES_TO_SAVE:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(pil_img)
        axes[0].set_title(f"Clean\nP(MEL)={p_mel_clean:.3f}")
        axes[0].axis('off')
        axes[1].imshow(vign_img)
        axes[1].set_title(f"+ Synthetic vignette\nP(MEL)={p_mel_vign:.3f}")
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
vign_probs = np.array([r['p_mel_vignette'] for r in results])

t_stat, p_value = stats.ttest_rel(vign_probs, clean_probs)  # PAIRED test -- same images

n_flipped_to_mel = np.sum(
    (clean_probs < 0.5) & (vign_probs >= 0.5)
)

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Mean P(MEL), clean:          {clean_probs.mean():.4f}")
print(f"Mean P(MEL), vignette added: {vign_probs.mean():.4f}")
print(f"Mean shift:                  {shifts.mean():+.4f}  (std={shifts.std():.4f})")
print(f"Paired t-test:               t={t_stat:.3f}, p={p_value:.6f}")
print(f"Significant at α=0.05?       {'YES' if p_value < 0.05 else 'no'}")
print(f"Images that crossed the MEL decision boundary "
      f"(P<0.5 -> P>=0.5) purely from the overlay: {n_flipped_to_mel}/{len(results)}")

# ─────────────────────────────────────────────
# 7. SAVE FULL RESULTS
# ─────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(RESULTS_DIR, "counterfactual_results.csv"), index=False)

fig, ax = plt.subplots(figsize=(6, 5))
for r in results:
    ax.plot([0, 1], [r['p_mel_clean'], r['p_mel_vignette']], 'o-', color='gray', alpha=0.4)
ax.axhline(0.5, color='red', linestyle='--', linewidth=1, label='Decision boundary')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Clean', '+ Synthetic vignette'])
ax.set_ylabel('P(MEL)')
ax.set_title(f'Counterfactual vignette overlay -- P(MEL) shift on {len(results)} true-NV images\n'
             f'mean shift={shifts.mean():+.4f}, paired t-test p={p_value:.4f}')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "counterfactual_shift_plot.png"), dpi=150)
plt.close(fig)

print(f"\nResults saved to: {RESULTS_DIR}")
print(f"Example before/after image pairs: {EXAMPLES_DIR}")
