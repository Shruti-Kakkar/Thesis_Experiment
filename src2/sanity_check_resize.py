"""
sanity_check_resize_real_images.py

Quick visual + quantitative check comparing:
  - naive resize (old train.py):        tf.image.resize(img, [224, 224])
  - aspect-preserving (train_new.py):    tf.image.resize_with_pad(img, 224, 224)

on a handful of REAL ISIC 2019 training images, before committing to a full
retrain. Run this once from src2/ on voxel.

Usage:
    conda activate vtcav_derma
    cd .../VTCAV_Dermatology/src2
    python sanity_check_resize_real_images.py
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_SIZE = 224
N_SAMPLES = 8  # how many real images to spot-check

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "datasets", "ISIC_2019_Training_Input",
                              "ISIC_2019_Training_Input")
TRAIN_CSV = os.path.join(BASE_DIR, "datasets", "ISIC_2019_Training_GroundTruth.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(TRAIN_CSV)
df['filepath'] = df['image'].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x + '.jpg'))
df = df[df['filepath'].apply(os.path.exists)]

# Sample a mix of images -- prioritize ones whose original aspect ratio is
# furthest from square, since that's where naive resize distorts the most.
def get_size(fp):
    img = tf.io.read_file(fp)
    img = tf.image.decode_jpeg(img, channels=3)
    return img.shape[0], img.shape[1]

sample_df = df.sample(n=min(200, len(df)), random_state=42)  # scan a subset for speed
sizes = sample_df['filepath'].apply(get_size)
sample_df = sample_df.assign(
    h=[s[0] for s in sizes], w=[s[1] for s in sizes])
sample_df['aspect_dev'] = (sample_df['w'] / sample_df['h'] - 1).abs()
picks = sample_df.sort_values('aspect_dev', ascending=False).head(N_SAMPLES)

fig, axes = plt.subplots(N_SAMPLES, 3, figsize=(9, 3 * N_SAMPLES))

for i, (_, row) in enumerate(picks.iterrows()):
    raw = tf.io.read_file(row['filepath'])
    img = tf.image.decode_jpeg(raw, channels=3)
    orig_h, orig_w = img.shape[0], img.shape[1]

    naive = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    padded = tf.image.resize_with_pad(img, IMG_SIZE, IMG_SIZE)

    axes[i, 0].imshow(img.numpy())
    axes[i, 0].set_title(f"{row['image']}\n{orig_w}x{orig_h}", fontsize=8)
    axes[i, 1].imshow(naive.numpy().astype('uint8'))
    axes[i, 1].set_title("naive resize", fontsize=8, color='darkred')
    axes[i, 2].imshow(padded.numpy().astype('uint8'))
    axes[i, 2].set_title("resize_with_pad", fontsize=8, color='darkgreen')

    for ax in axes[i]:
        ax.axis('off')

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'resize_sanity_check_real_images.png')
plt.savefig(out_path, dpi=150)
print(f"Saved comparison grid to: {out_path}")
print("\nOriginal aspect ratios of sampled (most-distorted) images:")
print(picks[['image', 'w', 'h', 'aspect_dev']].to_string(index=False))