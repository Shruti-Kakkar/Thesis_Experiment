"""
ruler_prevalence_by_class.py
Layer 1 of the ruler-bias experiment: before touching the model at all,
establish two facts about the DATA itself.

  1. CLASS-BALANCE SANITY CHECK
     Is the sequentially-sorted subset (first ~10k images, up through
     ISIC_0013778_downsampled) representative of the full 25,331-image
     dataset's class composition? If not, ruler-prevalence findings from
     this subset need much more cautious interpretation, since ISIC 2019
     aggregates multiple source datasets and sequential IDs may cluster
     by source.

  2. RULER PREVALENCE BY CLASS (ground truth, not model predictions)
     Of images truly labelled MEL, what fraction have a ruler
     (Normal_Ruler + Thick_Ruler) vs Clean? Same question for NV and
     the other 6 classes. Doubtful images are excluded from this ratio
     entirely (ambiguous by design -- shouldn't count as ruler or clean).

Author: Shruti Kakkar
"""

import os
import re
import pandas as pd

# ─────────────────────────────────────────────
# 0. CONFIG — fill in your actual ruler-sorted folder paths
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser(
    "~/scratch/dev-uos/projects/VTCAV_Dermatology"
)
TRAIN_CSV = os.path.join(PROJECT_ROOT, "datasets", "ISIC_2019_Training_GroundTruth.csv")

# TODO: set this to wherever your four sorted folders actually live
RULER_BASE_DIR = os.path.join(PROJECT_ROOT, "datasets", "ruler_sorted")  # <-- EDIT IF NEEDED

RULER_FOLDERS = {
    "Normal_Ruler": os.path.join(RULER_BASE_DIR, "Normal_Ruler"),
    "Thick_Ruler":  os.path.join(RULER_BASE_DIR, "Thick_Ruler"),
    "Clean":        os.path.join(RULER_BASE_DIR, "Clean"),
    "Doubtful":     os.path.join(RULER_BASE_DIR, "Doubtful"),
}

CUTOFF_IMAGE = "ISIC_0013778_downsampled"  # inclusive end of the sorted subset
CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

# ─────────────────────────────────────────────
# 1. LOAD GROUND TRUTH, SORT BY NUMERIC ID
# ─────────────────────────────────────────────
print("Loading ground truth CSV...")
df = pd.read_csv(TRAIN_CSV)
df['label'] = df[CLASS_NAMES].values.argmax(axis=1)
df['class'] = df['label'].apply(lambda i: CLASS_NAMES[i])

def extract_numeric_id(image_id):
    """ISIC_0013778 or ISIC_0013778_downsampled -> 13778"""
    match = re.search(r'ISIC_(\d+)', image_id)
    return int(match.group(1)) if match else None

df['numeric_id'] = df['image'].apply(extract_numeric_id)
df = df.sort_values('numeric_id').reset_index(drop=True)

# ─────────────────────────────────────────────
# 2. IDENTIFY THE SORTED SUBSET (up to and including CUTOFF_IMAGE)
# ─────────────────────────────────────────────
cutoff_numeric = extract_numeric_id(CUTOFF_IMAGE)
if cutoff_numeric is None:
    raise ValueError(f"Could not parse a numeric ID out of '{CUTOFF_IMAGE}'")

subset_df = df[df['numeric_id'] <= cutoff_numeric].copy()
full_df   = df.copy()

print(f"Full dataset: {len(full_df)} images")
print(f"Sorted subset (up to {CUTOFF_IMAGE}): {len(subset_df)} images")

# ─────────────────────────────────────────────
# 3. CLASS-BALANCE SANITY CHECK
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("CLASS-BALANCE CHECK: sorted subset vs. full dataset")
print("=" * 70)

full_counts = full_df['class'].value_counts()
subset_counts = subset_df['class'].value_counts()

full_pct = (full_counts / len(full_df) * 100).round(1)
subset_pct = (subset_counts / len(subset_df) * 100).round(1)

comparison = pd.DataFrame({
    'full_count': full_counts,
    'full_pct': full_pct,
    'subset_count': subset_counts,
    'subset_pct': subset_pct,
}).reindex(CLASS_NAMES).fillna(0)
comparison['pct_point_diff'] = (comparison['subset_pct'] - comparison['full_pct']).round(1)

print(comparison.to_string())
print("\n'pct_point_diff' > a few points in either direction means that class")
print("is over/under-represented in your sorted subset relative to the full dataset.")

# ─────────────────────────────────────────────
# 4. RULER PREVALENCE BY CLASS (ground truth only)
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("RULER PREVALENCE BY CLASS (ground-truth labels)")
print("=" * 70)

# Build image_id -> true class lookup (strip _downsampled and extension
# so folder filenames match regardless of exact naming variant)
def normalize_id(filename):
    base = os.path.splitext(filename)[0]
    base = base.replace('_downsampled', '')
    return base

id_to_class = dict(
    zip(df['image'].apply(normalize_id), df['class'])
)

def count_folder_by_class(folder_path):
    counts = {cls: 0 for cls in CLASS_NAMES}
    unmatched = 0
    if not os.path.isdir(folder_path):
        print(f"  [WARNING] Folder not found: {folder_path}")
        return counts, unmatched
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_id = normalize_id(fname)
        cls = id_to_class.get(img_id)
        if cls is None:
            unmatched += 1
        else:
            counts[cls] += 1
    return counts, unmatched

folder_class_counts = {}
for folder_name, folder_path in RULER_FOLDERS.items():
    counts, unmatched = count_folder_by_class(folder_path)
    folder_class_counts[folder_name] = counts
    total = sum(counts.values())
    print(f"\n{folder_name} ({folder_path}):")
    print(f"  Total matched: {total}, unmatched (not found in CSV): {unmatched}")
    for cls in CLASS_NAMES:
        if counts[cls] > 0:
            print(f"    {cls}: {counts[cls]}")

# ─────────────────────────────────────────────
# 5. RULER FRACTION PER CLASS
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("RULER FRACTION PER CLASS  =  (Normal_Ruler + Thick_Ruler) / (that + Clean)")
print("(Doubtful excluded entirely from this ratio)")
print("=" * 70)

results = []
for cls in CLASS_NAMES:
    ruler_n = folder_class_counts["Normal_Ruler"][cls] + folder_class_counts["Thick_Ruler"][cls]
    clean_n = folder_class_counts["Clean"][cls]
    denom = ruler_n + clean_n
    frac = (ruler_n / denom * 100) if denom > 0 else None
    results.append({
        'class': cls,
        'ruler_n': ruler_n,
        'clean_n': clean_n,
        'total_n': denom,
        'ruler_pct': round(frac, 1) if frac is not None else None,
    })

results_df = pd.DataFrame(results).set_index('class')
print(results_df.to_string())

print("\nCLASSES WITH FEW IMAGES (interpret their percentage with caution):")
LOW_N_THRESHOLD = 30
low_n = results_df[results_df['total_n'] < LOW_N_THRESHOLD]
if len(low_n) > 0:
    print(low_n.to_string())
else:
    print("  (none below threshold)")

print("\nDone.")