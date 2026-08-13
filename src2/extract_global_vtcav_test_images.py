"""
extract_global_tcav_test_images.py
Determines the EXACT set of images GlobalVisualTCAV used as test images
for MEL and NV -- not an approximation -- so ground-truth ruler sorting
targets the correct, precise set of ~200 + ~200 files.

WHY THIS ISN'T JUST "SORT THE FIRST 200 ALPHABETICALLY":
ImageActivationGenerator.get_images_for_concept (in VisualTCAV.py) does
NOT sort filenames before slicing to max_examples -- it uses
tf.io.gfile.listdir() directly:

    img_paths = [os.path.join(concept_dir, d)
                 for d in tf.io.gfile.listdir(concept_dir) if is_image(d)]
    imgs = self._load_images_from_files(img_paths, self.max_examples, ...)

Filesystem directory listing order is not guaranteed alphabetical. This
script calls tf.io.gfile.listdir() the exact same way, on the exact same
directories, so the list it produces is guaranteed identical to what
Global TCAV actually used -- as long as the folder contents haven't
changed since that run (a reasonable assumption for a static dataset).

Author: Shruti Kakkar
"""

import os
import tensorflow as tf

PROJECT_ROOT = os.path.expanduser(
    "~/scratch/dev-uos/projects/VTCAV_Dermatology"
)
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "datasets", "test_images_by_class")
MAX_EXAMPLES = 200  # must match the Global script's Model(max_examples=...)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs2", "ruler_bias_test_image_lists")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_image(filename):
    for ext in ["jpg", "jpeg", "png", "gif", "bmp"]:
        if filename.lower().endswith(ext):
            return True
    return False

print("=" * 60)
print("Extracting EXACT test images used by GlobalVisualTCAV")
print("=" * 60)

for class_name in ["MEL", "NV"]:
    class_dir = os.path.join(TEST_IMAGES_DIR, class_name)

    # EXACT replication of ImageActivationGenerator.get_images_for_concept's
    # file listing -- same function, same order, same slice.
    all_files = [f for f in tf.io.gfile.listdir(class_dir) if is_image(f)]
    used_files = all_files[:MAX_EXAMPLES]

    print(f"\n{class_name}:")
    print(f"  Total images in folder: {len(all_files)}")
    print(f"  Actually used by Global TCAV (max_examples={MAX_EXAMPLES}): {len(used_files)}")

    out_path = os.path.join(OUTPUT_DIR, f"{class_name}_test_images_used.txt")
    with open(out_path, 'w') as f:
        for fname in used_files:
            f.write(fname + "\n")
    print(f"  Saved list: {out_path}")

print(f"\n{'='*60}")
print("Done. These files are the exact ground-truth-sorting target --")
print("not an approximation. Sort these ~400 files (Normal_Ruler / Thick_Ruler")
print("/ Clean / Doubtful, same categories as before) to close the gap in the")
print("Global TCAV ruler-bias result.")
print(f"{'='*60}")