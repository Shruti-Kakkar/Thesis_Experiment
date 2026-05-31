"""
train.py
ResNet50V2 fine-tuning on ISIC 2019 for 8-class skin lesion classification.

Architecture: ResNet50V2 pretrained on ImageNet (De Santis et al., 2024)
Dataset: ISIC 2019 (Cassidy et al., 2022; Kassem et al., 2020)
Author: Shruti Kakkar
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ─────────────────────────────────────────────
# 1. PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_IMG_DIR = os.path.join(BASE_DIR, "datasets", "ISIC_2019_Training_Input",
                              "ISIC_2019_Training_Input")
TEST_IMG_DIR  = os.path.join(BASE_DIR, "datasets", "ISIC_2019_Test_Input",
                              "ISIC_2019_Test_Input")
TRAIN_CSV     = os.path.join(BASE_DIR, "datasets", "ISIC_2019_Training_GroundTruth.csv")
TEST_CSV      = os.path.join(BASE_DIR, "datasets", "ISIC_2019_Test_GroundTruth.csv")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 2. CONSTANTS
# ─────────────────────────────────────────────
IMG_SIZE    = 224          # ResNet50V2 input size
BATCH_SIZE  = 32
EPOCHS_P1   = 10           # Phase 1: frozen base
EPOCHS_P2   = 40           # Phase 2: fine-tuning
SEED        = 42
RUN_TAG     = "_run2"      # suffix appended to all output filenames

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
NUM_CLASSES = len(CLASS_NAMES)

# ─────────────────────────────────────────────
# 3. LOAD AND PREPARE DATA
# ─────────────────────────────────────────────
print("Loading training CSV...")
train_df = pd.read_csv(TRAIN_CSV)

# Get label (argmax of one-hot, only first 8 columns = known classes)
train_df['label'] = train_df[CLASS_NAMES].values.argmax(axis=1)
train_df['filepath'] = train_df['image'].apply(
    lambda x: os.path.join(TRAIN_IMG_DIR, x + '.jpg'))

# Verify files exist
train_df = train_df[train_df['filepath'].apply(os.path.exists)]
print(f"Training images found: {len(train_df)}")

# Train / validation split (90/10)
train_data, val_data = train_test_split(
    train_df, test_size=0.1, random_state=SEED,
    stratify=train_df['label'])

print(f"Train: {len(train_data)} | Val: {len(val_data)}")

# ─────────────────────────────────────────────
# 4. CLASS WEIGHTS (handle imbalance)
# ─────────────────────────────────────────────
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(NUM_CLASSES),
    y=train_data['label'].values)
class_weights = dict(enumerate(class_weights_array))
print("Class weights:", class_weights)

# ─────────────────────────────────────────────
# 5. DATA PIPELINE
# ─────────────────────────────────────────────
augmentation_layer = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

def load_and_preprocess(filepath, label, augment=False):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.keras.applications.resnet_v2.preprocess_input(img)

    if augment:
        img = augmentation_layer(img, training=True)

    label = tf.one_hot(label, NUM_CLASSES)
    return img, label

def make_dataset(df, augment=False, shuffle=False):
    filepaths = df['filepath'].values
    labels    = df['label'].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED)
    ds = ds.map(lambda fp, lb: load_and_preprocess(fp, lb, augment),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_data, augment=True,  shuffle=True)
val_ds   = make_dataset(val_data,   augment=False, shuffle=False)

# ─────────────────────────────────────────────
# 6. LOAD TEST DATA
# ─────────────────────────────────────────────
print("Loading test CSV...")
test_df = pd.read_csv(TEST_CSV)

# Remove UNK images
test_df = test_df[test_df['UNK'] != 1.0].copy()
test_df['label'] = test_df[CLASS_NAMES].values.argmax(axis=1)
test_df['filepath'] = test_df['image'].apply(
    lambda x: os.path.join(TEST_IMG_DIR, x + '.jpg'))
test_df = test_df[test_df['filepath'].apply(os.path.exists)]
print(f"Test images (known labels): {len(test_df)}")

test_ds = make_dataset(test_df, augment=False, shuffle=False)
# ─────────────────────────────────────────────
# 7. BUILD MODEL
# ─────────────────────────────────────────────
def build_model():
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False  # freeze for Phase 1

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.6)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model, base_model

model, base_model = build_model()
model.summary()

# ─────────────────────────────────────────────
# 8. CALLBACKS
# ─────────────────────────────────────────────
def get_callbacks(phase):
    return [
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            os.path.join(MODEL_DIR, f'resnet50v2_phase{phase}_best{RUN_TAG}.keras'),
            monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1)
    ]

# ─────────────────────────────────────────────
# 9. PHASE 1 — TRAIN HEAD ONLY
# ─────────────────────────────────────────────
print("\n--- Phase 1: Training classification head ---")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

history_p1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_P1,
    class_weight=class_weights,
    callbacks=get_callbacks(1),
    verbose=1)

# ─────────────────────────────────────────────
# 10. PHASE 2 — FINE-TUNE TOP LAYERS
# ─────────────────────────────────────────────
print("\n--- Phase 2: Fine-tuning top layers ---")

# Unfreeze top 30 layers of base model
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

history_p2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_P2,
    class_weight=class_weights,
    callbacks=get_callbacks(2),
    verbose=1)

# ─────────────────────────────────────────────
# 11. SAVE FINAL MODEL
# ─────────────────────────────────────────────
final_model_path = os.path.join(MODEL_DIR, f'resnet50v2_isic2019_final{RUN_TAG}.keras')
model.save(final_model_path)
print(f"\nFinal model saved to: {final_model_path}")

# ─────────────────────────────────────────────
# 12. PLOT TRAINING CURVES
# ─────────────────────────────────────────────
def plot_history(h1, h2, output_dir):
    acc  = h1.history['accuracy']      + h2.history['accuracy']
    val  = h1.history['val_accuracy']  + h2.history['val_accuracy']
    loss = h1.history['loss']          + h2.history['loss']
    vloss= h1.history['val_loss']      + h2.history['val_loss']

    epochs = range(1, len(acc) + 1)
    p1_end = len(h1.history['accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, acc,  label='Train Accuracy')
    ax1.plot(epochs, val,  label='Val Accuracy')
    ax1.axvline(x=p1_end, color='gray', linestyle='--', label='Phase 1→2')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(epochs, loss,  label='Train Loss')
    ax2.plot(epochs, vloss, label='Val Loss')
    ax2.axvline(x=p1_end, color='gray', linestyle='--', label='Phase 1→2')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_curves{RUN_TAG}.png'), dpi=150)
    print("Training curves saved.")

plot_history(history_p1, history_p2, OUTPUT_DIR)

# ─────────────────────────────────────────────
# 13. EVALUATE ON TEST SET
# ─────────────────────────────────────────────
print("\n--- Evaluating on official test set ---")

y_true = test_df['label'].values
y_pred = model.predict(test_ds, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# Classification report
report = classification_report(y_true, y_pred_classes,
                                target_names=CLASS_NAMES)
print(report)

with open(os.path.join(OUTPUT_DIR, f'classification_report{RUN_TAG}.txt'), 'w') as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix — ResNet50V2 on ISIC 2019')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix{RUN_TAG}.png'), dpi=150)
print("Confusion matrix saved.")
print("\nDone!")