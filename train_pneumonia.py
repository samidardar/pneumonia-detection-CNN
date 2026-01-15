# Pneumonia Detection Model Training Script
# ==========================================
# Complete training pipeline for chest X-ray pneumonia detection.
#
# Usage: python train_pneumonia.py
# Dataset path: C:\Users\PC\Downloads\archive (6)

import os
import sys
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 60)
print("   PNEUMONIA DETECTION FROM CHEST X-RAY IMAGES")
print("   Training with EfficientNetB0 Transfer Learning")
print("=" * 60)
print()

# Install required packages if needed
try:
    import tensorflow as tf
except ImportError:
    print("Installing TensorFlow...")
    os.system('pip install tensorflow-cpu --quiet')
    import tensorflow as tf

try:
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
except ImportError:
    print("Installing scikit-learn...")
    os.system('pip install scikit-learn --quiet')
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# =============================================================================
# CONFIGURATION
# =============================================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0001
CLASSES = ['NORMAL', 'PNEUMONIA']

# Dataset path - UPDATE THIS IF NEEDED
DATASET_PATH = r"C:\Users\PC\Downloads\archive (6)"

# Find chest_xray folder
DATA_DIR = None
for item in os.listdir(DATASET_PATH):
    item_path = os.path.join(DATASET_PATH, item)
    if os.path.isdir(item_path):
        if 'train' in os.listdir(item_path):
            DATA_DIR = item_path
            break
        elif item == 'chest_xray':
            DATA_DIR = item_path
            break

if DATA_DIR is None:
    # Check if train is directly in DATASET_PATH
    if 'train' in os.listdir(DATASET_PATH):
        DATA_DIR = DATASET_PATH
    else:
        print(f"ERROR: Could not find chest_xray dataset in {DATASET_PATH}")
        print("Please ensure the dataset has train/val/test folders")
        sys.exit(1)

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Output directory
OUTPUT_DIR = r"c:\Users\PC\Downloads\New folder\pneumonia_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n📁 Dataset: {DATA_DIR}")
print(f"📂 Output: {OUTPUT_DIR}")

# =============================================================================
# DATA LOADING
# =============================================================================
print("\n" + "=" * 50)
print("   LOADING DATA")
print("=" * 50)

# Count images
train_normal = len(os.listdir(os.path.join(TRAIN_DIR, 'NORMAL')))
train_pneumonia = len(os.listdir(os.path.join(TRAIN_DIR, 'PNEUMONIA')))
test_total = sum([len(os.listdir(os.path.join(TEST_DIR, c))) for c in CLASSES])

print(f"\n📊 Dataset Statistics:")
print(f"   Training - NORMAL: {train_normal}")
print(f"   Training - PNEUMONIA: {train_pneumonia}")
print(f"   Test: {test_total}")

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.85, 1.15],
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

print("\n📥 Loading training data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=CLASSES,
    shuffle=True
)

print("📥 Loading validation data...")
val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=CLASSES,
    shuffle=False
)

print("📥 Loading test data...")
test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=CLASSES,
    shuffle=False
)

# =============================================================================
# BUILD MODEL
# =============================================================================
print("\n" + "=" * 50)
print("   BUILDING MODEL")
print("=" * 50)

def create_model():
    """Create EfficientNetB0-based model."""
    print("\n🏗️ Loading EfficientNetB0...")
    
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False
    
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    
    return model, base_model

model, base_model = create_model()

total_params = model.count_params()
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
print(f"\n✅ Model created!")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")

# =============================================================================
# TRAIN MODEL - PHASE 1
# =============================================================================
print("\n" + "=" * 50)
print("   PHASE 1: TRAINING WITH FROZEN BASE")
print("=" * 50)

# Class weights
total = train_generator.samples
class_counts = np.bincount(train_generator.classes)
class_weights = {
    0: total / (2 * class_counts[0]),
    1: total / (2 * class_counts[1])
}
print(f"\n📊 Class weights: NORMAL={class_weights[0]:.2f}, PNEUMONIA={class_weights[1]:.2f}")

# Callbacks
model_path = os.path.join(OUTPUT_DIR, 'pneumonia_model_phase1.keras')
callbacks_list = [
    callbacks.ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

print(f"\n🚀 Training for {EPOCHS} epochs...")
history1 = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_list,
    class_weight=class_weights,
    verbose=1
)

# =============================================================================
# TRAIN MODEL - PHASE 2 (FINE-TUNING)
# =============================================================================
print("\n" + "=" * 50)
print("   PHASE 2: FINE-TUNING")
print("=" * 50)

# Unfreeze top layers
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with lower LR
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.AUC(name='auc')]
)

trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
print(f"🔓 Trainable parameters: {trainable_params:,}")

# Fine-tune callbacks
model_path_ft = os.path.join(OUTPUT_DIR, 'pneumonia_model_final.keras')
callbacks_list2 = [
    callbacks.ModelCheckpoint(
        model_path_ft,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

print(f"\n🚀 Fine-tuning for 10 epochs...")
history2 = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=callbacks_list2,
    class_weight=class_weights,
    verbose=1
)

# =============================================================================
# EVALUATION
# =============================================================================
print("\n" + "=" * 50)
print("   MODEL EVALUATION")
print("=" * 50)

# Predictions
print("\n📊 Evaluating on test set...")
test_generator.reset()
predictions = model.predict(test_generator, verbose=1)
y_pred_proba = predictions.flatten()
y_pred = (y_pred_proba > 0.5).astype(int)
y_true = test_generator.classes

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(test_generator, verbose=0)

print("\n" + "=" * 50)
print("         FINAL RESULTS")
print("=" * 50)
print(f"  ✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  ✅ Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"  ✅ Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"  ✅ F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
print(f"  ✅ AUC:       {test_auc:.4f}")
print(f"  ✅ Loss:      {test_loss:.4f}")
print("=" * 50)

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "=" * 50)
print("   SAVING RESULTS")
print("=" * 50)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES, annot_kws={'size': 20})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=14)
plt.xlabel('Predicted Label', fontsize=14)
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"✅ Saved: {cm_path}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve', fontsize=16, fontweight='bold')
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(OUTPUT_DIR, 'roc_curve.png')
plt.savefig(roc_path, dpi=150)
plt.close()
print(f"✅ Saved: {roc_path}")

# Training History
history = {}
for key in history1.history:
    history[key] = history1.history[key] + history2.history[key]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].plot(history['accuracy'], label='Train')
axes[0, 0].plot(history['val_accuracy'], label='Validation')
axes[0, 0].set_title('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(history['loss'], label='Train')
axes[0, 1].plot(history['val_loss'], label='Validation')
axes[0, 1].set_title('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(history['precision'], label='Train')
axes[1, 0].plot(history['val_precision'], label='Validation')
axes[1, 0].set_title('Precision')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(history['recall'], label='Train')
axes[1, 1].plot(history['val_recall'], label='Validation')
axes[1, 1].set_title('Recall')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
hist_path = os.path.join(OUTPUT_DIR, 'training_history.png')
plt.savefig(hist_path, dpi=150)
plt.close()
print(f"✅ Saved: {hist_path}")

# Save model
model.save(os.path.join(OUTPUT_DIR, 'pneumonia_model_final.keras'))
model.save(os.path.join(OUTPUT_DIR, 'pneumonia_model_final.h5'))
print(f"✅ Saved: pneumonia_model_final.keras")
print(f"✅ Saved: pneumonia_model_final.h5")

# Evaluation report
report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.txt')
with open(report_path, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("    PNEUMONIA DETECTION MODEL - EVALUATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("PERFORMANCE METRICS\n")
    f.write("-" * 40 + "\n")
    f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"Precision: {precision:.4f} ({precision*100:.2f}%)\n")
    f.write(f"Recall:    {recall:.4f} ({recall*100:.2f}%)\n")
    f.write(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)\n")
    f.write(f"AUC:       {test_auc:.4f}\n")
    f.write(f"Loss:      {test_loss:.4f}\n\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("-" * 40 + "\n")
    f.write(classification_report(y_true, y_pred, target_names=CLASSES))
print(f"✅ Saved: {report_path}")

# =============================================================================
# DONE
# =============================================================================
print("\n" + "=" * 60)
print("   🎉 TRAINING COMPLETE!")
print("=" * 60)
print(f"\n📁 All results saved to: {OUTPUT_DIR}")
print("\nFiles created:")
print("   - pneumonia_model_final.keras (trained model)")
print("   - pneumonia_model_final.h5 (trained model)")
print("   - confusion_matrix.png")
print("   - roc_curve.png")
print("   - training_history.png")
print("   - evaluation_report.txt")
print("\n" + "=" * 60)
