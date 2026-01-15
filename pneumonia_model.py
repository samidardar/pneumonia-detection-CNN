"""
Pneumonia Detection from Chest X-Ray Images using Deep Learning
================================================================

A comprehensive deep learning model for detecting pneumonia from chest X-ray images
using transfer learning with EfficientNetB0.

Author: AI Assistant
Date: January 2026
License: MIT

Usage:
    python pneumonia_model.py --data_dir <path_to_kaggle_dataset>

Dataset:
    Download from: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, 
    BatchNormalization, Input
)

# Metrics imports
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.0001
CLASSES = ['NORMAL', 'PNEUMONIA']
NUM_CLASSES = 2


def setup_gpu():
    """Configure GPU memory growth to avoid memory allocation issues."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("⚠ No GPU found, using CPU (training will be slower)")


def create_data_generators(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Create data generators with augmentation for training and validation.
    
    Args:
        data_dir: Path to the dataset directory
        img_size: Target image size
        batch_size: Batch size for training
    
    Returns:
        train_generator, val_generator, test_generator
    """
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation/Test data - only rescaling
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    print(f"\n📁 Loading data from: {data_dir}")
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        classes=CLASSES,
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        classes=CLASSES,
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        classes=CLASSES,
        shuffle=False
    )
    
    print(f"✓ Training samples: {train_generator.samples}")
    print(f"✓ Validation samples: {val_generator.samples}")
    print(f"✓ Test samples: {test_generator.samples}")
    
    return train_generator, val_generator, test_generator


def create_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE):
    """
    Create EfficientNetB0-based model for pneumonia detection.
    
    Args:
        img_size: Input image size
        num_classes: Number of output classes (2 for binary classification)
        learning_rate: Initial learning rate
    
    Returns:
        Compiled Keras model
    """
    print("\n🏗️ Building EfficientNetB0 model...")
    
    # Load pre-trained EfficientNetB0 (without top layers)
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Build custom classification head
    inputs = Input(shape=(img_size, img_size, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layer - binary classification
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    
    print("✓ Model created successfully")
    print(f"  - Base model: EfficientNetB0")
    print(f"  - Total parameters: {model.count_params():,}")
    print(f"  - Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    
    return model


def unfreeze_model(model, fine_tune_at=100, learning_rate=1e-5):
    """
    Unfreeze some layers of the base model for fine-tuning.
    
    Args:
        model: The compiled model
        fine_tune_at: Layer number from which to start fine-tuning
        learning_rate: Learning rate for fine-tuning (should be lower)
    
    Returns:
        Re-compiled model with unfrozen layers
    """
    print(f"\n🔓 Fine-tuning: Unfreezing layers from layer {fine_tune_at}...")
    
    # Get the base model
    base_model = model.layers[1]
    base_model.trainable = True
    
    # Freeze layers up to fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Re-compile with lower learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"✓ Trainable parameters after unfreezing: {trainable_params:,}")
    
    return model


def get_callbacks(model_save_path='pneumonia_model_best.h5'):
    """
    Create training callbacks for model checkpointing and optimization.
    
    Args:
        model_save_path: Path to save the best model
    
    Returns:
        List of callbacks
    """
    return [
        callbacks.ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=f'./logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]


def train_model(model, train_gen, val_gen, epochs=EPOCHS, callbacks_list=None):
    """
    Train the model with the provided data generators.
    
    Args:
        model: Compiled Keras model
        train_gen: Training data generator
        val_gen: Validation data generator
        epochs: Number of training epochs
        callbacks_list: List of training callbacks
    
    Returns:
        Training history
    """
    print(f"\n🚀 Starting training for {epochs} epochs...")
    
    # Calculate class weights for imbalanced data
    total_samples = train_gen.samples
    class_counts = np.bincount(train_gen.classes)
    class_weights = {
        0: total_samples / (2 * class_counts[0]),  # NORMAL
        1: total_samples / (2 * class_counts[1])   # PNEUMONIA
    }
    print(f"📊 Class weights: {class_weights}")
    
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=1
    )
    
    return history


def evaluate_model(model, test_gen, output_dir='./results'):
    """
    Comprehensive model evaluation with metrics and visualizations.
    
    Args:
        model: Trained Keras model
        test_gen: Test data generator
        output_dir: Directory to save evaluation results
    
    Returns:
        Dictionary of evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n📊 Evaluating model on test set...")
    
    # Get predictions
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    y_pred_proba = predictions.flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    y_true = test_gen.classes
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Model evaluation on test set
    test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(test_gen, verbose=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'test_loss': test_loss,
        'test_auc': test_auc
    }
    
    # Print results
    print("\n" + "="*50)
    print("           MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"  AUC:       {test_auc:.4f}")
    print(f"  Loss:      {test_loss:.4f}")
    print("="*50)
    
    # Classification Report
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    
    # Generate and save confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, CLASSES, output_dir)
    
    # Generate and save ROC curve
    plot_roc_curve(y_true, y_pred_proba, output_dir)
    
    # Save metrics to file
    save_metrics_report(metrics, y_true, y_pred, output_dir)
    
    return metrics


def plot_confusion_matrix(cm, classes, output_dir):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        annot_kws={'size': 16}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    print(f"✓ Confusion matrix saved to {output_dir}/confusion_matrix.png")


def plot_roc_curve(y_true, y_pred_proba, output_dir):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    plt.close()
    print(f"✓ ROC curve saved to {output_dir}/roc_curve.png")


def plot_training_history(history, output_dir):
    """Plot and save training history."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train', linewidth=2)
    axes[1, 0].plot(history.history['val_precision'], label='Validation', linewidth=2)
    axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train', linewidth=2)
    axes[1, 1].plot(history.history['val_recall'], label='Validation', linewidth=2)
    axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    plt.close()
    print(f"✓ Training history saved to {output_dir}/training_history.png")


def save_metrics_report(metrics, y_true, y_pred, output_dir):
    """Save metrics report to a text file."""
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("    PNEUMONIA DETECTION MODEL - EVALUATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("PERFORMANCE METRICS\n")
        f.write("-"*40 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)\n")
        f.write(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)\n")
        f.write(f"F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)\n")
        f.write(f"AUC:       {metrics['test_auc']:.4f}\n")
        f.write(f"Loss:      {metrics['test_loss']:.4f}\n\n")
        
        f.write("CLASSIFICATION REPORT\n")
        f.write("-"*40 + "\n")
        f.write(classification_report(y_true, y_pred, target_names=CLASSES))
        
    print(f"✓ Evaluation report saved to {report_path}")


def predict_single_image(model, image_path, img_size=IMG_SIZE):
    """
    Predict pneumonia from a single chest X-ray image.
    
    Args:
        model: Trained model
        image_path: Path to the X-ray image
        img_size: Target image size
    
    Returns:
        Prediction result and confidence
    """
    from tensorflow.keras.preprocessing import image
    
    # Load and preprocess image
    img = image.load_img(image_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Predict
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Interpret result
    if prediction > 0.5:
        result = "PNEUMONIA"
        confidence = prediction
    else:
        result = "NORMAL"
        confidence = 1 - prediction
    
    return result, confidence


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Pneumonia Detection from Chest X-Ray Images')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the Kaggle chest-xray-pneumonia dataset')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--model_path', type=str, default='pneumonia_model.h5',
                        help='Path to save the trained model')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Perform fine-tuning after initial training')
    parser.add_argument('--predict', type=str, default=None,
                        help='Path to a single image to predict (skip training)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("   PNEUMONIA DETECTION FROM CHEST X-RAY IMAGES")
    print("   Using Deep Learning with EfficientNetB0")
    print("="*60 + "\n")
    
    # Setup GPU
    setup_gpu()
    
    # Prediction mode
    if args.predict:
        print(f"🔍 Loading model from {args.model_path}...")
        model = load_model(args.model_path)
        result, confidence = predict_single_image(model, args.predict)
        print(f"\n📋 Prediction Result: {result}")
        print(f"   Confidence: {confidence:.2%}")
        return
    
    # Create output directory
    output_dir = './results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create data generators
    train_gen, val_gen, test_gen = create_data_generators(
        args.data_dir, 
        batch_size=args.batch_size
    )
    
    # Create model
    model = create_model()
    
    # Print model summary
    print("\n📐 Model Architecture:")
    model.summary()
    
    # Training Phase 1: Train with frozen base
    print("\n" + "="*50)
    print("   PHASE 1: Training with Frozen Base Model")
    print("="*50)
    
    callbacks_list = get_callbacks(args.model_path.replace('.h5', '_phase1.h5'))
    history1 = train_model(model, train_gen, val_gen, epochs=args.epochs, callbacks_list=callbacks_list)
    
    # Fine-tuning Phase 2 (optional)
    if args.fine_tune:
        print("\n" + "="*50)
        print("   PHASE 2: Fine-tuning")
        print("="*50)
        
        model = unfreeze_model(model, fine_tune_at=100)
        callbacks_list = get_callbacks(args.model_path)
        history2 = train_model(model, train_gen, val_gen, epochs=10, callbacks_list=callbacks_list)
        
        # Combine histories
        for key in history1.history:
            history1.history[key].extend(history2.history[key])
    
    # Plot training history
    plot_training_history(history1, output_dir)
    
    # Evaluate model
    metrics = evaluate_model(model, test_gen, output_dir)
    
    # Save final model
    model.save(args.model_path)
    print(f"\n✓ Model saved to {args.model_path}")
    
    print("\n" + "="*50)
    print("   TRAINING COMPLETE!")
    print("="*50)
    print(f"\nResults saved to: {output_dir}/")
    print(f"Model saved to: {args.model_path}")
    
    return model, history1, metrics


if __name__ == "__main__":
    main()
