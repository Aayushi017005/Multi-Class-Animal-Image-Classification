"""
train.py - MobileNetV2 Transfer Learning Training Script for Animal Classification

This script:
1. Loads strictly separated datasets: Training (350), Validation (75), and Testing (75).
2. Uses MobileNetV2 pre-trained on ImageNet as a feature extractor.
3. Adds in-model data augmentation and normalization ([-1, 1]).
4. Attaches a custom classification head (GlobalAveragePooling2D, Dropout, Dense, Softmax).
5. Trains the classification head with frozen base weights, followed by optional fine-tuning.
6. Employs EarlyStopping to preserve the best checkpoint weights.
7. Generates and saves:
   - Updated model: 'animal_classifier.keras'
   - Class mappings: 'class_names.json'
   - Learning curves: 'training_history.png'
   - Confusion matrix heatmap: 'confusion_matrix.png'
   - Scikit-learn Classification Report on unseen test samples
"""

import os
import json
import tempfile
import numpy as np

# Suppress minor TF & Matplotlib warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")
os.environ["KERAS_HOME"] = os.path.abspath(".keras_cache")

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Configuration Constants
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
INITIAL_EPOCHS = 12
FINE_TUNE_EPOCHS = 8
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

TRAIN_DIR = os.path.join("dataset", "train")
VAL_DIR = os.path.join("dataset", "val")
TEST_DIR = os.path.join("dataset", "test")

MODEL_SAVE_PATH = "animal_classifier.keras"
LABELS_SAVE_PATH = "class_names.json"
HISTORY_PLOT_PATH = "training_history.png"
CONFUSION_MATRIX_PATH = "confusion_matrix.png"


def load_datasets():
    """Load and separate Training, Validation, and Testing image datasets."""
    for path, name in [(TRAIN_DIR, "Training"), (VAL_DIR, "Validation"), (TEST_DIR, "Testing")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} directory '{path}' not found. Run 'python download_data.py' first."
            )

    print("Loading Training dataset...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )

    print("Loading Validation dataset...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Loading Testing dataset...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = train_ds.class_names
    print(f"\nClasses detected ({len(class_names)}): {class_names}")

    # Optimize pipeline with prefetching
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def build_mobilenetv2_model(input_shape=(128, 128, 3), num_classes=5):
    """
    Build Transfer Learning model using pre-trained MobileNetV2.
    - Data Augmentation (active during fit)
    - Normalization: scales [0, 255] -> [-1, 1] as expected by MobileNetV2
    - Pre-trained MobileNetV2 base (ImageNet weights, top removed, initially frozen)
    - Custom Classification Head: GlobalAveragePooling2D, Dropout, Dense, Softmax
    """
    # 1. Input layer
    inputs = layers.Input(shape=input_shape)

    # 2. Data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
    ], name="data_augmentation")
    x = data_augmentation(inputs)

    # 3. Rescaling for MobileNetV2 (maps [0, 255] to [-1, 1])
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(x)

    # 4. MobileNetV2 base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze pretrained feature extractor

    # Run in inference mode so batchnorm statistics are frozen
    x = base_model(x, training=False)

    # 5. Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="mobilenetv2_animal_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


def plot_and_save_history(history, save_path=HISTORY_PLOT_PATH):
    """Plot and save training and validation accuracy & loss curves."""
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4.5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy", marker="o", color="#2E86AB")
    plt.plot(epochs_range, val_acc, label="Val Accuracy", marker="s", color="#A23B72")
    plt.title("MobileNetV2 Transfer Learning Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss", marker="o", color="#2E86AB")
    plt.plot(epochs_range, val_loss, label="Val Loss", marker="s", color="#F18F01")
    plt.title("MobileNetV2 Transfer Learning Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history plot saved to '{save_path}'.")


def plot_and_save_confusion_matrix(cm, class_names, save_path=CONFUSION_MATRIX_PATH):
    """Render and save confusion matrix heatmap using Matplotlib."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (MobileNetV2)")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    display_names = [c.capitalize() for c in class_names]
    plt.xticks(tick_marks, display_names, rotation=45)
    plt.yticks(tick_marks, display_names)

    # Display count values inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, format(val, "d"), horizontalalignment="center", color=color, fontweight="bold")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved to '{save_path}'.")


def evaluate_test_set(model, test_ds, class_names):
    """Run full test evaluation, generating classification report and confusion matrix."""
    print("\n" + "=" * 60)
    print("Evaluating MobileNetV2 on Independent Test Set")
    print("=" * 60)

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Final Test Loss:     {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%\n")

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    display_names = [c.capitalize() for c in class_names]

    # Classification Report
    cr = classification_report(y_true, y_pred, target_names=display_names, digits=4, zero_division=0)
    print("Classification Report:")
    print("-" * 60)
    print(cr)
    print("-" * 60)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Save Confusion Matrix figure
    plot_and_save_confusion_matrix(cm, class_names, CONFUSION_MATRIX_PATH)

    return test_loss, test_acc, cr, cm


def main():
    print("=" * 60)
    print("Multi-Class Animal Classification - MobileNetV2 Pipeline")
    print("=" * 60)

    # 1. Load Data
    train_ds, val_ds, test_ds, class_names = load_datasets()

    # 2. Build MobileNetV2 Transfer Learning Model
    model, base_model = build_mobilenetv2_model(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        num_classes=len(class_names)
    )
    model.summary()

    # 3. Setup EarlyStopping callback
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # 4. Phase 1: Train Classification Head (Feature Extraction)
    print(f"\n--- Phase 1: Training Classification Head ({INITIAL_EPOCHS} epochs) ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=[early_stopping],
        verbose=1
    )

    # 5. Phase 2: Fine-Tuning Top Layers of MobileNetV2
    print("\n--- Phase 2: Fine-Tuning Top Layers of MobileNetV2 ---")
    base_model.trainable = True

    # Freeze all layers except the last 25 layers
    fine_tune_at = len(base_model.layers) - 25
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    print(f"Fine-tuning {len(base_model.layers) - fine_tune_at} layers of MobileNetV2...")

    # Re-compile with lower learning rate for gentle fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    fine_tune_early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TOTAL_EPOCHS,
        initial_epoch=len(history.history["loss"]),
        callbacks=[fine_tune_early_stopping],
        verbose=1
    )

    # Merge history
    for key in history.history:
        history.history[key].extend(history_fine.history.get(key, []))

    # 6. Save updated model and class labels
    model.save(MODEL_SAVE_PATH)
    print(f"\nUpdated model successfully saved to '{MODEL_SAVE_PATH}'.")

    with open(LABELS_SAVE_PATH, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved to '{LABELS_SAVE_PATH}'.")

    # 7. Save Training History Curves
    plot_and_save_history(history, HISTORY_PLOT_PATH)

    # 8. Evaluate on Independent Test Set
    evaluate_test_set(model, test_ds, class_names)

    print("\nMobileNetV2 training and evaluation workflow completed successfully!")


if __name__ == "__main__":
    main()
