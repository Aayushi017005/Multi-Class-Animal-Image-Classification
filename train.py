"""
train.py - Upgraded Model Training Script for Multi-Class Animal Image Classification

Enhancements:
1. Strict 3-way dataset separation: Training (350), Validation (75), and Testing (75) sets.
2. Lightweight, effective in-model Data Augmentation (RandomFlip, subtle RandomZoom).
3. Improved CNN architecture with 4 progressive Conv2D+MaxPool stages (32->64->128->128),
   Dropout regularization, and a compact Dense head (~1.1M parameters).
4. EarlyStopping callback monitoring val_loss with restore_best_weights=True.
5. Generates:
   - Final Test Loss & Accuracy
   - Scikit-learn Classification Report (Precision, Recall, F1-Score)
   - Confusion Matrix (printed and saved as 'confusion_matrix.png')
   - Learning curves (saved as 'training_history.png')
6. Exports updated model ('animal_classifier.keras') and class labels ('class_names.json').
"""

import os
import json
import tempfile
import numpy as np

# Suppress minor TF & Matplotlib warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Configuration Constants
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 18
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
    print(f"\nDiscovered {len(class_names)} classes: {class_names}")

    # Optimize data pipeline caching & prefetching
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def build_improved_cnn(input_shape=(128, 128, 3), num_classes=5):
    """
    Build an improved CNN architecture for animal image classification:
    - Data Augmentation: Random horizontal flip, subtle zoom
    - Rescaling: Normalizes pixel values [0, 255] -> [0, 1]
    - 4 progressive Conv2D+MaxPool blocks (32 -> 64 -> 128 -> 128)
    - Fully connected classification head with Dropout (0.5) to combat overfitting
    - Softmax output layer
    """
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(0.08),
    ], name="data_augmentation")

    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Augmentation & Normalization
        data_augmentation,
        layers.Rescaling(1.0 / 255.0),

        # Block 1: 128x128 -> 64x64
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Block 2: 64x64 -> 32x32
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Block 3: 32x32 -> 16x16
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Block 4: 16x16 -> 8x8
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Classification Head (8*8*128 = 8192 features -> 128 -> 5)
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ], name="improved_animal_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


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
    plt.title("Model Accuracy across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss", marker="o", color="#2E86AB")
    plt.plot(epochs_range, val_loss, label="Val Loss", marker="s", color="#F18F01")
    plt.title("Model Loss across Epochs")
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
    plt.title("Confusion Matrix on Test Set")
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
    print("Evaluating Model on Independent Test Set")
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
    print("Multi-Class Animal Image Classification - Retraining Pipeline")
    print("=" * 60)

    # 1. Load Data with separate Train, Val, Test sets
    train_ds, val_ds, test_ds, class_names = load_datasets()

    # 2. Build Improved Model with Data Augmentation
    model = build_improved_cnn(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        num_classes=len(class_names)
    )
    model.summary()

    # 3. Setup EarlyStopping callback
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    )

    # 4. Train Model using validation set for monitoring
    print(f"\nStarting model training (up to {EPOCHS} epochs with EarlyStopping)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        verbose=1
    )

    # 5. Save updated model and class labels
    model.save(MODEL_SAVE_PATH)
    print(f"\nUpdated model successfully saved to '{MODEL_SAVE_PATH}'.")

    with open(LABELS_SAVE_PATH, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved to '{LABELS_SAVE_PATH}'.")

    # 6. Save Training History Curves
    plot_and_save_history(history, HISTORY_PLOT_PATH)

    # 7. Evaluate on Independent Test Set
    evaluate_test_set(model, test_ds, class_names)

    print("\nRetraining and evaluation workflow completed successfully!")


if __name__ == "__main__":
    main()
