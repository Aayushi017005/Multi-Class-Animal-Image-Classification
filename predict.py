"""
predict.py - Image Prediction Module & CLI for Animal Image Classification

This script:
1. Loads the trained Keras CNN model ('animal_classifier.keras') and class labels ('class_names.json').
2. Preprocesses the input image by resizing to 128x128.
3. Performs inference and calculates class confidence percentages.
4. Handles errors such as missing model files, non-existent images, or corrupted file formats.
5. Can be executed from the terminal as a CLI or imported into app.py.
"""

import os
import sys
import json
import argparse
import tempfile
from PIL import Image, UnidentifiedImageError
import numpy as np

# Suppress minor TF & Matplotlib logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "animal_classifier.keras")
LABELS_PATH = os.path.join(BASE_DIR, "class_names.json")
TARGET_SIZE = (128, 128)

# Global cache for loaded model and labels to avoid reloading repeatedly
_cached_model = None
_cached_labels = None


def load_model_and_labels(model_path=MODEL_PATH, labels_path=LABELS_PATH):
    """Load and cache the trained Keras model and class names list."""
    global _cached_model, _cached_labels

    if _cached_model is not None and _cached_labels is not None:
        return _cached_model, _cached_labels

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run 'python train.py' first to train and save the model."
        )

    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Class labels file not found at '{labels_path}'. "
            "Please run 'python train.py' to generate the labels."
        )

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Load class labels
    with open(labels_path, "r") as f:
        labels = json.load(f)

    _cached_model = model
    _cached_labels = labels
    return model, labels


def preprocess_image(image_input):
    """
    Validate, load, convert to RGB, and resize input image to (128, 128).
    Accepts:
    - string path to image
    - PIL.Image object
    - file-like object / BytesIO
    """
    try:
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file does not exist: '{image_input}'")
            img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            # File-like object (e.g. from Streamlit file uploader)
            img = Image.open(image_input)

        # Convert to 3-channel RGB (handles RGBA, Grayscale, CMYK, etc.)
        img = img.convert("RGB")

        # Resize to target input dimensions (128, 128)
        img_resized = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)

        # Convert to NumPy array and add batch dimension (1, 128, 128, 3)
        img_array = np.array(img_resized, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)

        return img, img_batch

    except UnidentifiedImageError:
        raise ValueError("The provided file is not a valid or readable image format.")
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")


def predict_animal(image_input, model_path=MODEL_PATH, labels_path=LABELS_PATH, model=None, class_names=None):
    """
    Predict the animal class for the given image input.
    Can accept preloaded (cached) model and class_names to optimize memory in web frameworks like Streamlit.
    Returns:
        dict: {
            "predicted_label": str (Capitalized animal name),
            "confidence_percent": float (e.g. 95.4),
            "probabilities": dict {class_name: confidence_percentage},
            "original_image": PIL.Image
        }
    """
    if model is None or class_names is None:
        model, class_names = load_model_and_labels(model_path, labels_path)
    original_img, processed_batch = preprocess_image(image_input)

    # Model inference (Note: model has Rescaling layer built-in)
    predictions = model.predict(processed_batch, verbose=0)[0]

    # Find highest probability class
    top_index = int(np.argmax(predictions))
    top_label = class_names[top_index].capitalize()
    confidence = float(predictions[top_index]) * 100.0

    # Build dictionary of all class probabilities
    probabilities = {
        class_names[i].capitalize(): round(float(prob) * 100.0, 2)
        for i, prob in enumerate(predictions)
    }

    return {
        "predicted_label": top_label,
        "confidence_percent": round(confidence, 2),
        "probabilities": probabilities,
        "original_image": original_img
    }


def main():
    parser = argparse.ArgumentParser(description="Predict animal category for an image.")
    parser.add_argument("image_path", type=str, help="Path to the animal image file.")
    args = parser.parse_args()

    try:
        result = predict_animal(args.image_path)
        print("=" * 45)
        print("Animal Image Prediction Result")
        print("=" * 45)
        print(f"Predicted Animal:    {result['predicted_label']}")
        print(f"Confidence:          {result['confidence_percent']:.2f}%\n")
        print("All Category Confidences:")
        for animal, prob in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {animal:<12}: {prob:.2f}%")
        print("=" * 45)
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
