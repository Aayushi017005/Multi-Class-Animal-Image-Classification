"""
app.py - Streamlit Web Application for Multi-Class Animal Image Classification

This web app allows users to:
1. Upload an image of an animal (Cat, Dog, Elephant, Horse, Butterfly).
2. Or select a sample test image for quick testing.
3. View the predicted animal category and confidence score.
4. View probability distribution across all 5 animal classes.
5. Handle invalid files and error cases gracefully.
"""

import os
import streamlit as st
from PIL import Image

# Import prediction logic
from predict import predict_animal, load_model_and_labels, MODEL_PATH, LABELS_PATH

# Streamlit Page Configuration
st.set_page_config(
    page_title="Multi-Class Animal Classifier",
    page_icon="🐾",
    layout="centered"
)

st.title("🐾 Multi-Class Animal Image Classification")
st.markdown(
    """
    Upload an image of an animal to predict its category using a custom-trained **Convolutional Neural Network (CNN)**.
    
    **Supported Animals:** `Butterfly`, `Cat`, `Dog`, `Elephant`, `Horse`.
    """
)

# Verify and cache model and labels
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    st.error(
        "⚠️ Trained model not found! Please run the training script first:\n\n"
        "```bash\npython train.py\n```"
    )
    st.stop()


@st.cache_resource(show_spinner="Loading trained model into memory...")
def get_model_and_labels():
    return load_model_and_labels()


model, class_names = get_model_and_labels()

# Sidebar: Quick options and sample images
st.sidebar.header("Options & Sample Images")
use_sample = st.sidebar.checkbox("Use a sample test image instead", value=False)

selected_sample_path = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(BASE_DIR, "dataset", "test")

if use_sample and os.path.exists(test_dir):
    categories = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    if categories:
        chosen_cat = st.sidebar.selectbox("Choose Animal Category", categories)
        cat_folder = os.path.join(test_dir, chosen_cat)
        sample_files = [f for f in os.listdir(cat_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if sample_files:
            chosen_file = st.sidebar.selectbox("Choose Sample Image", sample_files)
            selected_sample_path = os.path.join(cat_folder, chosen_file)

# Main interface: File uploader
uploaded_file = st.file_uploader(
    "Upload an animal photo (JPG, JPEG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload an image of a Cat, Dog, Elephant, Horse, or Butterfly."
)

target_image = None
image_source_name = ""

if selected_sample_path and use_sample:
    target_image = selected_sample_path
    image_source_name = f"Sample: {os.path.basename(selected_sample_path)}"
elif uploaded_file is not None:
    target_image = uploaded_file
    image_source_name = uploaded_file.name

if target_image is not None:
    st.markdown("---")
    col1, col2 = st.columns([1, 1], gap="medium")

    try:
        # Perform classification
        with st.spinner("Analyzing image with CNN model..."):
            result = predict_animal(target_image, model=model, class_names=class_names)

        with col1:
            st.subheader("Input Image")
            st.image(result["original_image"], caption=image_source_name, use_container_width=True)

        with col2:
            st.subheader("Prediction Result")
            st.success(f"**Predicted Animal:** {result['predicted_label']}")
            st.metric(label="Confidence", value=f"{result['confidence_percent']:.1f}%")

            st.write("##### Probability Breakdown:")
            for animal, score in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
                st.write(f"**{animal}**: {score:.1f}%")
                st.progress(score / 100.0)

    except ValueError as ve:
        st.error(f"⚠️ Invalid image: {ve}")
    except Exception as e:
        st.error(f"⚠️ An unexpected error occurred while processing the image: {e}")

st.markdown("---")

# Expandable section for model evaluation diagnostics
with st.expander("📊 View Training History & Confusion Matrix"):
    st.write("##### Model Performance on Validation & Test Sets")
    eval_col1, eval_col2 = st.columns(2)
    history_img_path = os.path.join(BASE_DIR, "training_history.png")
    cm_img_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    with eval_col1:
        if os.path.exists(history_img_path):
            st.image(history_img_path, caption="Training & Validation Curves", use_container_width=True)
    with eval_col2:
        if os.path.exists(cm_img_path):
            st.image(cm_img_path, caption="Confusion Matrix on Test Set", use_container_width=True)

st.caption("Built with Python, TensorFlow/Keras, and Streamlit.")
