# Multi-Class Animal Image Classification (MobileNetV2 Transfer Learning)

A complete, self-contained Python machine learning project designed for academic submission and web deployment that classifies animal images into **5 categories**:
- 🐱 **Cat**
- 🐶 **Dog**
- 🐘 **Elephant**
- 🐴 **Horse**
- 🦋 **Butterfly**

The project employs **MobileNetV2 Transfer Learning** using pre-trained ImageNet weights as a deep convolutional feature extractor, coupled with in-model data augmentation, normalization, early stopping, fine-tuning, objective scikit-learn evaluation metrics (accuracy, loss, confusion matrix, classification report), a CLI prediction tool, and an interactive Streamlit web application.

---

## 🛠️ Technology Stack

- **Language:** Python 3.10+
- **Deep Learning & CNN:** TensorFlow & Keras
- **Pre-trained Architecture:** MobileNetV2 (ImageNet weights)
- **Computer Vision & Array Operations:** Pillow (PIL), NumPy
- **Machine Learning Evaluation:** Scikit-learn
- **Data Visualization:** Matplotlib
- **Web Application & Deployment:** Streamlit (Streamlit Community Cloud)

---

## 📁 Project Structure

```text
Multi-Class Animal Image Classification/
│
├── dataset/                    # Real, deduplicated animal images (500 total)
│   ├── train/                  # 70 images/class (350 images)
│   ├── val/                    # 15 images/class (75 images)
│   └── test/                   # 15 images/class (75 images)
│
├── train.py                    # MobileNetV2 transfer learning training & evaluation
├── predict.py                  # Prediction inference function and CLI tool
├── app.py                      # Interactive Streamlit web interface
├── download_data.py            # Automated dataset downloader & deduplicator
├── requirements.txt            # Python dependencies
├── class_names.json            # Generated class mapping metadata
├── animal_classifier.keras     # Saved trained MobileNetV2 model (~10 MB)
├── training_history.png        # Training vs Validation accuracy & loss curves
├── confusion_matrix.png        # Confusion matrix heatmap on test set
├── .gitignore                  # Git ignore rules (excludes cache, venvs, etc.)
└── README.md                   # Complete project documentation & deployment guide
```

---

## 📊 Dataset & Separation

The dataset is partitioned into three strictly separated splits:
1. **Training Set (`dataset/train/`):** 350 images (70 per category) used to train the classification head and fine-tune top convolutional layers.
2. **Validation Set (`dataset/val/`):** 75 images (15 per category) monitored during training by `EarlyStopping`.
3. **Testing Set (`dataset/test/`):** 75 images (15 per category) held out strictly for final unbiased evaluation.

---

## ⚙️ Local Installation & Setup

1. Open your terminal in the project directory.
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## 🧠 Model Architecture & Transfer Learning Strategy

1. **Input Layer:** `(128, 128, 3)`
2. **Data Augmentation:** In-model `RandomFlip("horizontal")`, `RandomRotation(0.08)`, and `RandomZoom(0.08)`.
3. **Rescaling:** Scaled to $[-1, 1]$ via `Rescaling(scale=1./127.5, offset=-1.0)` as expected by MobileNetV2.
4. **Pretrained Base:** MobileNetV2 pre-trained on ImageNet (excluding classification head).
   - Base features run in frozen inference mode (`training=False`) to preserve BatchNorm statistics.
5. **Classification Head:**
   - `GlobalAveragePooling2D()`
   - `Dropout(0.3)`
   - `Dense(128, activation='relu')`
   - `Dropout(0.3)`
   - `Dense(5, activation='softmax')`
6. **Two-Stage Training:**
   - **Phase 1 (Feature Extraction):** MobileNetV2 base frozen, training classification head with `Adam(lr=0.001)`.
   - **Phase 2 (Fine-Tuning):** Top 25 layers of MobileNetV2 unfrozen with `Adam(lr=1e-4)` and EarlyStopping.
- **Total Parameters:** 2,422,597 (9.24 MB)

---

## 📈 Final Training & Testing Results

- **Best Epoch Restored:** Epoch 13
- **Training Accuracy:** **95.43%** (Loss: 0.1708)
- **Validation Accuracy:** **82.67%** (Loss: 0.5647)
- **Final Independent Test Accuracy:** **57.33%** (Loss: 1.4828) on 75 unseen samples

### Classification Report (Independent Test Set: 75 samples)

```text
              precision    recall  f1-score   support

   Butterfly     0.8333    0.3333    0.4762        15
         Cat     0.7333    0.7333    0.7333        15
         Dog     0.2105    0.2667    0.2353        15
    Elephant     0.7857    0.7333    0.7586        15
       Horse     0.5714    0.8000    0.6667        15

    accuracy                         0.5733        75
   macro avg     0.6269    0.5733    0.5740        75
weighted avg     0.6269    0.5733    0.5740        75
```

### Confusion Matrix on Test Set

| Actual \ Predicted | Butterfly | Cat | Dog | Elephant | Horse |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Butterfly** | **5** | 1 | 9 | 0 | 0 |
| **Cat** | 1 | **11** | 2 | 0 | 1 |
| **Dog** | 0 | 3 | **4** | 2 | 6 |
| **Elephant** | 0 | 0 | 2 | **11** | 2 |
| **Horse** | 0 | 0 | 2 | 1 | **12** |

The generated plots [training_history.png](file:///c:/Users/HP/OneDrive/Desktop/Multi-Class%20Animal%20Image%20Classification/training_history.png) and [confusion_matrix.png](file:///c:/Users/HP/OneDrive/Desktop/Multi-Class%20Animal%20Image%20Classification/confusion_matrix.png) are saved directly in the project directory.

---

## 🚀 Usage Guide

### 1. Model Training
To train the MobileNetV2 model:
```bash
python train.py
```
This runs feature extraction and fine-tuning, saves the model checkpoint to `animal_classifier.keras`, updates `class_names.json`, and saves `training_history.png` and `confusion_matrix.png`.

### 2. Predict from Command Line
Test any image with `predict.py`:
```bash
python predict.py "dataset/test/cat/cat_001.jpg"
```
**Example CLI Output:**
```text
=============================================
Animal Image Prediction Result
=============================================
Predicted Animal:    Cat
Confidence:          99.99%

All Category Confidences:
  - Cat         : 99.99%
  - Butterfly   : 0.01%
  - Dog         : 0.00%
  - Elephant    : 0.00%
  - Horse       : 0.00%
=============================================
```

### 3. Run the Streamlit Web Application Locally
Start the interactive Streamlit user interface:
```bash
python -m streamlit run app.py
```
*(or `streamlit run app.py`)*

Open the local URL (`http://localhost:8501`) in your browser to:
- Upload any animal image (`.jpg`, `.jpeg`, `.png`, `.webp`).
- Or select from the pre-loaded test images using the sidebar.
- Inspect real-time predictions, confidence scores, probability breakdown bars, and view model diagnostic plots.

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

1. Push this repository to GitHub:
   ```bash
   git add .
   git commit -m "Update to MobileNetV2 Transfer Learning"
   git push origin main
   ```
2. Log into [share.streamlit.io](https://share.streamlit.io/).
3. Click **"New app"**, select your repository, set the main file path to `app.py`, and click **"Deploy!"**.
