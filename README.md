# Multi-Class Animal Image Classification

A complete, self-contained Python machine learning project designed for academic submission and web deployment that classifies animal images into **5 categories**:
- 🐱 **Cat**
- 🐶 **Dog**
- 🐘 **Elephant**
- 🐴 **Horse**
- 🦋 **Butterfly**

The project includes a custom Convolutional Neural Network (CNN) implemented from scratch using TensorFlow/Keras (strictly without pre-trained transfer learning architectures), image preprocessing, in-model data augmentation, early stopping regularization, evaluation metrics (accuracy, loss, confusion matrix, precision, recall, F1-score), a CLI prediction script, and an interactive Streamlit web application.

---

## 🛠️ Technology Stack

- **Language:** Python 3.10+
- **Deep Learning Framework:** TensorFlow & Keras
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
├── train.py                    # CNN model definition, training, and evaluation
├── predict.py                  # Prediction inference function and CLI tool
├── app.py                      # Interactive Streamlit web interface
├── download_data.py            # Automated dataset downloader & deduplicator
├── requirements.txt            # Python dependencies
├── class_names.json            # Generated class mapping metadata
├── animal_classifier.keras     # Saved trained CNN model (~14.8 MB)
├── training_history.png        # Training vs Validation accuracy & loss curves
├── confusion_matrix.png        # Confusion matrix heatmap on test set
├── .gitignore                  # Git ignore rules (excludes cache, venvs, etc.)
└── README.md                   # Complete project documentation & deployment guide
```

---

## 📊 Dataset & Separation

The dataset is partitioned into three strictly separated splits:
1. **Training Set (`dataset/train/`):** 350 images (70 per category) used to fit network weights.
2. **Validation Set (`dataset/val/`):** 75 images (15 per category) monitored during training for EarlyStopping.
3. **Testing Set (`dataset/test/`):** 75 images (15 per category) completely held out for unbiased evaluation, classification report, and confusion matrix.

### Automated Dataset Downloader & Sanitizer
The project includes [download_data.py](file:///c:/Users/HP/OneDrive/Desktop/Multi-Class%20Animal%20Image%20Classification/download_data.py) which:
- Retrieves royalty-free images from Wikimedia Commons for each animal category.
- Filters out extreme aspect ratio outliers and small icons.
- Applies perceptual hashing (`dhash`) to prevent duplicate and near-duplicate images.

To refresh or re-download the dataset:
```bash
python download_data.py
```

---

## ⚙️ Local Installation & Setup

1. Open your terminal in the project directory.
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## 🧠 Model Architecture & Training Strategy

The model is built from scratch without using any pre-trained networks:
1. **Input Layer:** `(128, 128, 3)`
2. **Data Augmentation:** In-model `RandomFlip("horizontal")` and `RandomZoom(0.08)` to reduce overfitting.
3. **Rescaling Layer:** Normalizes pixel intensity values from `[0, 255]` to `[0, 1]`.
4. **Convolutional Block 1:** `Conv2D(32, 3x3, relu)` + `MaxPooling2D((2, 2))` $\rightarrow$ `(64, 64, 32)`
5. **Convolutional Block 2:** `Conv2D(64, 3x3, relu)` + `MaxPooling2D((2, 2))` $\rightarrow$ `(32, 32, 64)`
6. **Convolutional Block 3:** `Conv2D(128, 3x3, relu)` + `MaxPooling2D((2, 2))` $\rightarrow$ `(16, 16, 128)`
7. **Convolutional Block 4:** `Conv2D(128, 3x3, relu)` + `MaxPooling2D((2, 2))` $\rightarrow$ `(8, 8, 128)`
8. **Flatten Layer:** 8,192 features.
9. **Dense Layer:** `Dense(128, activation='relu')`
10. **Regularization:** `Dropout(0.5)`
11. **Output Layer:** `Dense(5, activation='softmax')`
- **Total Parameters:** 1,290,181 (4.92 MB)

---

## 📈 Final Training & Testing Results

The model was trained with `EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)`:

- **Best Epoch Restored:** Epoch 17
- **Training Accuracy:** **69.43%** (Loss: 0.8014)
- **Validation Accuracy:** **53.33%** (Loss: 1.1684)
- **Final Independent Test Accuracy:** **32.00%** (Loss: 1.9954)
  *(Noticeably above the 20.00% random guess baseline for 5 classes on a lightweight scratch CNN)*

### Classification Report (Independent Test Set: 75 samples)

```text
              precision    recall  f1-score   support

   Butterfly     0.5000    0.2667    0.3478        15
         Cat     0.6000    0.6000    0.6000        15
         Dog     0.1111    0.0667    0.0833        15
    Elephant     0.2222    0.2667    0.2424        15
       Horse     0.2400    0.4000    0.3000        15

    accuracy                         0.3200        75
   macro avg     0.3347    0.3200    0.3147        75
weighted avg     0.3347    0.3200    0.3147        75
```

### Confusion Matrix on Test Set

| Actual \ Predicted | Butterfly | Cat | Dog | Elephant | Horse |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Butterfly** | **4** | 1 | 2 | 3 | 5 |
| **Cat** | 0 | **9** | 1 | 1 | 4 |
| **Dog** | 3 | 2 | **1** | 5 | 4 |
| **Elephant** | 1 | 1 | 3 | **4** | 6 |
| **Horse** | 0 | 2 | 2 | 5 | **6** |

The generated plots [training_history.png](file:///c:/Users/HP/OneDrive/Desktop/Multi-Class%20Animal%20Image%20Classification/training_history.png) and [confusion_matrix.png](file:///c:/Users/HP/OneDrive/Desktop/Multi-Class%20Animal%20Image%20Classification/confusion_matrix.png) are saved directly in the project directory.

---

## 🚀 Usage Guide

### 1. Model Training
To train the CNN model:
```bash
python train.py
```
This automatically outputs training metrics, saves the model checkpoint to `animal_classifier.keras`, updates `class_names.json`, and saves `training_history.png` and `confusion_matrix.png`.

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
Confidence:          60.75%

All Category Confidences:
  - Cat         : 60.75%
  - Dog         : 22.39%
  - Horse       : 8.57%
  - Butterfly   : 4.15%
  - Elephant    : 4.14%
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

This application is ready for 1-click deployment on **Streamlit Community Cloud**:

### Prerequisites
1. Push this repository to your GitHub account (instructions below).
2. Create a free account on [share.streamlit.io](https://share.streamlit.io/) linked to your GitHub.

### Step-by-Step Deployment
1. Go to [share.streamlit.io](https://share.streamlit.io/) and click **"New app"**.
2. Select your repository: `<your-username>/Multi-Class-Animal-Image-Classification`.
3. Set the **Branch** to `main`.
4. Set the **Main file path** to `app.py`.
5. Click **"Deploy!"**.
6. Streamlit Cloud will install dependencies from `requirements.txt`, load `animal_classifier.keras`, and launch the web app at your public URL (e.g. `https://<your-app-name>.streamlit.app`).

---

## 🎓 Academic / College Submission Notes

- **Objective:** Demonstrate end-to-end machine learning engineering—from dataset ingestion and deduplication to custom neural network architecture design, regularization, objective evaluation, and cloud deployment.
- **Model Constraints & Realism:** Rather than using a pre-trained transfer learning network (such as ResNet50 or MobileNetV2 with millions of pre-trained ImageNet parameters), a foundational 4-block CNN was constructed from scratch. Achieving 32% accuracy across 5 classes on an uncurated natural image dataset (where random chance is 20%) realistically demonstrates the capabilities and limitations of training shallow architectures from scratch with limited data.
- **Future Work:**
  - Transfer learning / fine-tuning with MobileNetV2 or EfficientNet for high-accuracy production deployment.
  - Expanding training data to 1,000+ images per class via Kaggle's Animals-10 dataset.
