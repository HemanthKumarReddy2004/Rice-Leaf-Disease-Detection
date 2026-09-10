# 🌾 Rice Leaf Disease Classifier & Audit Pipeline

An end-to-end, reproducible deep learning computer vision system designed to audit, preprocess, train, evaluate, and deploy multiclass image classification models for diagnosing rice leaf diseases. 

This project provides an automated, data-centric framework that ensures strict dataset integrity, visualizes structural metadata, and trains TensorFlow/Keras convolutional neural networks to support agricultural experts and field personnel in early crop-disease intervention.

---

## 📌 Problem Statement & Objectives

Rice (*Oryza sativa*) is a staple food for over half of the global population. Diseases like Bacterial Leaf Blight, Brown Spot, and Leaf Smut drastically reduce crop yields and farm profitability. Manual identification requires domain expertise and is prone to human error, especially across large-scale farms.

**Core Objectives:**
* **Automated Quality Auditing:** Perform byte-level and structural audits on field image datasets before feeding them to deep learning models.
* **Metadata Analysis:** Analyze image dimensions, aspect ratios, file types, and class imbalance metrics dynamically.
* **Reproducible Pipeline:** Build a deterministic TensorFlow pipeline (`SEED = 42`) for end-to-end data loading, augmentation, and training.
* **Model Serialization:** Generate verifiable evaluation metrics (Precision, Recall, F1-Score, Confusion Matrix) and export production-ready model artifacts.

> **Note:** This system operates as a decision-support system intended to complement agricultural extension workers and field agronomists.

---

## 📂 Dataset Overview & Metadata Summary

The dataset is dynamically discovered from raw directories containing labeled image files. The pipeline inspects and categorizes each class automatically.

| Target Class | Sample Count | Class Distribution (%) | Imbalance Ratio |
| :--- | :---: | :---: | :---: |
| **Bacterial Leaf Blight** | 40 | 33.61% | 1.000 |
| **Brown Spot** | 40 | 33.61% | 1.000 |
| **Leaf Smut** | 39 | 32.77% | 0.975 |
| **Total** | **119** | **100.00%** | **0.975 (Highly Balanced)** |

### Image Dimension Analysis
* **Total Image Count:** 119 readable image files.
* **Mean Resolution:** ~2383 × 707 pixels.
* **Mean Aspect Ratio:** 3.32 (Height-dominant elongated crop photos).
* **Format Consistency:** 100% `.jpg` images.

---

## 🔍 Pre-Training Quality & Integrity Audit

To prevent silent failures and label noise, the dataset undergoes a multi-pass audit prior to feature extraction:

* **Corrupted Image Audit:** 0 corrupt or unreadable image files detected (verified via `PIL.Image.verify()`).
* **Filename Collision Check:** 0 duplicate filenames across class folders.
* **Exact Duplicate Audit (SHA-256):** 0 identical byte-level duplicate files found.
* **Low-Resolution Filtering:** 0 images under the minimum 64×64 pixel threshold.

---

## 🛠️ Pipeline Architecture & Methodology
### Detailed Pipeline Stages:

1. **Deterministic Environment Setup:** Enforces operational determinism across Python `random`, NumPy, and TensorFlow using `tf.config.experimental.enable_op_determinism()` with `SEED = 42`.
2. **Data Ingestion & Quality Audit:** Standardizes image formats, calculates exact cryptographic hashes (`SHA-256`), checks file integrity, and verifies minimum dimension constraints.
3. **Exploratory Data Analysis (EDA):** Uses Matplotlib and Seaborn to visualize image aspect-ratio distributions, resolution scatters, and class proportions.
4. **Data Preprocessing & Augmentation:**
   * Resizes target images to uniform input dimensions (e.g., 224×224).
   * Normalizes pixel values from $[0, 255]$ to $[0, 1]$.
   * Applies spatial augmentations (horizontal flips, slight rotations, zooming) to prevent overfitting on smaller sample sizes.
5. **Model Architecture:** Builds a Convolutional Neural Network (CNN) with alternating `Conv2D`, `MaxPooling2D`, `BatchNormalization`, `Dropout`, and a `Dense` softmax layer for 3-class probability estimation.
6. **Evaluation & Artifact Export:** Computes macro/weighted macro Precision, Recall, F1-Scores, and plots Confusion Matrices using `scikit-learn`. Serializes trained models into `.keras` or `.h5` formats inside `data/rice_leaf_artifacts/`.

---

## 💻 Tech Stack & Dependencies

* **Language:** Python 3.10+
* **Deep Learning Framework:** TensorFlow 2.x / Keras
* **Data Processing & Analytics:** NumPy, Pandas, scikit-learn
* **Image Processing:** Pillow (PIL)
* **Data Visualization:** Matplotlib, Seaborn

---

## 🚀 Getting Started & Execution

### 1. Clone Repository & Setup Environment
```bash
# Clone the repository
git clone [https://github.com/your-username/rice-leaf-disease-detection.git](https://github.com/your-username/rice-leaf-disease-detection.git)
cd rice-leaf-disease-detection

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install numpy pandas matplotlib seaborn pillow scikit-learn tensorflow
