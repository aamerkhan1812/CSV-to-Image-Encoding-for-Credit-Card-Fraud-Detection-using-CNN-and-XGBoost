# CSV-to-Image Encoding for Credit Card Fraud Detection using CNN and XGBoost

## Overview

This project presents a novel approach to credit card fraud detection by converting tabular transaction data into RGB images and using a hybrid CNN + XGBoost pipeline for classification.

### Core Concept

Instead of treating credit card transactions as traditional tabular data, we encode each transaction as a 16×16 RGB image, allowing Convolutional Neural Networks (CNNs) to learn spatial patterns and feature relationships that might be missed by conventional machine learning models.

### Pipeline Architecture

```
Raw Transaction Data (28 features)
         ↓
   Normalization (RobustScaler)
         ↓
   Image Encoding (16×16 RGB)
         ↓
   CNN Feature Extraction (128-D)
         ↓
   XGBoost Classification
         ↓
   Fraud Prediction
```

## Methodology

### 1. Data Preprocessing

**Normalization Strategy:**
- **Method**: RobustScaler
- **Rationale**: Handles outliers effectively and works with both positive and negative values
- **Critical Rule**: Fitted only on training data to prevent data leakage

**Feature Selection:**
- Uses V1-V28 features (PCA-transformed by original dataset creators)
- Excludes 'Time' and 'Amount' columns
- Total: 28 features per transaction

### 2. Image Encoding Formula

Each transaction is converted to a 16×16 RGB image using a deterministic mapping:

**RGB Encoding Formula:**
```
For each feature value x:
  R (Red channel)   = x
  G (Green channel) = x²
  B (Blue channel)  = x³
```

**Spatial Mapping:**
- 28 features → first 28 pixels (row-major order)
- Remaining 228 pixels → zero-padded (black)
- Grid size: 16×16 = 256 total pixels
- Memory footprint: ~52 MB for entire dataset (float32)

**Why This Encoding Works:**

1. **Non-linear transformations**: x, x², x³ capture different scales and magnitudes
2. **Spatial locality**: Related features can be positioned near each other
3. **Color channels**: Three representations of each feature provide richer information
4. **CNN compatibility**: Enables use of powerful convolutional architectures

### 3. CNN Architecture

**Purpose**: Extract high-level features from encoded images

```
Input Layer: 16×16×3 RGB image

Block 1:
  Conv2D(32 filters, 3×3 kernel, padding='same')
  ReLU Activation
  Batch Normalization

Block 2:
  Conv2D(64 filters, 3×3 kernel, padding='same')
  ReLU Activation
  Batch Normalization
  MaxPooling2D(2×2)

Block 3:
  Conv2D(128 filters, 3×3 kernel, padding='same')
  ReLU Activation
  Batch Normalization

Output:
  Global Average Pooling
  → 128-dimensional feature vector
```

**Training Configuration:**
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Binary Cross-Entropy
- **Batch Size**: 64
- **Epochs**: 10 (with early stopping, patience=3)
- **Class Weighting**: Applied to handle extreme imbalance (0.172% fraud ratio)
- **Optimization Goal**: Maximize recall (fraud detection)
- **Device**: CPU-only execution

**Post-Training:**
- Classification head removed
- All weights frozen
- Saved as feature extractor only

### 4. XGBoost Classification

**Purpose**: Final fraud classification using CNN-extracted features

**Configuration:**
```python
{
    'objective': 'binary:logistic',
    'max_depth': 5,
    'learning_rate': 0.1,
    'scale_pos_weight': 577.88,  # Handles 0.172% fraud ratio
    'n_estimators': 100,
    'early_stopping_rounds': 10,
    'eval_metric': ['logloss', 'auc']
}
```

**Key Parameter:**
- `scale_pos_weight = negative_samples / positive_samples`
- Addresses extreme class imbalance
- Ensures model doesn't ignore minority (fraud) class

## Approach Highlights

### Design Decisions

1. **Stratified Splitting**: 80/20 train-test split preserving fraud ratio
2. **No Data Leakage**: Normalization parameters computed only on training data
3. **Frozen CNN**: Feature extractor weights locked during XGBoost training
4. **Memory Efficiency**: In-memory tensor generation, no disk I/O for images
5. **Reproducibility**: Random seed (42) set globally

### Why Hybrid CNN + XGBoost?

**CNN Strengths:**
- Learns spatial patterns in encoded images
- Captures non-linear feature interactions
- Reduces dimensionality (28 features → 128-D)

**XGBoost Strengths:**
- Excellent with tabular data
- Handles class imbalance well
- Fast inference
- Interpretable (SHAP values)

**Combined Benefits:**
- CNN learns rich representations
- XGBoost makes final decision with interpretability
- Better than either approach alone

### Innovation

This project demonstrates that:
1. Tabular data can be effectively represented as images
2. CNNs can extract meaningful features from encoded transactions
3. Hybrid approaches can outperform traditional methods
4. The encoding formula (R=x, G=x², B=x³) preserves information while enabling spatial learning

## Dataset

- **Source**: Kaggle Credit Card Fraud Detection Dataset
- **Size**: 284,807 transactions
- **Fraud Ratio**: 0.172% (492 fraud cases)
- **Features**: V1-V28 (PCA-transformed by dataset creators)
- **Split**: 227,845 training / 56,962 testing (stratified)

## Technical Stack

- **Python**: 3.11+ (tested on 3.13.9)
- **Deep Learning**: TensorFlow 2.13+
- **Gradient Boosting**: XGBoost 2.0+
- **Preprocessing**: scikit-learn 1.3+
- **Data Handling**: pandas 2.0+, numpy 1.24+
- **Visualization**: matplotlib 3.7+, seaborn 0.13+
- **Explainability**: SHAP 0.44+

---


