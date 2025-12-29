import os
import pickle
import numpy as np
import xgboost as xgb
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from preprocess import prepare_datasets
from config import MODELS_DIR, RANDOM_SEED

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def extract_features_from_cnn():
    print("=" * 60)
    print("STEP 1: Loading Data and CNN Feature Extractor")
    print("=" * 60)
    
    train_images, train_labels, test_images, test_labels, scaler = prepare_datasets()
    
    cnn_path = os.path.join(MODELS_DIR, 'cnn_feature_extractor.h5')
    print(f"\nLoading frozen CNN from: {cnn_path}")
    cnn = keras.models.load_model(cnn_path)
    
    print(f"CNN trainable: {cnn.trainable}")
    print(f"Total params: {cnn.count_params():,}")
    
    print("\n" + "=" * 60)
    print("STEP 2: Extracting Features from Training Set")
    print("=" * 60)
    
    print(f"Processing {len(train_images):,} training images...")
    train_features = cnn.predict(train_images, batch_size=128, verbose=1)
    
    print("\n" + "=" * 60)
    print("STEP 3: Extracting Features from Test Set")
    print("=" * 60)
    
    print(f"Processing {len(test_images):,} test images...")
    test_features = cnn.predict(test_images, batch_size=128, verbose=1)
    
    print("\n" + "=" * 60)
    print("Feature Extraction Complete")
    print("=" * 60)
    print(f"Train features: {train_features.shape}")
    print(f"Test features: {test_features.shape}")
    print(f"Feature dimensionality: {train_features.shape[1]}")
    print(f"Memory usage: {(train_features.nbytes + test_features.nbytes) / (1024**2):.2f} MB")
    
    return train_features, test_features, train_labels, test_labels


def train_xgboost(train_features, test_features, train_labels, test_labels):
    print("\n" + "=" * 60)
    print("STEP 4: Preparing XGBoost Training")
    print("=" * 60)
    
    n_negative = np.sum(train_labels == 0)
    n_positive = np.sum(train_labels == 1)
    scale_pos_weight = n_negative / n_positive
    
    print(f"\nClass distribution:")
    print(f"  Negative (0): {n_negative:,}")
    print(f"  Positive (1): {n_positive:,}")
    print(f"  Imbalance ratio: {scale_pos_weight:.2f}:1")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f}")
    
    xgb_params = {
        'objective': 'binary:logistic',
        'max_depth': 5,
        'learning_rate': 0.1,
        'scale_pos_weight': scale_pos_weight,
        'eval_metric': ['logloss', 'auc'],
        'random_state': RANDOM_SEED,
        'n_estimators': 100,
        'early_stopping_rounds': 10,
        'verbosity': 1
    }
    
    print(f"\nXGBoost Configuration:")
    for key, value in xgb_params.items():
        if key != 'early_stopping_rounds':
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("STEP 5: Training XGBoost")
    print("=" * 60)
    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    
    xgb_model.fit(
        train_features, train_labels,
        eval_set=[(test_features, test_labels)],
        verbose=True
    )
    
    print(f"\n✓ Training completed")
    print(f"✓ Best iteration: {xgb_model.best_iteration}")
    print(f"✓ Best score: {xgb_model.best_score:.4f}")
    
    return xgb_model


def evaluate_xgboost(xgb_model, test_features, test_labels):
    print("\n" + "=" * 60)
    print("STEP 6: Evaluating XGBoost Model")
    print("=" * 60)
    
    y_pred = xgb_model.predict(test_features)
    y_pred_proba = xgb_model.predict_proba(test_features)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(test_labels, y_pred, target_names=['Normal', 'Fraud']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, y_pred)
    print(cm)
    print(f"\nTrue Negatives:  {cm[0, 0]:,}")
    print(f"False Positives: {cm[0, 1]:,}")
    print(f"False Negatives: {cm[1, 0]:,}")
    print(f"True Positives:  {cm[1, 1]:,}")
    
    auc_score = roc_auc_score(test_labels, y_pred_proba)
    print(f"\nROC-AUC Score: {auc_score:.4f}")
    
    return y_pred, y_pred_proba, auc_score


def save_xgboost_model(xgb_model):
    print("\n" + "=" * 60)
    print("STEP 7: Saving XGBoost Model")
    print("=" * 60)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    model_path = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(xgb_model, f)
    
    print(f"✓ Model saved: {model_path}")
    
    return model_path


def main():
    print("\n" + "=" * 60)
    print("XGBoost Training Pipeline")
    print("=" * 60)
    print("Using CNN-extracted features for fraud detection")
    print()
    
    train_features, test_features, train_labels, test_labels = extract_features_from_cnn()
    xgb_model = train_xgboost(train_features, test_features, train_labels, test_labels)
    y_pred, y_pred_proba, auc_score = evaluate_xgboost(xgb_model, test_features, test_labels)
    model_path = save_xgboost_model(xgb_model)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"✓ CNN features extracted: 128-D vectors")
    print(f"✓ XGBoost trained on {len(train_features):,} samples")
    print(f"✓ Test ROC-AUC: {auc_score:.4f}")
    print(f"✓ Model saved: {model_path}")
    print(f"✓ CNN weights remained frozen throughout")
    
    return xgb_model, auc_score


if __name__ == "__main__":
    xgb_model, auc_score = main()
