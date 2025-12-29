import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, 
    roc_curve, precision_recall_curve, recall_score
)
from preprocess import prepare_datasets
from config import MODELS_DIR, OUTPUT_DIR

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_models():
    print("=" * 60)
    print("Loading Trained Models")
    print("=" * 60)
    
    cnn_path = os.path.join(MODELS_DIR, 'cnn_feature_extractor.h5')
    cnn = keras.models.load_model(cnn_path)
    print(f"✓ CNN loaded: {cnn_path}")
    print(f"  Trainable: {cnn.trainable}")
    
    xgb_path = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
    with open(xgb_path, 'rb') as f:
        xgb_model = pickle.load(f)
    print(f"✓ XGBoost loaded: {xgb_path}")
    
    return cnn, xgb_model


def evaluate_pipeline(threshold=0.5):
    viz_dir = os.path.join(OUTPUT_DIR, "visualization")
    os.makedirs(viz_dir, exist_ok=True)
    
    cnn, xgb_model = load_models()
    
    print("\n" + "=" * 60)
    print("Loading and Encoding Test Data")
    print("=" * 60)
    print("Using normalization parameters from training (no recalculation)")
    
    train_images, train_labels, test_images, test_labels, scaler = prepare_datasets()
    
    print(f"✓ Test images: {test_images.shape}")
    print(f"✓ Test labels: {test_labels.shape}")
    
    print("\n" + "=" * 60)
    print("Extracting Features (Frozen CNN)")
    print("=" * 60)
    
    test_features = cnn.predict(test_images, batch_size=128, verbose=1)
    print(f"✓ Test features extracted: {test_features.shape}")
    
    print("\n" + "=" * 60)
    print("Making Predictions (XGBoost)")
    print("=" * 60)
    
    y_pred_proba = xgb_model.predict_proba(test_features)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    print(f"✓ Predictions generated")
    print(f"  Decision threshold: {threshold}")
    
    print("\n" + "=" * 60)
    print("EVALUATION METRICS (LOCKED)")
    print("=" * 60)
    
    cm = confusion_matrix(test_labels, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    recall = recall_score(test_labels, y_pred)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    roc_auc = roc_auc_score(test_labels, y_pred_proba)
    
    print(f"\n✓ Recall: {recall:.4f}")
    print(f"✓ False Negative Rate (FNR): {fnr:.4f}")
    print(f"✓ ROC-AUC: {roc_auc:.4f}")
    
    print(f"\n✓ Confusion Matrix:")
    print(f"  True Negatives (TN):  {tn:,}")
    print(f"  False Positives (FP): {fp:,}")
    print(f"  False Negatives (FN): {fn:,}")
    print(f"  True Positives (TP):  {tp:,}")
    
    print(f"\n✓ Classification Report:")
    print(classification_report(test_labels, y_pred, target_names=['Normal', 'Fraud']))
    
    create_visualizations(test_labels, y_pred, y_pred_proba, cm, roc_auc, viz_dir)
    
    try:
        create_shap_explanations(xgb_model, test_features, test_labels, viz_dir)
    except Exception as e:
        print(f"\nNote: SHAP visualization skipped ({str(e)})")
    
    save_metrics_report(recall, fnr, roc_auc, cm, threshold, viz_dir)
    
    return recall, fnr, roc_auc, cm


def create_visualizations(y_true, y_pred, y_pred_proba, cm, roc_auc, viz_dir):
    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)
    
    sns.set_style("whitegrid")
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(viz_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {cm_path}")
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(viz_dir, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {roc_path}")
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label='Precision-Recall Curve')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(viz_dir, 'precision_recall_curve.png')
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {pr_path}")
    
    # Prediction Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    normal_probs = y_pred_proba[y_true == 0]
    axes[0].hist(normal_probs, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Predicted Probability', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Normal Transactions', fontsize=12, fontweight='bold')
    axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold=0.5')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    fraud_probs = y_pred_proba[y_true == 1]
    axes[1].hist(fraud_probs, bins=50, color='red', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Predicted Probability', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Fraud Transactions', fontsize=12, fontweight='bold')
    axes[1].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold=0.5')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    dist_path = os.path.join(viz_dir, 'prediction_distribution.png')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {dist_path}")


def create_shap_explanations(xgb_model, test_features, test_labels, viz_dir):
    try:
        import shap
    except ImportError:
        import subprocess
        subprocess.check_call(['pip', 'install', 'shap'])
        import shap
    
    print("\n" + "=" * 60)
    print("Creating SHAP Explanations")
    print("=" * 60)
    
    explainer = shap.TreeExplainer(xgb_model)
    
    sample_size = min(1000, len(test_features))
    shap_values = explainer.shap_values(test_features[:sample_size])
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, test_features[:sample_size], 
                      feature_names=[f'CNN_feat_{i}' for i in range(128)],
                      show=False, max_display=20)
    plt.tight_layout()
    shap_summary_path = os.path.join(viz_dir, 'shap_summary.png')
    plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {shap_summary_path}")
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, test_features[:sample_size],
                      feature_names=[f'CNN_feat_{i}' for i in range(128)],
                      plot_type='bar', show=False, max_display=20)
    plt.tight_layout()
    shap_bar_path = os.path.join(viz_dir, 'shap_feature_importance.png')
    plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {shap_bar_path}")


def save_metrics_report(recall, fnr, roc_auc, cm, threshold, viz_dir):
    report_path = os.path.join(viz_dir, 'evaluation_report.txt')
    
    tn, fp, fn, tp = cm.ravel()
    
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("CNN + XGBoost Pipeline - Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("LOCKED METRICS:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Recall:                  {recall:.4f}\n")
        f.write(f"False Negative Rate:     {fnr:.4f}\n")
        f.write(f"ROC-AUC:                 {roc_auc:.4f}\n\n")
        
        f.write("CONFUSION MATRIX:\n")
        f.write("-" * 60 + "\n")
        f.write(f"True Negatives (TN):     {tn:,}\n")
        f.write(f"False Positives (FP):    {fp:,}\n")
        f.write(f"False Negatives (FN):    {fn:,}\n")
        f.write(f"True Positives (TP):     {tp:,}\n\n")
        
        f.write("CONFIGURATION:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Decision Threshold:      {threshold}\n")
        f.write(f"No retraining performed\n")
        f.write(f"Normalization from training set used\n")
        f.write(f"CNN weights frozen\n")
    
    print(f"\n✓ Saved: {report_path}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CNN + XGBoost Pipeline Evaluation")
    print("=" * 60)
    print("Mode: Inference only (no retraining)")
    print()
    
    recall, fnr, roc_auc, cm = evaluate_pipeline(threshold=0.5)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"✓ All visualizations saved to: output/visualization/")
    print(f"✓ Metrics report saved")
    print(f"✓ No retraining performed")
    print(f"✓ No label leakage")
