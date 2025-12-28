import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
from preprocess import prepare_datasets
from cnn_model import build_training_model, build_cnn_feature_extractor
from config import CNN_BATCH_SIZE, CNN_EPOCHS, MODELS_DIR, RANDOM_SEED

# Force CPU-only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def train_cnn():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("STEP 1: Loading and Encoding Data")
    print("=" * 60)
    train_images, train_labels, test_images, test_labels, scaler = prepare_datasets()
    
    # Compute class weights for recall optimization
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight = {0: class_weights_array[0], 1: class_weights_array[1]}
    
    print(f"\nClass weights (for recall optimization):")
    print(f"  Normal (0): {class_weight[0]:.2f}")
    print(f"  Fraud (1): {class_weight[1]:.2f}")
    
    print("\n" + "=" * 60)
    print("STEP 2: Building CNN Model")
    print("=" * 60)
    model = build_training_model()
    model.summary()
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_recall',
            patience=3,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, 'cnn_training_best.h5'),
            monitor='val_recall',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("\n" + "=" * 60)
    print("STEP 3: Training CNN (CPU-only, optimized for recall)")
    print("=" * 60)
    print(f"Batch size: {CNN_BATCH_SIZE}")
    print(f"Max epochs: {CNN_EPOCHS}")
    print(f"Device: CPU")
    
    history = model.fit(
        train_images, train_labels,
        validation_data=(test_images, test_labels),
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "=" * 60)
    print("STEP 4: Evaluating Trained Model")
    print("=" * 60)
    results = model.evaluate(test_images, test_labels, verbose=1)
    
    print("\nTest Set Performance:")
    print(f"  Loss: {results[0]:.4f}")
    print(f"  Accuracy: {results[1]:.4f}")
    print(f"  Precision: {results[2]:.4f}")
    print(f"  Recall: {results[3]:.4f}")
    print(f"  AUC: {results[4]:.4f}")
    
    print("\n" + "=" * 60)
    print("STEP 5: Extracting Feature Extractor")
    print("=" * 60)
    
    feature_extractor = model.layers[0]
    
    # Freeze all weights
    feature_extractor.trainable = False
    for layer in feature_extractor.layers:
        layer.trainable = False
    
    print("✓ Classification head removed")
    print("✓ All CNN weights frozen")
    
    feature_extractor_path = os.path.join(MODELS_DIR, 'cnn_feature_extractor.h5')
    feature_extractor.save(feature_extractor_path)
    print(f"✓ Feature extractor saved: {feature_extractor_path}")
    
    # Verify
    print("\n" + "=" * 60)
    print("STEP 6: Verification")
    print("=" * 60)
    
    test_sample = test_images[:5]
    features = feature_extractor.predict(test_sample, verbose=0)
    
    print(f"Input shape: {test_sample.shape}")
    print(f"Output features shape: {features.shape}")
    print(f"Feature vector dimensionality: {features.shape[1]}")
    print(f"✓ Expected: 128-D feature vectors")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"✓ CNN trained for {len(history.history['loss'])} epochs")
    print(f"✓ Best validation recall: {max(history.history['val_recall']):.4f}")
    print(f"✓ Feature extractor: 16×16×3 → 128-D")
    print(f"✓ Model saved: {feature_extractor_path}")
    print(f"✓ All weights frozen")
    
    return feature_extractor, history


if __name__ == "__main__":
    print("Starting CNN Training Pipeline...")
    print("Device: CPU-only")
    print(f"TensorFlow version: {tf.__version__}")
    print()
    
    feature_extractor, history = train_cnn()
