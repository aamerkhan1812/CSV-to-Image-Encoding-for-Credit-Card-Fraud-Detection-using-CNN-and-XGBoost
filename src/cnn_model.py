import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import IMAGE_HEIGHT, IMAGE_WIDTH


def build_cnn_feature_extractor():
    # Architecture: Conv(32) -> Conv(64) -> MaxPool -> Conv(128) -> GAP -> 128-D
    model = keras.Sequential([
        layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        
        layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        
        layers.GlobalAveragePooling2D()
    ], name='cnn_feature_extractor')
    
    return model


def build_training_model():
    # Add temporary classification head for training
    feature_extractor = build_cnn_feature_extractor()
    
    model = keras.Sequential([
        feature_extractor,
        layers.Dense(1, activation='sigmoid', name='classification_head')
    ], name='cnn_training_model')
    
    return model


if __name__ == "__main__":
    print("Feature Extractor (Final Model):")
    print("=" * 60)
    feature_model = build_cnn_feature_extractor()
    feature_model.summary()
    
    print("\n\nTraining Model (with classification head):")
    print("=" * 60)
    training_model = build_training_model()
    training_model.summary()
    
    import numpy as np
    dummy_input = np.random.randn(1, 16, 16, 3).astype(np.float32)
    features = feature_model.predict(dummy_input, verbose=0)
    print(f"\n✓ Feature vector shape: {features.shape} (should be (1, 128))")
