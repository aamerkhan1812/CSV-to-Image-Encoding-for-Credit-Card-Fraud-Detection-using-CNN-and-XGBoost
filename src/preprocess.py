import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from config import TRAIN_CSV, TEST_CSV, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_FEATURES, RANDOM_SEED


def load_and_prepare_data():
    print("Loading datasets...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Use only V1-V28 features
    feature_cols = [f'V{i}' for i in range(1, 29)]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['Class'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['Class'].values
    
    print(f"\nFeatures shape: {X_train.shape} (train), {X_test.shape} (test)")
    print(f"Labels shape: {y_train.shape} (train), {y_test.shape} (test)")
    print(f"Fraud ratio - Train: {100 * y_train.mean():.3f}%, Test: {100 * y_test.mean():.3f}%")
    
    return X_train, X_test, y_train, y_test


def normalize_features(X_train, X_test):
    print("\nNormalizing features with RobustScaler...")
    scaler = RobustScaler()
    
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    print(f"Train - Min: {X_train_norm.min():.3f}, Max: {X_train_norm.max():.3f}")
    print(f"Test - Min: {X_test_norm.min():.3f}, Max: {X_test_norm.max():.3f}")
    
    return X_train_norm, X_test_norm, scaler


def encode_row_to_image(row):
    # RGB encoding: R=x, G=x², B=x³
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
    
    for i in range(NUM_FEATURES):
        pixel_row = i // IMAGE_WIDTH
        pixel_col = i % IMAGE_WIDTH
        
        x = row[i]
        image[pixel_row, pixel_col, 0] = x
        image[pixel_row, pixel_col, 1] = x ** 2
        image[pixel_row, pixel_col, 2] = x ** 3
    
    return image


def encode_dataset_to_images(X_norm, y, batch_size=10000):
    n_samples = len(X_norm)
    print(f"\nEncoding {n_samples:,} samples to 16x16 images...")
    
    images = np.zeros((n_samples, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        
        for i in range(start_idx, end_idx):
            images[i] = encode_row_to_image(X_norm[i])
        
        if (end_idx) % 50000 == 0 or end_idx == n_samples:
            print(f"  Processed {end_idx:,}/{n_samples:,} samples...")
    
    memory_mb = images.nbytes / (1024 ** 2)
    print(f"\nImage tensor shape: {images.shape}")
    print(f"Memory usage: {memory_mb:.2f} MB")
    print(f"Data type: {images.dtype}")
    
    return images, y


def prepare_datasets():
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    X_train_norm, X_test_norm, scaler = normalize_features(X_train, X_test)
    
    train_images, train_labels = encode_dataset_to_images(X_train_norm, y_train)
    test_images, test_labels = encode_dataset_to_images(X_test_norm, y_test)
    
    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETE")
    print("="*60)
    print(f"Training set: {train_images.shape} images, {train_labels.shape} labels")
    print(f"Test set: {test_images.shape} images, {test_labels.shape} labels")
    print(f"Total memory: {(train_images.nbytes + test_images.nbytes) / (1024**2):.2f} MB")
    
    return train_images, train_labels, test_images, test_labels, scaler


if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels, scaler = prepare_datasets()
    
    print("\n" + "="*60)
    print("SAMPLE IMAGE VERIFICATION")
    print("="*60)
    print(f"First image shape: {train_images[0].shape}")
    print(f"First image R channel (first 28 pixels):")
    print(train_images[0, :2, :, 0])
    print(f"\nNon-zero pixels: {np.count_nonzero(train_images[0])}")
    print(f"Label: {train_labels[0]}")
