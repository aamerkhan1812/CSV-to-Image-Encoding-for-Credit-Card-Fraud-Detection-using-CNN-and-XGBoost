import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Data files
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

# Image encoding
IMAGE_HEIGHT = 16
IMAGE_WIDTH = 16
NUM_FEATURES = 28  # V1-V28 only
TOTAL_PIXELS = IMAGE_HEIGHT * IMAGE_WIDTH

# Model hyperparameters
CNN_EPOCHS = 10
CNN_BATCH_SIZE = 64
CNN_LEARNING_RATE = 0.001

RANDOM_SEED = 42
