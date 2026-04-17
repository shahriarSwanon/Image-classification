# config.py

DATASET_DIR = "../Animal Dataset"
# We resize to a smaller size because flattening 224x224 creates too many features 
# (224*224*3 = 150,528 features), which makes Logistic Regression extremely slow.
IMG_SIZE = (64, 64) 
MODEL_SAVE_PATH = "logistic_regression_model.pkl"
MAX_ITER = 1000
RANDOM_STATE = 42
TEST_SIZE = 0.2
