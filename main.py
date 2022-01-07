import time

# import local files
from src.data.make_dataset import download_dataset
from src.features.build_features import preprocess_data


# file paths
RAW_DATA_PATH = 'data/raw'
PROCESSED_DATASET_PATH = 'data/processed'

# download and process the gtzan music genre dataset
# download_dataset(DATA_DIR, RAW_DATA_PATH) # takes ~11 minutes
# preprocess_data(RAW_DATA_PATH, PROCESSED_DATASET_PATH) # takes ~3:30 minutes
