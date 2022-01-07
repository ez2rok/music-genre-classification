# import local files
from data.make_dataset import download_dataset
from features.build_features import preprocess_data


# file paths
DATA_DIR = 'data'
RAW_DATA_PATH = DATA_DIR + '/raw'
PROCESSED_DATASET_PATH = DATA_DIR + '/processed'

# download and process the gtzan music genre dataset
#download_dataset(DATA_DIR, RAW_DATA_PATH)
preprocess_data(RAW_DATA_PATH, PROCESSED_DATASET_PATH)
