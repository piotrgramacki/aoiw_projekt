import os

PROJECT_DIRECTORY = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIRECTORY = os.path.join(PROJECT_DIRECTORY, "data")
RAW_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, "raw")
PROCESSED_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, "processed")
TRAIN_DATA_DIRECTORY = os.path.join(PROCESSED_DATA_DIRECTORY, "train")
TEST_DATA_DIRECTORY = os.path.join(PROCESSED_DATA_DIRECTORY, "test")
UC_MERCED_DATA_DIRECTORY = os.path.join(RAW_DATA_DIRECTORY, "uc_merced")
UC_MERCED_BLUR_DATA_DIRECTORY = os.path.join(RAW_DATA_DIRECTORY, "uc_merced_blur")
UC_MERCED_EQ_DATA_DIRECTORY = os.path.join(RAW_DATA_DIRECTORY, "uc_merced_eq")
UC_MERCED_EQ_BLUR_DATA_DIRECTORY = os.path.join(RAW_DATA_DIRECTORY, "uc_merced_eq_blur")
PATTERN_NET_DATA_DIRECTORY = os.path.join(RAW_DATA_DIRECTORY, "pattern_net")
ORTO_DATA_DIRECTORY = os.path.join(RAW_DATA_DIRECTORY, "orto")
RANDOM_WALKS_DIRECTORY = "."

RESULTS_DIRECTORY = os.path.join(PROJECT_DIRECTORY, "results")
EDA_DIRECTORY = os.path.join(RESULTS_DIRECTORY, "eda")
RANDOM_WALKS_DIRECTORY = os.path.join(RESULTS_DIRECTORY, "random-walks")
