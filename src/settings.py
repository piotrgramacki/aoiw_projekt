import os

PROJECT_DIRECTORY = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIRECTORY = os.path.join(PROJECT_DIRECTORY, "data")
RAW_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, "raw")
PROCESSED_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, "processed")
TRAIN_DATA_DIRECTORY = os.path.join(PROCESSED_DATA_DIRECTORY, "train")
TEST_DATA_DIRECTORY = os.path.join(PROCESSED_DATA_DIRECTORY, "test")
UC_MERCED_DATA_DIRECTORY = os.path.join(RAW_DATA_DIRECTORY, "uc_merced")
PATTERN_NET_DATA_DIRECTORY = os.path.join(RAW_DATA_DIRECTORY, "pattern_net")
RANDOM_WALKS_DIRECTORY = "."