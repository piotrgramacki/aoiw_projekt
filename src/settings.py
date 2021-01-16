import os

PROJECT_DIRECTORY = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIRECTORY = os.path.join(PROJECT_DIRECTORY, "data")
RAW_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, "raw")
PROCESSED_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, "processed")
TRAIN_DATA_DIRECTORY = os.path.join(PROCESSED_DATA_DIRECTORY, "train")
TEST_DATA_DIRECTORY = os.path.join(PROCESSED_DATA_DIRECTORY, "test")

RESULTS_DIRECTORY = os.path.join(PROJECT_DIRECTORY, "results")
RANDOM_WALKS_DIRECTORY = os.path.join(RESULTS_DIRECTORY, "random-walks")
