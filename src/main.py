from typing import List
from sklearn import datasets
from data.ucmerced_dataset import TripletDataModule, TripletDataset
from experiments import run_all_bovw, run_all_triplet, create_path_if_not_exists
from models.bovw import BoVWRetriever
from src.eda.eda import get_color_intensity_counts_per_class, generate_color_histograms
from src.settings import RESULTS_DIRECTORY, EDA_DIRECTORY, UC_MERCED_DATA_DIRECTORY, PATTERN_NET_DATA_DIRECTORY, UC_MERCED_EQ_DATA_DIRECTORY
import os
from visualisation import visualize_anmrr_per_class
import matplotlib
from src.preprocessing import transform_images, equalize_histogram

def preprocess_datasets():
    transform_images(UC_MERCED_DATA_DIRECTORY, UC_MERCED_EQ_DATA_DIRECTORY, [equalize_histogram])
    

def create_histograms():
    DATASETS = [
        ("uc_merced", UC_MERCED_DATA_DIRECTORY),
        ("uc_merced_eq", UC_MERCED_EQ_DATA_DIRECTORY),
        ("pattern_net", PATTERN_NET_DATA_DIRECTORY),
    ]

    create_path_if_not_exists(RESULTS_DIRECTORY)
    create_path_if_not_exists(EDA_DIRECTORY)
    
    for dataset_name, dataset_path in DATASETS:
        result_path = os.path.join(EDA_DIRECTORY, dataset_name)
        print("Calculating intensities")
        create_path_if_not_exists(result_path)
        color_intensities = get_color_intensity_counts_per_class(dataset_path)
        print("Generating histograms")
        generate_color_histograms(color_intensities, result_path)

if __name__ == '__main__':
    matplotlib.use('Agg')
    preprocess_datasets()
    create_histograms()
    run_all_bovw()
    run_all_triplet()