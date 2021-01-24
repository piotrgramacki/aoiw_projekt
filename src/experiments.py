from gc import callbacks
import os
from typing import List
from venv import create
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics.pairwise import euclidean_distances
from measures import anmrr
from src.models.bovw import BoVWRetriever
from src.data.ucmerced_dataset import TripletDataModule, TripletDataset
from src.settings import RESULTS_DIRECTORY, UC_MERCED_BLUR_DATA_DIRECTORY, UC_MERCED_DATA_DIRECTORY, PATTERN_NET_DATA_DIRECTORY, UC_MERCED_EQ_BLUR_DATA_DIRECTORY, UC_MERCED_EQ_DATA_DIRECTORY
import wandb
from src.models.triplet_retriever import TripletRetriever
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from src.utils import get_paths_and_classes, calculate_embeddings_torch

from src.visualisation import visualize_anmrr_per_class, visualize_best_and_worst_queries, visualize_embeddings
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import pandas as pd
from src.utils import create_path_if_not_exists
import pickle as pkl


def run_bovw_experiments(train_data: TripletDataset, test_data: TripletDataset, cluster_sizes: List[int], samples_counts: List[int], dataset_name: str, output_path: str):
    model = BoVWRetriever(100, 10000)
    resampled_train_descriptors, resampled_train_labels, train_descriptors, train_labels = model.get_resampled_descriptors(train_data, return_not_resampled=True)
    test_descriptors = model.get_descriptors(test_data, labels=False)
    y_test = model.get_class_labels(test_data)

    values = []
    values_per_class = []
    cluster_numbers = []
    sample_numbers = []
    undersampling = []

    def run_experiment(clusters, samples, undersample):
        experiment_path = os.path.join(output_path, f"{clusters}_{samples}_{undersample}")  
        create_path_if_not_exists(experiment_path)
        cluster_numbers.append(clusters)
        sample_numbers.append(samples)
        undersampling.append(undersample)
        print(f"Clusters: {clusters}, samples: {samples}, undersampling: {undersample}")
        model.clusters = clusters
        model.samples_count = samples
        if undersample:
            model.fit_precomputed(resampled_train_descriptors, resampled_train_labels)
        else:
            model.fit_precomputed(train_descriptors, train_labels)

        value, value_per_class, nmrr, embeddings = model.eval_precomputed(test_descriptors, y_test)
        paths, _ = get_paths_and_classes(test_data)
        visualize_best_and_worst_queries(paths, embeddings, nmrr, 3, experiment_path, 16)
        
        values.append(value)
        print(value)
        values_per_class.append(value_per_class)
        result_path = os.path.join(experiment_path, f"anmrr_{clusters}_{samples}.png")
        visualize_anmrr_per_class(value_per_class, train_data.label_name_mapping, dataset_name, result_path, f"BoVW, c={clusters}, s={samples}, u={undersample}")
        visualize_embeddings(embeddings, paths, experiment_path)
        model_path = os.path.join(experiment_path, f"bovw.pkl.gz")
        with open(model_path, "wb") as f:
            pkl.dump(model, f)

    for clusters in cluster_sizes:
        for samples in samples_counts:
            for undersample in [True, False]:
                run_experiment(clusters, samples, undersample)
    
    df = pd.DataFrame.from_dict({"clusters": cluster_numbers, "samples": sample_numbers, "anmrr": values, "undersampling": undersampling})
    class_names = test_data.class_names
    values_per_class_without_labels = [list(list(zip(*single_experiment))[1]) for single_experiment in values_per_class]
    full_df = pd.concat([df, pd.DataFrame(np.array(values_per_class_without_labels), columns=class_names)], axis=1)
    full_df.to_pickle(os.path.join(output_path, f"results_{dataset_name}.pkl.gz"))
    
    return values, values_per_class, cluster_numbers, sample_numbers


def run_all_bovw():
    datasets = [
        ("UC Merced", UC_MERCED_DATA_DIRECTORY, "uc_merced"), 
        ("UC Merced Equalized", UC_MERCED_EQ_DATA_DIRECTORY, "uc_merced_eq"),
        ("PatternNet", PATTERN_NET_DATA_DIRECTORY, "pattern_net")
    ]
    image_size = 256

    output_sizes = [25, 50, 100, 150]
    samples = [5000, 10000, 20000, 50000, 100_000]
    bovw_path = os.path.join(RESULTS_DIRECTORY, "bovw")
    create_path_if_not_exists(bovw_path)

    for dataset_name, dataset_path, output_name in datasets:
        dm = TripletDataModule(dataset_path, image_size, 0.8, 100, augment=False, normalize=False, permute=True)
        dm.setup(None)
        train_dataset = dm.train_dataset
        test_dataset = dm.val_dataset

        output_path = os.path.join(bovw_path, output_name)
        create_path_if_not_exists(output_path)

        run_bovw_experiments(train_dataset, test_dataset, output_sizes, samples, dataset_name, output_path)

def get_checkpoint_callback():
    checkpoint_callback = ModelCheckpoint(
        monitor='val_anmrr',
        filename='{epoch:02d}-{val_anmrr:.2f}',
        save_top_k=1,
        save_last=True,
        mode='min',
    )
    return checkpoint_callback

def get_early_stopping_callback():
    early_stop_callback = EarlyStopping(
        monitor='val_anmrr',
        patience=5,
        mode='min'
    )
    return early_stop_callback

def run_all_triplet():
    models = ['resnet18', 'resnet50', 'resnet101']
    datasets = [
        ("uc_merced", "UC Merced", UC_MERCED_DATA_DIRECTORY),
        ("uc_merced_eq", "UC Merced Equalized", UC_MERCED_EQ_DATA_DIRECTORY),
        ("pattern_net", "PatternNet", PATTERN_NET_DATA_DIRECTORY),
    ]

    output_sizes = [25, 50, 100]

    epochs = 50
    
    for dataset_path_name, dataset_name, dataset_path in datasets:
        for augment in [True, False]:
            results_path = os.path.join(RESULTS_DIRECTORY, "triplet")
            if augment:
                results_path += "_augment"
            create_path_if_not_exists(results_path)

            output_path = os.path.join(results_path, dataset_path_name)
            create_path_if_not_exists(output_path)
            used_models = []
            used_output_sizes = []
            values = []
            values_per_class = []
            class_names = []
            for model in models:
                for output_size in output_sizes:
                    experiment_path = os.path.join(output_path, f"{model}_{output_size}")
                    create_path_if_not_exists(experiment_path)
                    used_models.append(model)
                    used_output_sizes.append(output_size)
                    checkpoint_callback = get_checkpoint_callback()
                    early_stop_callback = get_early_stopping_callback()
                    triplet_retriever = TripletRetriever(model, output_size)
                    if augment:
                        dm = TripletDataModule(dataset_path, 224, 0.8, 100, jitter_brightness=0.5, jitter_contrast=0.4, jitter_saturation=0.5, rotate=True)
                    else:
                        dm = TripletDataModule(dataset_path, 224, 0.8, 100)

                    experiment_name = f'{model}_{output_size}'
                    if augment:
                        experiment_name += "_a"

                    wandb_logger = WandbLogger(experiment_name, project=f'triplet_retrieval_{dataset_path_name}')
                    wandb_logger.watch(triplet_retriever, 'all', log_freq=10)
                    trainer = pl.Trainer(max_epochs=epochs, gpus=-1, logger=wandb_logger, callbacks=[checkpoint_callback, early_stop_callback])
                    trainer.fit(triplet_retriever, dm)

                    paths, embeddings, classes = calculate_embeddings_torch(triplet_retriever, dm.val_dataloader())
                    anmrr_value, anmrr_per_class, nmrr = anmrr(embeddings, classes, euclidean_distances, class_mean=True, all_queries=True)
                    values.append(anmrr_value)
                    values_per_class.append(anmrr_per_class)
                    class_names = dm.class_names
                    print("Creating visualisations")
                    visualize_best_and_worst_queries(paths, embeddings, nmrr, 3, experiment_path, 16)
                    
                    anmrr_per_class_path = os.path.join(experiment_path, f"anmrr_{model}_{output_size}.png")
                    visualize_anmrr_per_class(anmrr_per_class, dm.label_name_mapping, dataset_name, anmrr_per_class_path, f"Triplet, m={model}, o={output_size}")
                    visualize_embeddings(embeddings, paths, experiment_path)
                    
                    wandb.finish()

            df = pd.DataFrame.from_dict({"model": used_models, "output_size": used_output_sizes, "anmrr": values})
            values_per_class_without_labels = [list(list(zip(*single_experiment))[1]) for single_experiment in values_per_class]
            full_df = pd.concat([df, pd.DataFrame(np.array(values_per_class_without_labels), columns=class_names)], axis=1)
            full_df.to_pickle(os.path.join(output_path, f"results_{dataset_path_name}.pkl.gz"))
