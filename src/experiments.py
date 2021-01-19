from gc import callbacks
import os
from typing import List
from venv import create
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from src.models.bovw import BoVWRetriever
from src.data.ucmerced_dataset import TripletDataModule, TripletDataset
from src.settings import RESULTS_DIRECTORY, UC_MERCED_DATA_DIRECTORY, PATTERN_NET_DATA_DIRECTORY
import wandb
from src.models.triplet_retriever import TripletRetriever
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from src.utils import get_paths_and_classes

from src.visualisation import visualize_anmrr_per_class, visualize_best_and_worst_queries, visualize_embeddings

import pandas as pd

def create_path_if_not_exists(path: str):
    if not os.path.exists(path):
        os.mkdir(path)

def run_bovw_experiments(train_data: TripletDataset, test_data: TripletDataset, cluster_sizes: List[int], samples_counts: List[int], dataset_name: str, output_path: str):
    model = BoVWRetriever(100, 10000)
    resampled_train_descriptors, resampled_train_labels = model.get_resampled_descriptors(train_data)
    test_descriptors = model.get_descriptors(test_data, labels=False)
    y_test = model.get_class_labels(test_data)

    values = []
    values_per_class = []
    cluster_numbers = []
    sample_numbers = []

    for clusters in cluster_sizes:
        for samples in samples_counts:
            experiment_path = os.path.join(output_path, f"{clusters}_{samples}")
            create_path_if_not_exists(experiment_path)
            cluster_numbers.append(clusters)
            sample_numbers.append(samples)
            print(f"Clusters: {clusters}, samples: {samples}")
            model.clusters = clusters
            model.samples_count = samples
            model.fit_precomputed(resampled_train_descriptors, resampled_train_labels)
            value, value_per_class, nmrr, embeddings = model.eval_precomputed(test_descriptors, y_test)
            paths, _ = get_paths_and_classes(test_data)
            visualize_best_and_worst_queries(paths, embeddings, nmrr, 3, experiment_path, 16)
            
            values.append(value)
            values_per_class.append(value_per_class)
            result_path = os.path.join(experiment_path, f"anmrr_{clusters}_{samples}.png")
            visualize_anmrr_per_class(value_per_class, train_data.label_name_mapping, dataset_name, result_path, f"BoVW, c={clusters}, s={samples}")
            visualize_embeddings(embeddings, paths, experiment_path)
    
    df = pd.DataFrame.from_dict({"clusters": cluster_numbers, "samples": sample_numbers, "anmrr": values})
    class_names = test_data.class_names
    values_per_class_without_labels = [list(list(zip(*single_experiment))[1]) for single_experiment in values_per_class]
    full_df = pd.concat([df, pd.DataFrame(np.array(values_per_class_without_labels), columns=class_names)], axis=1)
    results_long_form = full_df.melt(id_vars=['clusters', 'samples', 'anmrr'], var_name='class', value_name='anmrr_per_class')
    results_long_form['experiment_name'] = results_long_form.apply(lambda row: str(row['clusters']) + "_" + str(row['samples']), axis=1)
    results_long_form.to_pickle(os.path.join(output_path, f"results_{dataset_name}.pkl.gz"))
    
    return values, values_per_class, cluster_numbers, sample_numbers


def run_all_bovw():
    image_size = 256
    dm = TripletDataModule(UC_MERCED_DATA_DIRECTORY, image_size, 0.8, 100, augment=False, normalize=False, permute=True)
    dm.setup(None)
    train_dataset = dm.train_dataset
    test_dataset = dm.val_dataset

    # output_sizes = [25, 50, 100, 150]
    output_sizes = [25, 50]
    samples = [10000]
    bovw_path = os.path.join(RESULTS_DIRECTORY, "bovw")
    create_path_if_not_exists(bovw_path)
    dataset_path = os.path.join(bovw_path, "uc_merced")
    create_path_if_not_exists(dataset_path)

    run_bovw_experiments(train_dataset, test_dataset, output_sizes, samples, "UC Merced", dataset_path)

def get_checkpoint_callback():
    checkpoint_callback = ModelCheckpoint(
        monitor='val_anmrr',
        filename='{epoch:02d}-{val_anmrr:.2f}',
        save_top_k=1,
        save_last=True,
        mode='min',
    )
    return checkpoint_callback

if __name__ == "__main__":
    models = ['resnet18', 'resnet50', 'resnet101']
    datasets = {
        # "uc_merced": UC_MERCED_DATA_DIRECTORY,
        "pattern_net": PATTERN_NET_DATA_DIRECTORY,
    }

    output_sizes = [50, 100, 150]

    epochs = 50

    for dataset_name, dataset_path in datasets.items():
        for model in models:
            for output_size in output_sizes:
                checkpoint_callback = get_checkpoint_callback()
                triplet_retriever = TripletRetriever(model, output_size)
                dm = TripletDataModule(dataset_path, 224, 0.8, 100)
                wandb_logger = WandbLogger(f'{model}_{output_size}', project=f'triplet_retrieval_{dataset_name}')
                wandb_logger.watch(triplet_retriever, 'all', log_freq=10)
                trainer = pl.Trainer(max_epochs=epochs, gpus=-1, logger=wandb_logger, callbacks=[checkpoint_callback])
                trainer.fit(triplet_retriever, dm)
                wandb.finish()
