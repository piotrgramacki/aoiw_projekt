from gc import callbacks
import os
from typing import List
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from models.bovw import BoVWRetriever
from src.data.ucmerced_dataset import TripletDataModule, TripletDataset
from src.settings import UC_MERCED_DATA_DIRECTORY, PATTERN_NET_DATA_DIRECTORY
import wandb
from src.models.triplet_retriever import TripletRetriever
from pytorch_lightning.callbacks import ModelCheckpoint

from visualisation import visualize_anmrr_per_class

import pandas as pd

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
            cluster_numbers.append(clusters)
            sample_numbers.append(samples)
            print(f"Clusters: {clusters}, samples: {samples}")
            model.clusters = clusters
            model.samples_count = samples
            model.fit_precomputed(resampled_train_descriptors, resampled_train_labels)
            value, value_per_class = model.eval_precomputed(test_descriptors, y_test)
            values.append(value)
            values_per_class.append(values_per_class)
            result_path = os.path.join(output_path, f"bovw_anmrr_{dataset_name}_{clusters}_{samples}.png")
            visualize_anmrr_per_class(value_per_class, train_data.label_name_mapping, dataset_name, result_path)
    
    # df = pd.DataFrame({"clusters": cluster_numbers, "samples": sample_numbers, "anmrr": values, "anmrr_per_class": values_per_class})
    # df.to_pickle(os.path.join(output_path, f"bovw_{dataset_name}.pkl.gz"))


def run_all_bovw():
    image_size = 256
    dm = TripletDataModule(UC_MERCED_DATA_DIRECTORY, image_size, 0.8, 100, augment=False, normalize=False, permute=True)
    dm.setup(None)
    train_dataset = dm.train_dataset
    test_dataset = dm.val_dataset

    output_sizes = [25, 50, 100, 150]
    samples = [10000]
    run_bovw_experiments(train_dataset, test_dataset, output_sizes, samples, "UC Merced", "results")

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
