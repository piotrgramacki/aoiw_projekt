import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from src.data.ucmerced_dataset import TripletDataModule
from src.settings import UC_MERCED_DATA_DIRECTORY, PATTERN_NET_DATA_DIRECTORY
import wandb
from src.models import TripletRetriever
from pytorch_lightning.callbacks import ModelCheckpoint


def get_checkpoint_callback():
    checkpoint_callback = ModelCheckpoint(
        monitor='val_anmrr',
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        save_last=True,
        mode='min',
    )
    return checkpoint_callback

if __name__ == "__main__":
    models = ['resnet18', 'resnet50', 'resnet101']
    datasets = {
        "uc_merced": UC_MERCED_DATA_DIRECTORY,
        "pattern_net": PATTERN_NET_DATA_DIRECTORY,
    }

    output_sizes = [50, 100, 150]

    for dataset_name, dataset_path in datasets.items():
        for model in models:
            for output_size in output_sizes:
                triplet_retriever = TripletRetriever(model, 224, output_size)
                dm = TripletDataModule(dataset_path, 224, 0.8, 100)
                wandb_logger = WandbLogger(f'{model}_{output_size}', project=f'triplet_retrieval_{dataset_name}')
                wandb_logger.watch(triplet_retriever, 'all', log_freq=10)
                trainer = pl.Trainer(max_epochs=50, gpus=-1, logger=wandb_logger)
                trainer.fit(triplet_retriever, dm)
                wandb.finish()
