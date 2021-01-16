import torch
import pytorch_lightning as pl
import os
from src.evaluation import evaluate_anmrr, evaluate_anmrr_precalculated
from sklearn.metrics.pairwise import euclidean_distances
from typing import List, Any

class TripletRetriever(pl.LightningModule):
    def __init__(
        self, model_name: str, image_size: int, last_layer_size=100
    ):
        super().__init__()
        self.model = torch.hub.load("pytorch/vision", model_name, pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(in_features, last_layer_size, bias=True)
        self.set_training_model_layers(False, 8)
        self.criterion = torch.nn.TripletMarginLoss()
        self.image_size = image_size
        self.train_dataset = None
        self.val_dataset = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        anchors = batch["a"]
        positives = batch["p"]
        negatives = batch["n"]
        a = self.model(anchors)
        p = self.model(positives)
        n = self.model(negatives)
        loss = self.criterion(a, p, n)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        anchors = batch["a"]
        positives = batch["p"]
        negatives = batch["n"]
        a = self.model(anchors)
        p = self.model(positives)
        n = self.model(negatives)
        a_y = batch["a_y"]
        loss = self.criterion(a, p, n)
        self.log("val_loss", loss, prog_bar=True)
        return loss, a, a_y
    
    def validation_epoch_end(self, outputs: List[Any]) -> None:
        losses, a, a_y = zip(*outputs)
        val_anmrr = evaluate_anmrr_precalculated(a, a_y, euclidean_distances)
        self.log("val_anmrr", val_anmrr, on_epoch=True, logger=True)
        self.train(False)
        with torch.no_grad():
            train = self.train_dataloader()
            train_anmrr = evaluate_anmrr(self, train, euclidean_distances)
            self.log("train_anmrr", train_anmrr, on_epoch=True, logger=True)
        self.train(True)
        return super().validation_epoch_end(outputs)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), weight_decay=1e-5)
        return optim

    def set_training_model_layers(self, training: bool, up_to_index: int):
        i = 0
        for child in self.model.children():
            if i > up_to_index:
                break
            for param in child.parameters():
                param.requires_grad = training
            i += 1
