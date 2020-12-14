from torch.utils.data import dataloader
from src.measures import anmrr
from torch.utils.data import DataLoader
from torch.nn import Module
import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from typing import Callable, Tuple, List, Union


def evaluate_anmrr(
    model: Module,
    data: DataLoader,
    measure: Callable[[np.ndarray], np.ndarray],
    class_mean: bool = False,
) -> Union[float, Tuple[float, List[Tuple[int, float]]]]:
    x_features = []
    y = []
    for _, sample_batched in enumerate(data):
        anchors = sample_batched["a"].cuda()
        a = model(anchors).cpu().numpy()
        a_y = sample_batched["a_y"].cpu().numpy()
        x_features.append(a)
        y.append(a_y)
    x_features = np.concatenate(x_features, axis=0)
    y = np.concatenate(y, axis=0)[:, None]
    result = anmrr(x_features, y, measure, class_mean=class_mean)
    return result


def evaluate_loss(model: Module, data: DataLoader, criterion) -> float:
    loss_sum = 0.0
    for _, sample_batched in enumerate(data):
        anchors = sample_batched["a"].cuda()
        positives = sample_batched["p"].cuda()
        negatives = sample_batched["n"].cuda()
        a = model(anchors)
        p = model(positives)
        n = model(negatives)
        loss = criterion(a, p, n)
        loss_sum += loss.item()

    loss = loss_sum / len(data)
    return loss
