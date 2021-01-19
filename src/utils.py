import torch
import numpy as np
from data.ucmerced_dataset import TripletDataset

from models.bovw import BoVWRetriever

def calculate_embeddings_torch(model, dataloader):
    paths = []
    embeddings = []
    classes = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            anchors = sample_batched['a'].cuda()
            y = sample_batched['a_y']
            classes.append(y.cpu().numpy())
            anchor_paths = sample_batched['path']
            paths.extend(anchor_paths)
            a = model(anchors).cpu().numpy()
            embeddings.append(a)

        embeddings = np.concatenate(embeddings)
        classes = np.concatenate(classes)
        paths = np.array(paths)
    return paths, embeddings, classes

def get_paths_and_classes(dataset: TripletDataset):
    paths = []
    classes = []
    for example in dataset:
        y = example['a_y']
        classes.append(y)
        path = example['path']
        paths.append(path)
    classes = np.array(classes)
    paths = np.array(paths)
    return paths, classes