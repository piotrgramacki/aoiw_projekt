from typing import Union

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_ubyte
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from tqdm import trange, tqdm

from imblearn.under_sampling import RandomUnderSampler

from src.data.ucmerced_dataset import TripletDataset
from src.measures import anmrr

class BoVWRetriever:
    def __init__(self, clusters: int, samples_count: int, random_state=42):
        self.clusters = clusters
        self.samples_count = samples_count
        self.random_state = random_state
        self.sift = cv2.SIFT_create()
        self.k_means = None
    
    def fit(self, data: TripletDataset):
        x_train = np.empty(shape=(len(data), 256, 256, 3))
        y_train = np.empty(shape=(len(data),), dtype=np.int)
        for idx in trange(len(data)):
            item = data[idx]

            x_train[idx] = item["a"]
            y_train[idx] = item["a_y"]

        x_train_descriptors, y_train_descriptors = self.get_descriptors(x_train, y_train)
        # x_train_descriptors, y_train_descriptors = self.get_descriptors_for_dataset(data, labels=True)
        stacked_train_descriptors = np.vstack(x_train_descriptors)
        stacked_train_labels = np.hstack(y_train_descriptors)
        under_sampler = RandomUnderSampler(random_state=self.random_state)
        resampled_train_descriptors, resampled_train_labels = under_sampler.fit_resample(
            stacked_train_descriptors, stacked_train_labels
        )

        samples_ratio_for_kmeans = self.samples_count / resampled_train_descriptors.shape[0]

        _, descriptors_for_kmeans, _, labels_for_kmeans = train_test_split(
            resampled_train_descriptors,
            resampled_train_labels,
            test_size=samples_ratio_for_kmeans,
            random_state=self.random_state
        )

        self.k_means = KMeans(n_clusters=self.clusters, random_state=self.random_state)

        self.k_means.fit(descriptors_for_kmeans)
    
    def eval(self, data: TripletDataset) -> float:
        x_test = np.empty(shape=(len(data), 256, 256, 3))
        y_test = np.empty(shape=(len(data),), dtype=np.int)

        for idx in trange(len(data)):
            item = data[idx]
            x_test[idx] = item["a"]
            y_test[idx] = item["a_y"]
        

        x_test_encoded = self.encode_as_bovw(x_test)

        anmrr_value = anmrr(x_test_encoded, y_test[:, None], euclidean_distances)
        return anmrr_value

    def get_descriptors_for_dataset(self, data: TripletDataset, labels=True):
        desc = []
        matching_labels = []
        for idx in trange(len(data)):
            item = data[idx]

            img = item["a"]
            if labels:
                label = item["a_y"]

            cv_img = img_as_ubyte(img)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
            _, d = self.sift.detectAndCompute(cv_img, None)
            if d is not None:
                desc.append(d)

                if labels:
                    matching_labels.append(np.repeat(label, len(d)))

        if labels:
            return desc, matching_labels
        else:
            return desc

    def get_descriptors(
        self, images: np.ndarray, labels: Union[np.ndarray, None] = None
    ):
        desc = []
        matching_labels = []

        for idx, img in tqdm(
            enumerate(images),
            desc="Calculating SIFT descriptors",
            total=images.shape[0],
        ):
            cv_img = img_as_ubyte(img)
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
            _, d = self.sift.detectAndCompute(cv_img, None)
            if d is not None:
                desc.append(d)

                if labels is not None:
                    matching_labels.append(np.repeat(labels[idx], len(d)))

        if labels is not None:
            return desc, matching_labels
        else:
            return desc

    # def encode_dataset_as_bovw(self, data: TripletDataset) -> np.ndarray:
    #     descriptors = self.get_descriptors_for_dataset(data, labels=False)

    #     res = np.empty(shape=(len(data), self.k_means.n_clusters))

    #     for idx, desc in tqdm(
    #         enumerate(descriptors), total=len(descriptors), desc="Encoding as BOVW"
    #     ):
    #         words = self.k_means.predict(desc)
    #         bovw, _ = np.histogram(words, bins=range(self.k_means.n_clusters + 1))
    #         res[idx] = bovw / desc.shape[0]

    #     return res

    def encode_as_bovw(self, x: np.ndarray) -> np.ndarray:
        descriptors = self.get_descriptors(x)

        res = np.empty(shape=(x.shape[0], self.k_means.n_clusters))

        for idx, desc in tqdm(
            enumerate(descriptors), total=len(descriptors), desc="Encoding as BOVW"
        ):
            words = self.k_means.predict(desc)
            bovw, _ = np.histogram(words, bins=range(self.k_means.n_clusters + 1))
            res[idx] = bovw / desc.shape[0]

        return res
