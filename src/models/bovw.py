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

from src.data.ucmerced_dataset import UcMercedDataset
from src.measures import anmrr
from src.settings import TRAIN_DATA_DIRECTORY, TEST_DATA_DIRECTORY, RANDOM_WALKS_DIRECTORY