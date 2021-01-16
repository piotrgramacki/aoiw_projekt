import re
from typing import Callable, Union, List, Tuple
import numpy as np


def anmrr(
    x_features: np.ndarray,
    y: np.ndarray,
    pairwise_distance: Callable[[np.ndarray], np.ndarray],
    K: float = 2.0,
    class_mean: bool = False,
) -> Union[float, Tuple[float, List[Tuple[int, float]]]]:
    """Calculates anmrr measure for dataset given measure

    Parameters
    ----------
    x_features : np.ndarray
        Objects' features. Should be NxF where N-observations, F-features.
    y : np.ndarray
        Objects' labels. Has to be Nx1.
    pairwise_distance : Callable[[np.ndarray], np.ndarray]
        A function that takes x_features and returns NxN matrix of pairwise distances between objects.
    K : float
        A scaling parameter for ground_truth size. (default 2.)
    class_mean : bool
        Whether to calculate in-class anmrr means too. (default False)
    Returns
    -------
    Union[float, Tuple[float, List[Tuple[int, float]]]]
        If class_mean=False then a value between 0 and 1 where lower values indicate better retrieval performance.
        Otherwise this value, and anmrr mean for each class.
    """

    unique_classes, counts = np.unique(y, return_counts=True)

    sorted_counts = counts[np.argsort(unique_classes)]

    query_sizes = sorted_counts[y]
    K_sizes = K * query_sizes

    distances = pairwise_distance(x_features)

    rankings = np.argsort(distances, axis=1)

    selected_classes = y[rankings]

    selected_classes = np.squeeze(selected_classes, axis=-1)

    selected_classes_equal_ground_truth = selected_classes == y

    indices = np.indices(distances.shape)[1] + 1

    ranks = (indices <= K_sizes) * indices + (indices > K_sizes) * 1.25 * K_sizes

    penalties = selected_classes_equal_ground_truth * ranks

    avr = np.sum(penalties, axis=1, keepdims=True) / query_sizes

    nmrr = (avr - 0.5 * (1 + query_sizes)) / (1.25 * K_sizes - 0.5 * (1 + query_sizes))

    anmrr = np.mean(nmrr)

    if class_mean:
        class_anmrr = [(c, np.mean(nmrr[y == c])) for c in unique_classes]
        return anmrr, class_anmrr

    return anmrr


if __name__ == "__main__":
    x = np.array([0])
    y = np.array([0, 1, 1, 2])[:, None]

    dist = lambda x: np.array([[0, 3, 4, 5], [3, 0, 1, 3], [4, 1, 0, 4], [5, 3, 4, 0]])
    print(anmrr(x, y, dist))