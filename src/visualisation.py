import os

import imageio
import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.settings import RANDOM_WALKS_DIRECTORY


def random_walk(
        images: np.ndarray,
        images_encoded: np.ndarray,
        trained_n_neighbours: NearestNeighbors,
        steps: int = 20,
        starting_index: int = 42,
        search_range: int = 10,
        filename_prefix: str = ""
):
    current_index = starting_index

    result = []

    for i in range(steps + 1):
        result.append(np.array(images[current_index] * 255, dtype=np.uint8))

        neighbours_idx = trained_n_neighbours.kneighbors(
            images_encoded[current_index].reshape(1, -1),
            n_neighbors=search_range,
            return_distance=False
        ).squeeze()

        current_index = np.random.choice(neighbours_idx, 1).squeeze()

    print("Done walking")

    imageio.mimsave(
        os.path.join(
            RANDOM_WALKS_DIRECTORY,
            f'{filename_prefix}start_at_{starting_index}.gif'
        ),
        result
    )
