import os

import imageio
import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.settings import RANDOM_WALKS_DIRECTORY
from typing import List, Dict, Tuple
import plotly.express as px
from skimage import io
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
from umap import UMAP

def random_walk(
        images: np.ndarray,
        images_encoded: np.ndarray,
        trained_n_neighbours: NearestNeighbors,
        steps: int = 20,
        starting_index: int = 42,
        search_range: int = 10,
        filename_prefix: str = ""
) -> None:
    """

    :param images: original images used for gif creation
    :param images_encoded: images embeddings (should match images)
    :param trained_n_neighbours: NearestNeighbours model built on given
    embeddings
    :param steps: how many steps in random walk
    :param starting_index: on which image to start
    :param search_range: how many closest images to choose next step from
    :param filename_prefix: optional prefix for generated gif file
    """
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


def visualize_anmrr_per_class(anmrr_per_class: List[Tuple[int, float]], label_name_mapping: Dict[int, str], dataset_name: str, result_path: str, additional_title_string=None):
    names_with_values = [(label_name_mapping[label], value) for label, value in anmrr_per_class]
    names, values = zip(*names_with_values)
    title = f"ANMRR per class in {dataset_name} dataset"
    if additional_title_string is not None:
        title += f" [{additional_title_string}]"
    fig = px.bar(x=names, y=values, labels={'x': '', 'y': 'ANMRR'}, title=title)
    fig.update_xaxes(tickangle=-90)
    fig.update_layout({'font': {'size': 30}})
    fig.write_image(result_path, width=1800, height=960)
    return fig


def visualize_best_and_worst_queries(paths, embeddings, nmrr, n_queries, output_path, n_images_per_query):
    distances = euclidean_distances(embeddings)
    nmrr = nmrr.squeeze()

    best_queries_indices = np.argsort(nmrr)
    n_best_queries_indices = best_queries_indices[:n_queries]
    n_worst_queries_indices = best_queries_indices[-n_queries:][::-1]


    rankings = np.argsort(distances, axis=1)
    selected_images = paths[rankings]

    cols = rows = int(np.ceil(np.sqrt(n_images_per_query)))

    def visualize_queries(query_indices, query_type: str):
        for query_number, query_index in enumerate(query_indices):
            query_image_path = paths[query_index]
            query_image_name = os.path.split(str(query_image_path))[1]
            query = selected_images[query_index, :].squeeze()

            fig=plt.figure(figsize=(8, 8))
            for i in range (cols * rows):
                path = query[i]
                image = io.imread(path)
                fig.add_subplot(rows, cols, i+1)
                plt.title(os.path.split(path)[1])
                plt.axis("off")
                
                plt.imshow(image)
            fig.suptitle(f"Response to query: {query_image_name}, NMRR:{nmrr[query_index]:.2f}")
            fig.savefig(os.path.join(output_path, f"{query_type}_{query_number}.png"), dpi=300, bbox_inches='tight', pad_inches=0)
            plt.clf()
            plt.cla()
            plt.close()
    visualize_queries(n_worst_queries_indices, "worst")
    visualize_queries(n_best_queries_indices, "best")


def visualize_embeddings(embeddings: np.ndarray, image_paths, output_path):
    tsne_embeddings = TSNE(n_components=2).fit_transform(embeddings)
    umap_embeddings = UMAP().fit_transform(embeddings)
    _visualize_embeddings(tsne_embeddings, image_paths, os.path.join(output_path, f"tsne.png"))
    _visualize_embeddings(umap_embeddings, image_paths, os.path.join(output_path, f"umap.png"))


def _visualize_embeddings(two_dimensional_embeddings, image_paths, output_path):
    def get_image(path):
            img = Image.open(path)
            a = np.asarray(img)
            return OffsetImage(a, zoom=0.15)

    fig, ax = plt.subplots(figsize=(15,15))
    ax.scatter(two_dimensional_embeddings[:, 0], two_dimensional_embeddings[:, 1]) 
    for image_path, (x, y) in zip(image_paths, two_dimensional_embeddings):
        ab = AnnotationBbox(get_image(image_path), (x, y), frameon=False)
        ax.add_artist(ab)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.cla()
    plt.close()
