import os
import plotly.express as px
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from skimage import io
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def list_images_in_classes(data_dir: str) -> Dict[str, List[Tuple[str, str]]]:
    classes = os.listdir(data_dir)
    images_in_classes = {}
    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        class_images_filenames = os.listdir(class_path)
        class_images_paths = [os.path.join(class_path, file_name) for file_name in class_images_filenames]
        images_in_classes[class_name] = list(zip(class_images_filenames, class_images_paths))
    return images_in_classes

def plot_counts(data_dir: str, dataset_name: str, file_name_suffix: str):
    images_in_classes = list_images_in_classes(data_dir)
    class_names_with_counts = { class_name: len(images) for class_name, images in images_in_classes.items() }
    class_names, class_counts = zip(*class_names_with_counts.items())
    class_names = list(class_names)
    class_counts = list(class_counts)
    fig = px.bar(x=class_names, y=class_counts, labels={'x': '', 'y':'Liczba obrazów'}, title=f"Liczności klas w zbiorze {dataset_name}")
    fig.update_xaxes(tickangle=-90, type='category')
    fig.write_image(f'counts_{file_name_suffix}.svg')
    return fig


def show_example_images(data_dir: str, file_name_suffix: str):
    images_in_classes = list_images_in_classes(data_dir)
    num_classes = len(images_in_classes)
    fig = plt.figure(figsize=(8, 11))
    cols = 4
    rows = np.ceil(num_classes / cols)
    # rows = np.ceil(np.sqrt(num_classes))
    for i, (class_name, images) in enumerate(images_in_classes.items()):
        _, image_paths = zip(*images)
        random_example = np.random.choice(image_paths)
        image = io.imread(random_example)
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(image)
        # im_title = class_name[:15] + '...' if len(class_name) > 15 else class_name
        plt.title(class_name, fontsize=8)
        plt.axis("off")
    fig.savefig(f"examples_{file_name_suffix}.png", dpi=300, bbox_inches = 'tight', pad_inches = 0)
    return fig

def get_color_intensity_counts_per_class(data_dir: str):
    images_in_classes = list_images_in_classes(data_dir)
    dfs = {}

    for c, images in images_in_classes.items():
        cols = np.zeros((256, 4))
        values = np.arange(0, 256)
        for name, path in tqdm(images):
            im = io.imread(path)
            im = np.reshape(im, (-1, 3))
            for i in range(3):
                cols[:, i] = np.bincount(im[:, i], minlength=256)
        cols[:, 3] = values

        df = pd.DataFrame(cols, columns=['R', 'G', 'B', 'x'])
        df['x'] = values
        df = df.melt(id_vars=['x'], value_vars=['R', 'G', 'B'], var_name='Kolor')
        dfs[c] = df
    return dfs

def generate_color_histograms(dataframes: Dict[str, pd.DataFrame], output_dir: str):
    for c, df in dataframes.items():
        fig, ax = plt.subplots(figsize=(5.8, 1.95))
        ax = sns.histplot(df, x='x', weights='value', stat='density', hue='Kolor', discrete=True, kde=True, palette={'R': 'red', 'G': 'green', 'B': 'blue'}, legend=False)
        ax.set(title=f'Histogram kolorów dla klasy {c}')
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        file_path = os.path.join(output_dir, f"{c}.png")
        fig.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0)