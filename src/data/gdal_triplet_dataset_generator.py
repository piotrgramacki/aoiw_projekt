import multiprocessing

import click
import rasterio
from rasterio.windows import from_bounds, Window, get_data_window, transform
import numpy as np
import os
from tqdm import tqdm
import pathlib
from rasterio.io import DatasetReader
import math
from typing import Tuple, List
from multiprocessing import Pool
import multiprocessing as mp

from tqdm.cli import main


def assert_dir_exists(dir: str):
    if not os.path.exists(dir):
        os.mkdir(dir)



def chunks(l, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(l), n)
    res = []
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        res.append(l[si:si+(d+1 if i < r else d)])
    return res


class GdalTripletDatasetGenerator:
    def __init__(
        self,
        source_dataset_path: str,
        shift_min_percent: float = 0.2,
        shift_max_percent: float = 0.8,
        driver: str = "GTiff",
        file_extension=".tif",
    ):
        self.source_dataset_path = source_dataset_path
        self.shift_min_percent = shift_min_percent
        self.shift_max_percent = shift_max_percent
        self.driver = driver
        self.file_extension = file_extension

    def generate_dataset(self, anchor_files_dir_path: str, destination_dir_path: str, shifted_files_per_anchor: int):
        assert_dir_exists(destination_dir_path)

        with rasterio.open(self.source_dataset_path) as dataset:
            for anchor_file in tqdm(os.listdir(anchor_files_dir_path)):
                self._assert_destination_dir_for_file_exists(
                    destination_dir_path, anchor_file
                )
                source_file_path = os.path.join(anchor_files_dir_path, anchor_file)
                self._generate_shifted_files(source_file_path, shifted_files_per_anchor, dataset, destination_dir_path)


    def _generate_shifted_files(self, source_file_path: str, shifted_files_per_anchor: int, dataset: DatasetReader, destination_dir_path: str):
        for i in range(shifted_files_per_anchor):
            image_valid = False
            while not image_valid:
                window, image =  self._generate_shifted_image(source_file_path, dataset)
                image_valid = self._is_image_valid(image)
                if image_valid:
                    destination_path = self._get_destination_file_path(destination_dir_path, source_file_path, i)
                    self._save_raster_file(destination_path, dataset, window, image)

    def _generate_shifted_image(self, source_file_path: str, dataset: DatasetReader) -> Tuple[Window, np.ndarray]:
        with rasterio.open(source_file_path) as image:
            image_width, image_height = image.width, image.height
            image_x, image_y = image.xy(0, 0)

        min_shift_x = math.floor(self.shift_min_percent * image_width)
        max_shift_x = math.floor(self.shift_max_percent * image_width)

        min_shift_y = math.floor(self.shift_min_percent * image_height)
        max_shift_y = math.floor(self.shift_max_percent * image_height)

        x_min_bound = 0
        x_max_bound = dataset.width - image_width
        y_min_bound = 0
        y_max_bound = dataset.height - image_height


        dataset_row, dataset_col = dataset.index(image_x, image_y)

        def _valid_lower_point(threshold, value):  
            return max(threshold, value)
        
        def _valid_upper_point(threshold, value):
            return min(threshold, value)

        def _valid_lower_x(x):
            return _valid_lower_point(x_min_bound, x)
        
        def _valid_lower_y(y):
            return _valid_lower_point(y_min_bound, y)
        
        def _valid_upper_x(x):
            return _valid_upper_point(x_max_bound, x)
        
        def _valid_upper_y(y):
            return _valid_upper_point(y_max_bound, y)


        lower_x_range = np.arange(_valid_lower_x(dataset_col - max_shift_x), _valid_lower_x(dataset_col - min_shift_x))
        upper_x_range = np.arange(_valid_upper_x(dataset_col + min_shift_x), _valid_upper_x(dataset_col + max_shift_x))

        lower_y_range = np.arange(_valid_lower_y(dataset_row - max_shift_y), _valid_lower_y(dataset_row - min_shift_y))
        upper_y_range = np.arange(_valid_upper_y(dataset_row + min_shift_y), _valid_upper_y(dataset_row + max_shift_y))
        x_range = np.concatenate([
            lower_x_range, upper_x_range
        ])

        y_range = np.concatenate([
            lower_y_range, upper_y_range
        ])

        new_x = np.random.choice(x_range)
        new_y = np.random.choice(y_range)

        dataset_window = Window(new_x, new_y, image_width, image_height)
        p = dataset.read(window=dataset_window)
        return dataset_window, p

    def _assert_destination_dir_for_file_exists(
        self, destination_dir_path: str, anchor_file: str
    ):
        source_file_name = pathlib.Path(anchor_file).stem
        destination_dir = os.path.join(destination_dir_path, source_file_name)
        assert_dir_exists(destination_dir)

    def _get_destination_file_path(
        self, destination_dir_path: str, file_path: str, file_number: int
    ) -> str:
        source_file_name = pathlib.Path(file_path).stem
        destination_file_name = f"{source_file_name}_{file_number}{self.file_extension}"
        destination_file_path = os.path.join(
            destination_dir_path, source_file_name, destination_file_name
        )
        return destination_file_path

    def _save_raster_file(
        self, destination_path: str, dataset: DatasetReader, window: Window, image: np.ndarray
    ):
        save_profile = self._get_save_profile(dataset, window)

        with rasterio.open(destination_path, "w", **save_profile) as destination:
            destination.write(image)

    def _get_save_profile(self, dataset: DatasetReader, window: Window):
        kwargs = dataset.meta.copy()
        kwargs.update(
            {
                "driver": self.driver,
                "height": window.height,
                "width": window.width,
                "transform": transform(window, dataset.transform),
                "compress": "lzw",
            }
        )
        return kwargs

    def _is_image_valid(self, image: np.ndarray) -> bool:
        return True # this problem has been solved differently
        # percent_zeros = np.sum(image == 0) / np.size(image)
        # return percent_zeros < 0.001

@click.command('generate')
@click.option('-s', '--source-raster', 'source_raster', type=click.Path(dir_okay=False), required=True)
@click.option('-a', '--anchors-path', 'anchors_path', type=click.Path(file_okay=False), required=True)
@click.option('-o', '--output-path', 'output_path', type=click.Path(file_okay=False, writable=True, exists=True), required=True)
@click.option('-n', '--num-positive', 'num_positive', type=int, required=False, default=20)
def generate_triplet_dataset(source_raster: str, anchors_path: str, output_path: str, num_positive: int):
    np.random.seed(42)
    generator = GdalTripletDatasetGenerator(source_raster)
    generator.generate_dataset(anchors_path, output_path, num_positive)

if __name__ == "__main__":
    generate_triplet_dataset()