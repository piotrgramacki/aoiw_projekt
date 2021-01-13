import pathlib
import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset
import os
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms
from typing import List, Tuple

class OrtoDataset(Dataset):
    def __init__(self, root_dir: str, image_size: int, train: bool = True):
        
        if train:
            self.transforms = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.root_dir = root_dir
        self.anchors_dir = os.path.join(self.root_dir, "anchor")
        self.positive = os.path.join(self.root_dir, "positive")

        
        file_names = os.listdir(self.anchors_dir)
        anchor_paths = [os.path.join(self.anchors_dir, file_name) for file_name in file_names]
        without_extensions = [pathlib.Path(f).stem for f in file_names]

        self.paths_and_names = list(zip(anchor_paths, without_extensions))

    def __len__(self):
        return len(self.paths_and_names)

    def list_positives_for_anchor(self, anchor_index) -> Tuple[List[str], str]:
        _, anchor_without_extension = self.paths_and_names[anchor_index]
        positive_path = os.path.join(self.positive, anchor_without_extension)
        positive_examples_paths = [os.path.join(positive_path, f) for f in os.listdir(positive_path)]
        return positive_examples_paths, positive_path

    def get_positive_path(self, anchor_index) -> str:
        positive_examples_paths, _ = self.list_positives_for_anchor(anchor_index)
        return np.random.choice(positive_examples_paths)

    def get_negative_path(self, anchor_index) -> str:
        negative_index = anchor_index
        while negative_index == anchor_index:
            negative_index = np.random.randint(len(self))

        negative_path, _ = self.paths_and_names[negative_index]

        anchors_for_negative, _ = self.list_positives_for_anchor(negative_index)
        anchors_for_negative.append(negative_path)

        chosen_negative = np.random.choice(anchors_for_negative)
        return chosen_negative

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        def read_image(path: str):
            img = Image.open(path)
            tensor = self.transforms(img)
            return tensor

        anchor_path, _ = self.paths_and_names[idx]
        positive_path = self.get_positive_path(idx)
        negative_path = self.get_negative_path(idx)

        anchor = read_image(anchor_path)
        positive = read_image(positive_path)
        negative = read_image(negative_path)

        sample = {'a': anchor, 'p': positive, 'n': negative, 'path': anchor_path }
        return sample
