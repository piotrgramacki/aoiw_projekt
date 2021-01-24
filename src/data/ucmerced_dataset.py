from typing import Optional
import numpy as np
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

class TripletDataModule(LightningDataModule):
    def __init__(self, data_dir: str, image_size: int, train_percentage: float, batch_size: int, augment=True, normalize=True, permute=False, 
    jitter_brightness = 0, jitter_contrast=0, jitter_saturation=0, jitter_hue=0, rotate=False, random_seed = 42):
        super().__init__()
        self.data_dir = data_dir
        self.train_percentage = train_percentage
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.image_size = image_size

        train_transforms = [transforms.Resize((self.image_size, self.image_size))]
        test_transforms = [transforms.Resize((self.image_size, self.image_size))]

        if augment:
            train_transforms.append(transforms.RandomHorizontalFlip())
            train_transforms.append(transforms.RandomVerticalFlip())
            train_transforms.append(transforms.ColorJitter(brightness=jitter_brightness, contrast=jitter_contrast, saturation=jitter_saturation, hue=jitter_hue))
            if rotate:
                train_transforms.append(transforms.RandomRotation(45))
        
        train_transforms.append(transforms.ToTensor())
        test_transforms.append(transforms.ToTensor())

        if normalize:
            train_transforms.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
            test_transforms.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        
        if permute:
            train_transforms.append(transforms.Lambda(lambda t: t.permute(1,2,0)))
            test_transforms.append(transforms.Lambda(lambda t: t.permute(1,2,0)))

        self.augmentations = transforms.Compose(train_transforms)
        self.transforms = transforms.Compose(test_transforms)

        self.class_names = sorted([path for path in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, path))])
        self.label_name_mapping = dict(enumerate(self.class_names))
        self.name_label_mapping = {class_name: label for label, class_name in self.label_name_mapping.items()}
        image_paths = []
        y = []
        for class_name in self.class_names:
            class_path = os.path.join(self.data_dir, class_name)
            class_image_paths = []
            for file_name in os.listdir(class_path):
                path = os.path.join(self.data_dir, class_name, file_name)
                class_image_paths.append(path)
            image_paths.extend(class_image_paths)
            y.extend([self.name_label_mapping[class_name]] * len(class_image_paths))
        
        self.image_paths = np.array(image_paths)
        self.y = np.array(y)
        
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str]):
        if stage == 'fit' or stage is None:
            images_train, images_val, y_train, y_val = train_test_split(
                self.image_paths,
                self.y,
                train_size=self.train_percentage,
                shuffle=True,
                stratify=self.y,
                random_state=self.random_seed,
            )
            self.train_dataset = TripletDataset(self.class_names, self.label_name_mapping, self.name_label_mapping, images_train, y_train, self.augmentations)
            self.val_dataset = TripletDataset(self.class_names, self.label_name_mapping, self.name_label_mapping, images_val, y_val, self.transforms)
    

    def train_dataloader(self):
        train = DataLoader(self.train_dataset, batch_size=self.batch_size)
        return train

    def val_dataloader(self):
        val = DataLoader(self.val_dataset, batch_size=self.batch_size)
        return val

class TripletDataset(Dataset):
    def __init__(self, class_names, label_name_mapping, name_label_mapping, image_paths, y, transforms):
        self.class_names = class_names
        self.label_name_mapping = label_name_mapping
        self.name_label_mapping = name_label_mapping
        self.image_paths = image_paths
        self.y = y
        self.transforms = transforms
        self.positive_mapping = {}

        for class_name in self.class_names:
            class_label = self.name_label_mapping[class_name]
            y_where_class = self.y == class_label
            class_indices = np.nonzero(y_where_class)[0]
            self.positive_mapping[self.name_label_mapping[class_name]] = class_indices

        all_indices = np.arange(len(self.image_paths))

        self.negative_mapping = {
            class_label: all_indices[~np.in1d(all_indices, positive)]
            for class_label, positive in self.positive_mapping.items()
        }

    def __len__(self):
        return self.image_paths.shape[0]

    def get_positive(self, anchor_index):
        return self.get_sample(self.positive_mapping, anchor_index)

    def get_negative(self, anchor_index):
        return self.get_sample(self.negative_mapping, anchor_index)

    def get_sample(self, mapping_dict, y):
        idx = int(np.random.choice(mapping_dict[y]))
        return idx

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        def read_image(idx):
            img_name = self.image_paths[idx]
            img = Image.open(img_name)
            tensor = self.transforms(img)
            return tensor

        y = self.y[idx]
        anchor = read_image(idx)
        p_id = self.get_positive(y)
        n_id = self.get_negative(y)
        positive = read_image(p_id)
        negative = read_image(n_id)
        positive_y = self.y[p_id]
        negative_y = self.y[n_id]
        anchor_path = self.image_paths[idx]

        sample = {'a': anchor, 'p': positive, 'n': negative, 'a_y': y, 'p_y': positive_y, 'n_y': negative_y, 'path': anchor_path }
        return sample

    def get_label_mapping(self, id):
        return self.label_name_mapping[id]

    def get_index_mapping(self, name):
        return self.name_label_mapping[name]


class UcMercedDataset(Dataset):
    def __init__(self, root_dir: str, image_size: int, train: bool = True):
        
        if train:
            self.transforms = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.root_dir = root_dir
        self.class_names = sorted([path for path in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, path))])
        self.label_name_mapping = dict(enumerate(self.class_names))
        self.name_label_mapping = {class_name: label for label, class_name in self.label_name_mapping.items()}
        self.positive_mapping = {}
        image_paths = []
        y = []
        for class_name in self.class_names:
            class_path = os.path.join(self.root_dir, class_name)
            class_image_paths = []
            for file_name in os.listdir(class_path):
                path = os.path.join(self.root_dir, class_name, file_name)
                # if os.path.isfile(path):
                #     img = Image.open(path)
                #     if img.shape[:2] == (256, 256):
                class_image_paths.append(path)
            self.positive_mapping[self.name_label_mapping[class_name]] = np.arange(len(image_paths), len(image_paths) + len(class_image_paths))
            image_paths.extend(class_image_paths)
            y.extend([self.name_label_mapping[class_name]] * len(class_image_paths))
        self.image_paths = np.array(image_paths)
        self.y = np.array(y)
        all_indices = np.arange(len(self.image_paths))

        self.negative_mapping = {
            class_label: all_indices[~np.in1d(all_indices, positive)]
            for class_label, positive in self.positive_mapping.items()
        }

    def __len__(self):
        return self.image_paths.shape[0]

    def get_positive(self, anchor_index):
        return self.get_sample(self.positive_mapping, anchor_index)

    def get_negative(self, anchor_index):
        return self.get_sample(self.negative_mapping, anchor_index)

    def get_sample(self, mapping_dict, y):
        idx = int(np.random.choice(mapping_dict[y]))
        return idx

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        def read_image(idx):
            img_name = self.image_paths[idx]
            img = Image.open(img_name)
            tensor = self.transforms(img)
            return tensor
            # return io.imread(img_name) / 255.0

        y = self.y[idx]
        anchor = read_image(idx)
        p_id = self.get_positive(y)
        n_id = self.get_negative(y)
        positive = read_image(p_id)
        negative = read_image(n_id)
        positive_y = self.y[p_id]
        negative_y = self.y[n_id]
        anchor_path = self.image_paths[idx]

        sample = {'a': anchor, 'p': positive, 'n': negative, 'a_y': y, 'p_y': positive_y, 'n_y': negative_y, 'path': anchor_path }
        return sample

    def get_label_mapping(self, id):
        return self.label_name_mapping[id]

    def get_index_mapping(self, name):
        return self.name_label_mapping[name]
