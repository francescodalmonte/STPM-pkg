import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T2


class FilterClothsDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 is_train: bool,
                 resize: int = 224,
                 cropsize: int = 224):
        self.data_path = data_path
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set_transforms
        if self.is_train:
            self.transform_v2 = T2.Compose([T2.Resize(resize, Image.BILINEAR),
                                            T2.ToDtype(torch.float32, scale=True),
                                            # augmentation
                                            T2.RandomAdjustSharpness(sharpness_factor=3),
                                            T2.ColorJitter(brightness=.1, contrast=.1),
                                            T2.RandomHorizontalFlip(p=0.5),
                                            T2.RandomVerticalFlip(p=0.5),
                                            T2.RandomResizedCrop(resize, scale=(0.85, 1.0), ratio=(0.85, 1.15)),
                                            T2.Normalize(mean=[0.5], std=[.08])
                                            ])
        else:
            self.transform_v2 = T2.Compose([T2.Resize(resize, Image.BILINEAR),
                                            T2.ToDtype(torch.float32, scale=True),
                                            T2.Normalize(mean=[0.5], std=[.08])
                                            ])
            
    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        if y == 0:
            mask = np.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask).convert("L")
        mask = tv_tensors.Mask(mask)
        y = torch.tensor(y, dtype=torch.long)

        x = Image.open(x).convert('RGB')
        x = tv_tensors.Image(x)

        x, mask = self.transform_v2(x, mask)

        return x, y, mask


    def __len__(self):
        return len(self.x)


    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.data_path, "custom", phase, "tele")
        gt_dir = os.path.join(self.data_path, "custom", phase+"_maps", "tele")

        classes = sorted(os.listdir(img_dir))
        for cl in classes:
            # load images
            img_class_dir = os.path.join(img_dir, cl)
            img_fpath_list = sorted([os.path.join(img_class_dir, f) for f in os.listdir(img_class_dir)
                                     if (f.endswith('.png') or f.endswith('.jpg'))])
            x.extend(img_fpath_list)

            # load gt maps
            if cl == 'normal':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, cl)
                img_fname_list = [os.path.basename(fp) for fp in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, fn) for fn in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)
    

def custom_collate_function(batch):
    x, y, mask = zip(*batch)
    x = torch.stack(x)
    y = torch.stack(y)
    mask = torch.stack(mask)

    return x, y, mask


class FilterClothsDataset_multi(Dataset):
    def __init__(self,
                 data_path: str,
                 is_train: bool,
                 resize: int = 224,
                 cropsize: int = 224):
        self.data_path = data_path
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set_transforms
        if self.is_train:
            self.transform_v2 = T2.Compose([T2.Resize(resize, Image.BILINEAR),
                                            T2.ToDtype(torch.float32, scale=True),
                                            #T2.Normalize(mean=[0.5, 0.5, 0.5], std=[.08, .08, .08]),
                                            # augmentation
                                            T2.RandomAdjustSharpness(sharpness_factor=3),
                                            T2.ColorJitter(brightness=.1, contrast=.1),
                                            T2.RandomHorizontalFlip(p=0.5),
                                            T2.RandomVerticalFlip(p=0.5),
                                            T2.RandomResizedCrop(resize, scale=(0.85, 1.1), ratio=(0.85, 1.15))
                                            ])
        else:
            self.transform_v2 = T2.Compose([T2.Resize(resize, Image.BILINEAR),
                                            T2.ToDtype(torch.float32, scale=True),
                                            #T2.Normalize(mean=[0.5, 0.5, 0.5], std=[.08, .08, .08])
                                            ])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        if y == 0:
            mask = np.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask).convert("L")
        mask = tv_tensors.Mask(mask)
        y = torch.tensor(y, dtype=torch.long)

        x = Image.open(x).convert('RGB')
        x = tv_tensors.Image(x)

        x, mask = self.transform_v2(x, mask)

        return x, y, mask


    def __len__(self):
        return len(self.x)


    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.data_path, "custom", phase, "tele")
        gt_dir = os.path.join(self.data_path, "custom", phase+"_maps", "tele")

        classes = sorted(os.listdir(img_dir))
        for cl in classes:
            # load images
            img_class_dir = os.path.join(img_dir, cl)
            img_fpath_list = sorted([os.path.join(img_class_dir, f) for f in os.listdir(img_class_dir)
                                     if (f.endswith('.png') or f.endswith('.jpg'))])
            print(img_fpath_list)
            x.extend(img_fpath_list)

            # load gt maps
            if cl == 'normal':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, cl)
                img_fname_list = [os.path.basename(fp) for fp in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, fn) for fn in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)
    

def is_image(filename):
    exts = ['.jpg', '.jpeg', '.png', '.bmp', 
            '.JPG', '.JPEG', '.PNG', '.BMP']
    return any(filename.endswith(extension) for extension in exts)


class ImagenetteDataset(Dataset):
    def __init__(self, data_path, resize, seed=42,
                 max_samples=np.inf, prefetch=False):
        self.data_path = data_path
        self.seed = seed
        self.max_samples = max_samples
        self.prefetch = prefetch
        self.transforms = T2.Compose([
            T2.Resize((resize, resize)),
            T2.Grayscale(num_output_channels=3),
            T2.ToDtype(torch.float32, scale=True),
            # augmentation
            T2.RandomAdjustSharpness(sharpness_factor=3),
            T2.ColorJitter(brightness=.1, contrast=.1),
            T2.RandomHorizontalFlip(p=0.5),
            T2.RandomVerticalFlip(p=0.5),
            T2.RandomResizedCrop(resize, scale=(0.6, 1.), ratio=(0.6, 1.4)),
            T2.Normalize(mean=[0.449], std=[0.226])
        ])

        # load dataset
        self.x = self.load_dataset_folder()

        # prefetch data if needed
        if self.prefetch:
            self.prefetch_data()

    def get_from_path(self, x_path):
        # load image 
        x = Image.open(x_path).convert('RGB')
        x = tv_tensors.Image(x)

        # apply transforms
        x = self.transforms(x)

        return x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.prefetch:
            return x, torch.tensor(0), torch.zeros(x.shape[-2:]).unsqueeze(0)
        else:
            x = self.get_from_path(x)
            return x, torch.tensor(0), torch.zeros(x.shape[-2:]).unsqueeze(0)

    def load_dataset_folder(self):
        """Loop through the dataset folder and load images paths."""

        x = []

        root = os.path.join(self.data_path, "train")
        img_folders = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]

        for folder in img_folders:
            for file in os.listdir(folder):
                if is_image(file):
                    x.append(os.path.join(folder, file))

        # shuffle the dataset
        if self.seed is not None:
            np.random.seed(self.seed)
        np.random.shuffle(x)

        # clip the dataset if needed
        if self.max_samples < len(x):
            x = x[:self.max_samples]
        
        return x
    
    def prefetch_data(self):
        print("Prefetching data...")
        for idx in range(len(self)):
            print(f"{idx}/{len(self)}", end="\r")
            self.x[idx] = self.get_from_path(self.x[idx])
    

class InfiniteDataLoader:
    """borrowed from https://discuss.pytorch.org/t/infinite-dataloader/17903/16"""
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)  # Reset the data loader
            data = next(self.data_iter)
        return data


