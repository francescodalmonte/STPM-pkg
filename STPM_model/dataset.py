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
            self.transform_v2 = T2.Compose([#T2.Resize(resize, Image.BILINEAR),
                                            T2.ToDtype(torch.float32, scale=True),
                                            #T2.Normalize(mean=[0.5, 0.5, 0.5], std=[1., 1., 1.])
                                            # augmentation
                                            T2.RandomAdjustSharpness(sharpness_factor=3),
                                            T2.ColorJitter(brightness=.4, contrast=[.4, 2.]),
                                            T2.RandomHorizontalFlip(p=0.5),
                                            T2.RandomVerticalFlip(p=0.5),
                                            T2.RandomResizedCrop(resize, scale=(0.25, 1.0), ratio=(0.8, 1.2))
                                            ])
        else:
            self.transform_v2 = T2.Compose([#T2.Resize(resize, Image.BILINEAR),
                                            T2.ToDtype(torch.float32, scale=True)
                                            #T2.Normalize(mean=[0.5, 0.5, 0.5], std=[1., 1., 1.])
                                            ])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        if y == 0:
            mask = np.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask).convert("L")
        mask = tv_tensors.Mask(mask)

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
                                     if f.endswith('.png')])
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