import os

import nrrd
import numpy as np
import torch
from torch.utils.data import Dataset


class FattyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(
            self.mask_dir, self.images[index].replace(".nrrd", "_mask.nrrd")
        )
        image, _ = nrrd.read(img_path, index_order="C")
        mask, _ = nrrd.read(mask_path, index_order="C")
        image = np.array(image, dtype="float32")
        mask = np.array(mask, dtype="int64")
        # image = np.expand_dims(image, 0)
        # mask = np.expand_dims(mask, 0)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        image = torch.from_numpy(image.copy())
        mask = torch.from_numpy(mask.copy())
        image = image.type(torch.float)
        mask = mask.type(torch.long)
        # return image, mask
        return image, mask
