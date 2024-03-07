import os, sys
from torch.utils.data import Dataset
import numpy as np
from PIL import Image



# HorseZebra
class HorseZebraDataset(Dataset):
    def __init__(self, root_y, root_x, transform=None):
        self.root_y = root_y
        self.root_x = root_x
        self.transform = transform

        self.y_images = os.listdir(root_y)
        self.x_images = os.listdir(root_x)
        self.length_dataset = max(len(self.y_images), len(self.x_images)) # 1000, 1500
        self.y_len = len(self.y_images)
        self.x_len = len(self.x_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        y_img = self.y_images[index % self.y_len]
        x_img = self.x_images[index % self.x_len]

        zebra_path = os.path.join(self.root_y, y_img)
        horse_path = os.path.join(self.root_x, x_img)

        y_img = np.array(Image.open(zebra_path).convert("RGB"))
        x_img = np.array(Image.open(horse_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=y_img, image0=x_img)
            y_img = augmentations["image"]
            x_img = augmentations["image0"]

        return y_img, x_img