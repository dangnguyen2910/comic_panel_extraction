import os 
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import tv_tensors

class VagabondDatasetPanel(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir 
        self.transform = transform 
        self.mask_transform = mask_transform

    def __getitem__(self, idx):
        img_name_list = os.listdir(self.img_dir)
        mask_name_list = os.listdir(self.mask_dir)

        mask_name = mask_name_list[idx]
        img_name = mask_name[:-3] + "jpg"

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)


        img = read_image(img_path)
        mask = read_image(mask_path)


        if self.transform:
            img = self.transform(img)

        if self.mask_transform:
            mask = self.transform(mask)


        return img, mask

    def __len__(self):
        return len(os.listdir(self.mask_dir))