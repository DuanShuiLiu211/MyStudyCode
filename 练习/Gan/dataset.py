import torch
from torch.utils.data import Dataset
import os

import matplotlib.pyplot as plt
from PIL import Image

def default_loader(path):
    return Image.open(path)

class MyDataset(Dataset):
    def __init__(self, filename, data_root, transform, file_loader = default_loader):

        with open(filename,'r') as f:
            data_list = f.readlines()

        self.data = [fn.strip() for fn in data_list]
        self.transform = transform
        self.file_loader = file_loader
        self.data_root = data_root

    def __getitem__(self, index):
        fn = self.data[index]
        img = self.file_loader(os.path.join(self.data_root,fn))
        
        imgTensor = self.transform(img)

        return imgTensor

    def __len__(self):
        return len(self.data)