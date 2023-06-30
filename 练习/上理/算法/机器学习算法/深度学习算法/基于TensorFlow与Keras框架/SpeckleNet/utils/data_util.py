import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def image_loader(path):
    return Image.open(path).convert('L')


class MyDataset(Dataset):
    def __init__(self, imgFile, data_root = None, transform = None,  loader=image_loader):
        imgs = []
        with open(imgFile, 'r') as f:
            for line in f:
                temp = line.strip().split(' ')
                # input1 label
                imgs.append((temp[0], temp[1]))

        self.data_root = data_root
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn1, labelfn = self.imgs[index]
        img1 = self.loader(os.path.join(self.data_root, fn1))
        label = self.loader(os.path.join(self.data_root, labelfn))

        if self.transform is not None:
            img1 = self.transform(img1)
            label = self.transform(label)

        img1 = self._ImgtoNumpy(img1)
        label = self._ImgtoNumpy(label)

        img1 = img1.astype(np.float64) / 255
        ir_label = np.expand_dims((255-label) / 255, 2)
        label = np.expand_dims(label / 255, 2)
        #label = np.concatenate((label, ir_label), axis=2)

        fn1 = fn1.split('\\')[1:]
        fn = fn1[0] + ',' + fn1[1]

        return [img1], label, fn

    def __len__(self):
        return len(self.imgs)

    def _ImgtoTensor(self, pilImg):
        return transforms.ToTensor()(pilImg)

    def _ImgtoNumpy(self, pilImg):
        return np.array(pilImg)

def collate_fn(batch):
    image, label, fn = zip(*batch)
    image = np.transpose(np.asarray(image), (0, 2, 3, 1))
    label = np.asarray(label)
    return [image], label, fn

def generate_dataloader(config):
    input_size = (config['input_size'][0], config['input_size'][1])

    trainset = MyDataset(config['train_db'], config['db_root'],
                            transform=transforms.Compose([
                                transforms.Resize(input_size)
                            ]), loader=image_loader)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'],
                                              shuffle=True, num_workers=0, collate_fn=collate_fn)

    testset = MyDataset(config['test_db'], config['db_root'],
                            transform=transforms.Compose([
                                transforms.Resize(input_size)
                            ]), loader=image_loader)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=0, collate_fn=collate_fn)

    return trainloader, testloader

if __name__ == '__main__':

    trainset = MyDataset('..\\all_image\\label_01_train.txt', '../all_image',
                                    transform=transforms.Compose([
                                        transforms.Resize((256, 256))
                                    ]),loader=image_loader)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=3,
                                              shuffle=True, num_workers=0,collate_fn=collate_fn)

    for batch_idx, (dataList, target, fn) in enumerate(trainloader):
        print(batch_idx, (dataList, target, fn))

