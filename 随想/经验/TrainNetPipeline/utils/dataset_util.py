# encoding:utf-8
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from utils.load_util import file_loader, data_to_one


class MyDataSet1(Dataset):
    """
    one data to one label
    """
    def __init__(self, data_root, data_file, data_size, label_size, config):
        self.data_root = data_root
        self.data_size = data_size
        self.label_size = label_size
        self.loader = file_loader(config)

        datafn_list = []
        with open(data_file, 'r') as f:
            for line in f:
                temp = line.strip().split(' ')
                datafn_list.append((temp[0], temp[1]))
        self.datafn_list = datafn_list

    def __getitem__(self, index):
        datafn, labelfn = self.datafn_list[index]
        data = self.loader(os.path.join(self.data_root, datafn), 0)
        data = transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((self.data_size[-2],
                                                      self.data_size[-1]))])(data)
        label = self.loader(os.path.join(self.data_root, labelfn), 1)
        label = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((self.label_size[-2],
                                                       self.label_size[-1]))])(label)

        if datafn.split('\\')[1:]:
            temp = datafn.split('\\')[1:]
            fn = temp[0] + '_' + temp[1]
        else:
            temp = datafn.split('/')[1:]
            fn = temp[0] + '_' + temp[1]

        return data, label, fn

    def __len__(self):
        return len(self.datafn_list)


class MyDataSet2(Dataset):
    """
    one data to two label
    """
    def __init__(self, data_root, data_file, data_size, label_size, otherLabel_size, config):
        self.data_root = data_root
        self.data_size = data_size
        self.label_size = label_size
        self.edge_size = otherLabel_size
        self.loader = file_loader(config)

        datafn_list = []
        with open(data_file, 'r') as f:
            for line in f:
                temp = line.strip().split(' ')
                datafn_list.append((temp[0], temp[1], temp[2]))

        self.datafn_list = datafn_list

    def __getitem__(self, index):
        datafn, labelfn, otherLabelfn = self.datafn_list[index]
        data = self.loader(os.path.join(self.data_root, datafn), 0)
        data = torch.from_numpy(data)
        data = transforms.Compose([transforms.Resize((self.data_size[-2],
                                                      self.data_size[-1]))])(data)
        label = self.loader(os.path.join(self.data_root, labelfn), 1)
        label = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((self.label_size[-2],
                                                       self.label_size[-1]))])(label)
        edge_abel = self.loader(os.path.join(self.data_root, otherLabelfn), 1)
        edge_abel = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((self.edge_size[-2],
                                                           self.edge_size[-1]))])(edge_abel)

        if datafn.split('\\')[1:]:
            temp = datafn.split('\\')[1:]
            fn = temp[0] + '_' + temp[1]
        else:
            temp = datafn.split('/')[1:]
            fn = temp[0] + '_' + temp[1]

        return data, [label, edge_abel], fn

    def __len__(self):
        return len(self.datafn_list)


def train_test_dataloader(config):
    trainset = MyDataSet2(data_root=config['database_root'], data_file=config['train_database'],
                          data_size=config['data_size'], label_size=config['label_size'], otherLabel_size=config['otherLabel_size'],
                          config=config)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config['train_loader_batch'], shuffle=True,
                                              num_workers=4, pin_memory=True)

    testset = MyDataSet2(data_root=config['database_root'], data_file=config['test_database'],
                         data_size=config['data_size'], label_size=config['label_size'], otherLabel_size=config['otherLabel_size'],
                         config=config)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=config['test_loader_batch'], shuffle=False,
                                             num_workers=4, pin_memory=True)

    return trainloader, testloader


def verify_dataloader(config):
    verifyset = MyDataSet2(data_root=config['database_root'], data_file=config['verify_database'],
                           data_size=config['data_size'], label_size=config['label_size'], otherLabel_size=config['otherLabel_size'],
                           config=config)

    verifyloader = torch.utils.data.DataLoader(verifyset, batch_size=config['verify_loader_batch'], shuffle=False,
                                               num_workers=0, pin_memory=True)

    return verifyloader


if __name__ == '__main__':
    import json
    with open(r'../config.json', 'r') as f1:
        config1 = json.load(f1)

    trainset1 = MyDataSet2(data_root="../data_input_label", data_file="../data_input_label/label_train.txt",
                           data_size=config1['data_size'], label_size=config1['label_size'],
                           otherLabel_size=config1['otherLabel_size'],
                           config=config1)

    trainloader1 = torch.utils.data.DataLoader(dataset=trainset1, batch_size=config1['test_loader_batch'], shuffle=True,
                                               num_workers=0, pin_memory=True)
    for datas in trainloader1:
        pass

    from collections.abc import Iterable, Iterator
    a = range(5)
    print(isinstance(trainset1, Iterable), isinstance(trainset1, Iterator), sep=' ')
    print(trainset1.__dir__())  # 具有可迭代性质非可迭代对象和可迭代器器, 没有 __iter__() 与 __next__() 方法
    print(a.__dir__())  # 同上

    print(isinstance(iter(trainset1), Iterable), isinstance(iter(trainset1), Iterator), sep=' ')
    print(iter(trainset1).__dir__())  # 具有了 __iter__() 与 __next__() 方法, 变成了迭代器
    print(iter(a).__dir__())  # 同上
    print(iter(a))

    print(trainloader1.__dir__())
    # 具有 __iter__() 方法返回的对象属于 _BaseDataLoaderIter 类，实现了 __iter__() 与 __next__() 是迭代器即是一个生成器
    print(iter(trainloader1).__dir__())
    print(iter(trainloader1))











