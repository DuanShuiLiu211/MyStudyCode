# encoding:utf-8
import sys
import os
import copy
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from utils.load_util import FileLoader


# 图像重建
class ReconstructionDataSetOne(Dataset):
    """
    one data to one label
    """
    def __init__(self, data_file, input_size, label_size, config):
        self.input_size = input_size
        self.label_size = label_size
        self.file_loader = FileLoader(config)

        datafn_list = []
        with open(data_file, 'r') as f:
            for line in f:
                temp = line.strip().split(' ')
                datafn_list.append((temp[0], temp[1]))
        self.datafn_list = datafn_list

    def __getitem__(self, index):
        inputfn, labelfn = self.datafn_list[index]
        inputs = self.file_loader(inputfn, "inputs")
        inputs = transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((self.input_size[-2],
                                                      self.input_size[-1]))])(inputs)
        labels = self.file_loader(labelfn, "labels")
        labels = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((self.label_size[-2],
                                                       self.label_size[-1]))])(labels)

        if inputfn.split('\\')[1:]:
            temp = inputfn.split('\\')[1:]
            fn = temp[0] + '_' + temp[1]
        else:
            temp = inputfn.split('/')[1:]
            fn = temp[0] + '_' + temp[1]

        return inputs, labels, fn

    def __len__(self):
        return len(self.datafn_list)


class ReconstructionDataSetTwo(Dataset):
    """
    one data to two label
    """
    def __init__(self, data_file, input_size, label_one_size, label_two_size, config):
        self.input_size = input_size
        self.label_one_size = label_one_size
        self.label_two_size = label_two_size
        self.file_loader = FileLoader(config)

        datafn_list = []
        with open(data_file, 'r') as f:
            for line in f:
                temp = line.strip().split(' ')
                datafn_list.append((temp[0], temp[1], temp[2]))

        self.datafn_list = datafn_list

    def __getitem__(self, index):
        inputfn, oneLabelfn, twoLabelfn = self.datafn_list[index]
        inputs = self.file_loader(inputfn, "inputs")
        inputs = torch.from_numpy(inputs)
        inputs = transforms.Compose([transforms.Resize((self.input_size[-2],
                                                        self.input_size[-1]))])(inputs)
        one_label = self.file_loader(os.path.join(self.data_root, oneLabelfn), "labels")
        one_label = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((self.label_size[-2],
                                                           self.label_size[-1]))])(one_label)
        two_abel = self.file_loader(os.path.join(self.data_root, twoLabelfn), 1)
        two_abel = transforms.Compose([transforms.ToTensor(),
                                       transforms.Resize((self.edge_size[-2],
                                                          self.edge_size[-1]))])(two_abel)

        if inputfn.split('\\')[1:]:
            temp = inputfn.split('\\')[1:]
            fn = temp[0] + '_' + temp[1]
        else:
            temp = inputfn.split('/')[1:]
            fn = temp[0] + '_' + temp[1]

        return inputs, [one_label, two_abel], fn

    def __len__(self):
        return len(self.datafn_list)


def train_verify_reconstruction_dataloader(config):
    trainset = ReconstructionDataSetTwo(data_file=config['train_database'],
                                        input_size=config['input_size'], label_one_size=config['label_one_size'], label_two_size=config['label_two_size'],
                                        config=config)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config['train_loader_batch'], shuffle=True,
                                              num_workers=8, pin_memory=True)

    verifyset = ReconstructionDataSetTwo(data_file=config['test_database'], 
                                         input_size=config['input_size'], label_one_size=config['label_one_size'], label_two_size=config['label_two_size'],
                                         config=config)

    verifyloader = torch.utils.data.DataLoader(dataset=verifyset, batch_size=config['test_loader_batch'], shuffle=False,
                                               num_workers=8, pin_memory=True)

    return trainloader, verifyloader


def test_reconstruction_dataloader(config):
    testset = ReconstructionDataSetTwo(data_file=config['test_database'],
                                       input_size=config['input_size'], label_one_size=config['label_one_size'], label_two_size=config['label_two_size'],
                                       config=config)

    testloader = torch.utils.data.DataLoader(testset, batch_size=config['test_loader_batch'], shuffle=False,
                                             num_workers=0, pin_memory=True)

    return testloader


# 图像分类
class ClassifyDataSetOne(Dataset):
    """
    one data to one label
    """
    def __init__(self, data_file, input_size, input_mean_std, config):
        datafn_list = []
        with open(data_file, 'r') as f:
            for line in f:
                temp = line.strip().split(' ')
                datafn_list.append((temp[0], temp[1]))
        self.datafn_list = datafn_list
        self.input_size = input_size
        self.input_mean_std = input_mean_std
        self.file_loader = FileLoader(config)

    def __getitem__(self, index):
        inputsfn, labelsfn = self.datafn_list[index]
        inputs = np.asarray(self.file_loader(inputsfn, "inputs"), dtype=np.uint8)
        
        rawdatas = transforms.Compose([transforms.ToPILImage(),
                                       transforms.Resize((self.input_size[-2], self.input_size[-1])),
                                       transforms.ToTensor()])(copy.deepcopy(inputs))
        inputs = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize((self.input_size[-2], self.input_size[-1])),
                                     transforms.ToTensor(),
                                     transforms.Normalize(self.input_mean_std[0], self.input_mean_std[1])])(inputs)
        labels = 3
        if labelsfn == "0":
            labels = torch.tensor(0, dtype=torch.long)
        elif labelsfn == "1":
            labels = torch.tensor(1, dtype=torch.long)
        elif labelsfn == "2":
            labels = torch.tensor(2, dtype=torch.long)
        else:
            print(f"the {labelsfn} is not exist in kind")
            sys.exit

        if inputsfn.split('\\')[1:]:
            temp = inputsfn.split('\\')
            fn = temp[-2] + '_' + temp[-1]
        else:
            temp = inputsfn.split('/')
            fn = temp[-2] + '_' + temp[-1]

        return inputs, labels, fn, rawdatas

    def __len__(self):
        return len(self.datafn_list)


def train_verify_classify_dataloader(config):
    trainset = ClassifyDataSetOne(data_file=config['train_database'],
                                  input_size=config['input_size'], input_mean_std=config["norm_input_mean_std"],
                                  config=config)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=config['train_loader_batch'], shuffle=True,
                                              num_workers=4, pin_memory=True)

    verifyset = ClassifyDataSetOne(data_file=config['verify_database'],
                                   input_size=config['input_size'], input_mean_std=config["norm_input_mean_std"],
                                   config=config)

    verifyloader = torch.utils.data.DataLoader(dataset=verifyset, batch_size=config['verify_loader_batch'], shuffle=False,
                                               num_workers=4, pin_memory=True)

    return trainloader, verifyloader


def test_classify_dataloader(config):
    testset = ClassifyDataSetOne(data_file=config['test_database'],
                                 data_size=config['input_size'], data_mean_std=config["norm_input_mean_std"],
                                 config=config)

    testloader = torch.utils.data.DataLoader(testset, batch_size=config['test_loader_batch'], shuffle=False,
                                             num_workers=0, pin_memory=True)

    return testloader


if __name__ == '__main__':
    import json
    with open(r'C:\Users\10469\Desktop\WorkFile\Code\TrainNetPipeline-main\config.json', 'r') as f1:
        config1 = json.load(f1)

    trainset1 = ClassifyDataSetOne(data_file=r"C:\Users\10469\Desktop\WorkFile\Code\TrainNetPipeline-main\datas\label_train.txt",
                                   input_size=config1['input_size'], input_mean_std=config1["norm_input_mean_std"],
                                   config=config1)

    trainloader1 = torch.utils.data.DataLoader(dataset=trainset1, batch_size=config1['train_loader_batch'], shuffle=True,
                                               num_workers=0, pin_memory=True)
    for datas1 in trainloader1:
        pass
