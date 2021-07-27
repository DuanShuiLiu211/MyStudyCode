import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils

from models import Conv1DGenerator,Conv1Discriminator,Conv2DGenerator,Conv2Discriminator
from dataset import MyDataset
import torch.backends.cudnn as cudnn
import math
import os
import matplotlib.pyplot as plt

def get_generator_input_sampler(m,n):
    return torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian

def train_gan(modelG,modelD,epoch,trainloader,optimizerG,optimizerD,criterion):
    modelG.eval()
    modelD.train()
    for batch_idx, data in enumerate(trainloader):
        data = data.cuda()
        optimizerD.zero_grad()
        # real
        output_real = modelD(data)
        loss_real = criterion(output_real, torch.ones([output_real.size(0)],dtype=torch.int64).cuda())
        loss_real.backward()

        # fake
        fake_noise = get_generator_input_sampler(trainloader.batch_size,100).cuda()
        fake_data = modelG(fake_noise)
        output_fake = modelD(fake_data)
        loss_fake = criterion(output_fake, torch.zeros([trainloader.batch_size],dtype=torch.int64).cuda())
        loss_fake.backward()
        optimizerD.step()
        optimizerD.zero_grad()

        modelG.train()
        modelD.eval()
        # 2. Train G on D's response (but DO NOT train D on these labels)
        fake_noise = get_generator_input_sampler(trainloader.batch_size, 100).cuda()
        fake_data = modelG(fake_noise)
        output_fake = modelD(fake_data)
        loss_fake_g = criterion(output_fake, torch.ones([trainloader.batch_size],dtype=torch.int64).cuda())  # Train G to pretend it's genuine
        loss_fake_g.backward()
        optimizerG.step()  # Only optimizes G's parameters

        if batch_idx % 20 == 0:
            print('Epoch:{}, G_Loss:{}, D_real_Loss:{},D_fake_Loss:{}.'.format(epoch,loss_fake_g.data.item(),
                                                                               loss_real.data.item(),loss_fake.data.item()))
        torch.save(modelG.state_dict(), 'G_model_temp.pth')
        torch.save(modelD.state_dict(), 'D_model_temp.pth')

def save_imglist(img_list):
    save_dir = './generated_img'
    # for idx, img_data in enumerate(img_list):
    #     vutils.save_image(img_data, os.path.join(save_dir, f'{idx}.jpg'))
    rows = math.ceil(math.sqrt(len(img_list)))
    vutils.save_image(torch.cat(img_list), os.path.join(save_dir, 'generation.jpg'),normalize=True, nrow=rows)
    

def generate_data():

    modelG = Conv2DGenerator().cuda()
    modelG.load_state_dict(torch.load('G_model_temp.pth')) # G_model.pth is a file of trained model
    modelG.eval()
    data_list = []
    for i in range(9):
        fake_noise = get_generator_input_sampler(1, 100).cuda()
        generated_data = modelG(fake_noise).cpu().detach()
        data_list.append(generated_data)

    return data_list

def train_gan_issue():
    # dataset, train_list.txt is a file including all the filenames of training samples
    # ./img/ is a folder name including all training samples
    trainset = MyDataset('train_list.txt', './img/', transform=transforms.Compose([
        transforms.Resize((64,64)),
    	transforms.ToTensor()]))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,shuffle=True, num_workers=1)

    cudnn.benchmark = True

    modelG = Conv2DGenerator().cuda()
    modelD = Conv2Discriminator().cuda()

    criterion = nn.CrossEntropyLoss()

    optimizerG = optim.SGD(
        modelG.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5)
    optimizerD = optim.SGD(
        modelD.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5)

    entireEpoch = 300 
    best_auc, best_weight = None, None
    for epoch in range(entireEpoch):
        train_gan(modelG, modelD, epoch, trainloader, optimizerG, optimizerD, criterion)

    torch.save(modelG.state_dict(), 'G_model_final.pth')
    torch.save(modelD.state_dict(), 'D_model_final.pth')


if __name__ == '__main__':

    # train stage
    train_gan_issue()

    # after training, generation stage
    # imglist = generate_data()
    # save_imglist(imglist)