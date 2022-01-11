import numpy as np
import torch
import torch.nn as nn



class AdjustSize(nn.Module):
    def __init__(self):
        super(AdjustSize, self).__init__()

    def forward(self, x):
        return x.view(x.size(0),-1)

class ExtendDim(nn.Module):
    def __init__(self):
        super(ExtendDim, self).__init__()

    def forward(self, x):
        return x.unsqueeze(1)

class Conv1DGenerator(nn.Module):
    def __init__(self):
        super(Conv1DGenerator, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(100, 64),
            nn.LeakyReLU(inplace=True),
            ExtendDim(),
            nn.BatchNorm1d(1)
        )
        self.con1 = nn.Sequential(
            nn.Conv1d(1,128,3,padding=1),
            nn.Tanh(),
            nn.BatchNorm1d(128)
        )
        self.con2 = nn.Sequential(
            nn.Conv1d(128, 64, 3, padding=1),
            nn.Tanh(),
            nn.BatchNorm1d(64)
        )
        self.con3 = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1),
            nn.Tanh(),
            nn.BatchNorm1d(32)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(32*64,512),
            nn.Tanh(),
            ExtendDim(),
            nn.BatchNorm1d(1),
            AdjustSize(),
            nn.Linear(512,43), # 43 is the size of output, such as 43 means R^(1x43)
            # nn.ReLU(inplace=True),
            ExtendDim(),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.con1(x)
        x = self.con2(x)
        x = self.con3(x)
        x = x.view(x.size(0),-1)
        x = self.fc2(x)
        return x


class Conv1Discriminator(nn.Module):
    def __init__(self):
        super(Conv1Discriminator, self).__init__()

        self.con1 = nn.Sequential(
            nn.Conv1d(1,64,3,padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(64)
        )
        self.con2 = nn.Sequential(
            nn.Conv1d(64, 32, 3, padding=1),
            nn.Tanh(),
            nn.BatchNorm1d(32)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(43*32,128),
            nn.Tanh(),
            nn.Linear(128,2),
            # nn.Softmax()
        )

    def forward(self, x):
        x = self.con1(x)
        x = self.con2(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        return x

class Conv2DGenerator(nn.Module):
    def __init__(self):
        super(Conv2DGenerator, self).__init__()
        
        self.linear = torch.nn.Linear(100, 1024*4*4)
        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=3, kernel_size=4,
                stride=2, padding=1, bias=False
            )
        )
        self.out = torch.nn.Tanh()

    def forward(self, x):
        # Project and reshape
        x = self.linear(x)
        x = x.view(x.shape[0], 1024, 4, 4)
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Apply Tanh
        return self.out(x)

class Conv2Discriminator(nn.Module):
    def __init__(self):
        super(Conv2Discriminator, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=128, kernel_size=4, 
                stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(1024*4*4, 2),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Flatten and apply sigmoid
        x = x.view(-1, 1024*4*4)
        x = self.out(x)
        return x


if __name__ == '__main__':
    

    g = Conv2DGenerator()
    d = Conv2Discriminator()