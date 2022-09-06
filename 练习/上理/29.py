import time
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class MyConv(nn.Module):
    def __init__(self):
        super().__init__()
        """下面参数是假设"""
        conv1 = DoubleConv(128, 64, 128)
        conv2 = DoubleConv(128, 64, 128)
        conv3 = DoubleConv(128, 64, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


import numpy as np
import matplotlib.pyplot as plot
fig = plot.figure(1, figsize=(2, 2), dpi=300)
image = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
plot.imshow(image, cmap='gray')
plot.margins(0., 0.)  # 让轴中的内容紧贴轴
plot.axis('off')
plot.subplots_adjust(0., 0., 1., 1., 0., 0.)  # 让全部轴所处的区域与图完全重合
fig.savefig('image.tiff', bbox_inches='tight', pad_inches=0., transparent=True)
plot.show()
time.sleep(5)
plot.close(fig=fig)
