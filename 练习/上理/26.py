import torch
import numpy as np
from PIL import Image
from scipy.signal import convolve
from torchvision import transforms
import matplotlib.pyplot as plot

# 加载图像
path = r'/Users/WangHao/工作/纳米光子中心/全光相关/实验-0303/0303.png'
img = Image.open(path).convert('L')
img = transforms.Compose([transforms.ToTensor(), transforms.Resize((203, 203))])(img)
img = img.squeeze(0)
# img = torch.nn.functional.pad(img, [1,1,1,1], value=0)
# 比较空域卷积与频域卷积
a = torch.arange(1,10,1).reshape(3,3).float()
# a = torch.tensor([[1. ,1 ,1], [1 ,-8 ,1], [1 ,1 ,1]])
result1 = torch.tensor(convolve(img, a, 'same'))
result2 = torch.conv2d(img.unsqueeze(0).unsqueeze(0), a.unsqueeze(0).unsqueeze(0), padding=1).squeeze(0).squeeze(0)

img_k_1 = torch.fft.fft2(img)
b = torch.nn.functional.pad(a, [100,100,100,100], value=0)
b_k_1 = torch.fft.fft2(b)
b_k_2 = torch.fft.fft2(b.reshape(-1).flipud().reshape(203, 203))

result3 = torch.fft.ifftshift(torch.fft.ifft2(img_k_1 * b_k_1))
result4 = torch.fft.ifftshift(torch.fft.ifft2(img_k_1 * b_k_2))

error1 = result3.abs() - result1.abs()
error2 = result4.abs() - result2.abs()

plot.figure(0)
plot.subplot(221)
plot.imshow(img)
plot.axis("off")
plot.subplot(222)
plot.imshow(img)
plot.axis("off")
plot.subplot(223)
plot.imshow(a)
plot.axis("off")
plot.subplot(224)
plot.imshow(b)
plot.axis("off")

plot.figure(1)
plot.imshow(result1.abs())
plot.colorbar()
plot.figure(2)
plot.imshow(result2.abs())
plot.colorbar()

plot.figure(3)
plot.subplot(121)
plot.imshow(result3.abs())
plot.colorbar()
plot.subplot(122)
plot.imshow(result3.angle())
plot.colorbar()

plot.figure(4)
plot.subplot(121)
plot.imshow(result4.abs())
plot.colorbar()
plot.subplot(122)
plot.imshow(result4.angle())
plot.colorbar()

plot.figure(5)
plot.imshow(error1)
plot.axis("off")
plot.colorbar()
plot.figure(6)
plot.imshow(error2)
plot.axis("off")
plot.colorbar()

plot.show()