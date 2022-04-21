import matplotlib.pyplot as plot
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


path = r'/Users/WangHao/工作/纳米光子中心/王老师相关/Random Diffuser/20220113.png'
img = Image.open(path).convert('L')
img = transforms.Compose([transforms.ToTensor(), transforms.Resize((31, 31))])(img).squeeze(0)

img_93x93_1 = torch.zeros((93, 93))
img_93x93_1[1 * 31:2 * 31, 1 * 31:2 * 31] = img

flag = False
if flag:
    [switch1, switch2, switch3, switch4, switch5] = [True, True, True, True, True]
else:
    [switch1, switch2, switch3, switch4, switch5] = [False, False, False, False, False]

# 可视化以图像为中心周围填充一倍原图大小空白
if switch1:
    fig1 = plot.figure(1)
    plot.imshow(img_93x93_1, cmap='gray')
    plot.axis('off')
    plot.show()

list_kernel_3x3 = []
kernel_3x3_1 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
list_kernel_3x3.append(kernel_3x3_1)
kernel_3x3_2 = torch.tensor([[2, 0, 0], [0, -1, 0], [0, 0, -1]])
list_kernel_3x3.append(kernel_3x3_2)
kernel_3x3_3 = torch.tensor([[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]])
list_kernel_3x3.append(kernel_3x3_3)
kernel_3x3_4 = torch.tensor([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
list_kernel_3x3.append(kernel_3x3_4)
kernel_3x3_5 = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
list_kernel_3x3.append(kernel_3x3_5)
kernel_3x3_6 = torch.tensor([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
list_kernel_3x3.append(kernel_3x3_6)
kernel_3x3_7 = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
list_kernel_3x3.append(kernel_3x3_7)
kernel_3x3_8 = torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
list_kernel_3x3.append(kernel_3x3_8)
kernel_3x3_9 = torch.tensor([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
list_kernel_3x3.append(kernel_3x3_9)

# 拼接小卷积核成大核再整体与扩充输入图像中心对齐
kernel_9x9_1 = torch.zeros((9, 9))
k = 0
for c in range(3):
    for r in range(3):
        kernel_9x9_1[c * 3:(c + 1) * 3, r * 3:(r + 1) * 3] = list_kernel_3x3[k]
        k += 1
kernel_93x93_1 = torch.zeros((93, 93))
kernel_93x93_1[46 - 4:46 + 5, 46 - 4:46 + 5] = kernel_9x9_1

# 使每个小卷积核与扩充输入图像的子图像依次中心对齐
kernel_93x93_2 = torch.zeros((93, 93))
k = 0
for r in range(3):
    for c in range(3):
        kernel_93x93_2[(31 // 2 - 1) + 31 * r:(31 // 2 + 2) + 31 * r, (31 // 2 - 1) + 31 * c:(31 // 2 + 2) + 31 * c] = \
        list_kernel_3x3[k]
        k += 1

kernel_93x93_1 = torch.rot90(kernel_93x93_1, 2, (-2, -1))
kernel_93x93_2 = torch.rot90(kernel_93x93_2, 2, (-2, -1))

# 可视化平铺卷积核
if switch2:
    fig2 = plot.figure(2)
    vmin = np.array(kernel_9x9_1).min(initial=0)
    vmax = np.array(kernel_9x9_1).max(initial=1)
    for i in range(9):
        plot.subplot(3, 3, i + 1)
        plot.imshow(list_kernel_3x3[i], vmin=vmin, vmax=vmax, cmap='gray')
        plot.axis('off')
    plot.show()

    fig3 = plot.figure(3)
    vmin = np.array(kernel_9x9_1).min(initial=0)
    vmax = np.array(kernel_9x9_1).max(initial=1)
    plot.imshow(kernel_9x9_1, vmin=vmin, vmax=vmax, cmap='gray')
    plot.axis('off')
    plot.show()

    fig4 = plot.figure(4)
    vmin = np.array(kernel_9x9_1).min(initial=0)
    vmax = np.array(kernel_9x9_1).max(initial=1)
    plot.imshow(kernel_93x93_1, vmin=vmin, vmax=vmax, cmap='gray')
    plot.axis('off')
    plot.show()

    fig5 = plot.figure(5)
    vmin = np.array(kernel_9x9_1).min(initial=0)
    vmax = np.array(kernel_9x9_1).max(initial=1)
    plot.imshow(kernel_93x93_2, vmin=vmin, vmax=vmax, cmap='gray')
    plot.axis('off')
    plot.show()

# 计算填充图像和平铺卷积核频域乘积
img_93x93_1_k = torch.fft.fftshift(torch.fft.fft2(img_93x93_1), dim=(-2, -1))
kernel_93x93_1_k = torch.fft.fftshift(torch.fft.fft2(kernel_93x93_1), dim=(-2, -1))
kernel_93x93_2_k = torch.fft.fftshift(torch.fft.fft2(kernel_93x93_2), dim=(-2, -1))

# 可视化输入图像和卷积核频谱
if switch3:
    fig6 = plot.figure(6)
    plot.imshow(torch.abs(img_93x93_1_k), cmap='gray')
    plot.axis('off')
    plot.show()

    fig7 = plot.figure(7)
    plot.imshow(img_93x93_1_k.angle(), cmap='gray')
    plot.axis('off')
    plot.show()

    fig8 = plot.figure(8)
    plot.imshow(torch.abs(kernel_93x93_1_k), cmap='gray')
    plot.axis('off')
    plot.show()

    fig9 = plot.figure(9)
    plot.imshow(kernel_93x93_1_k.angle(), cmap='gray')
    plot.axis('off')
    plot.show()

    fig10 = plot.figure(10)
    plot.imshow(torch.abs(kernel_93x93_2_k), cmap='gray')
    plot.axis('off')
    plot.show()

    fig11 = plot.figure(11)
    plot.imshow(kernel_93x93_2_k.angle(), cmap='gray')
    plot.axis('off')
    plot.show()

# 序列大小为奇数时 fftshift 和 ifftshift 才能构成一个闭合的移位圈，偶数时任意两个都能构成一个闭合的移位圈
result_1_k = img_93x93_1_k * kernel_93x93_1_k
result_1_k_abs = torch.abs(torch.fft.ifftshift(torch.fft.ifft2(result_1_k), dim=(-2, -1)))

result_2_k = img_93x93_1_k * kernel_93x93_2_k
result_2_k_abs = torch.abs(torch.fft.ifftshift(torch.fft.ifft2(result_2_k), dim=(-2, -1)))

# 可视化频域乘积运算后的图像
if switch4:
    fig12 = plot.figure(12)
    plot.imshow(result_1_k_abs, cmap='gray')
    plot.axis('off')
    plot.show()

    fig13 = plot.figure(13)
    plot.imshow(result_2_k_abs, cmap='gray')
    plot.axis('off')
    plot.show()

# 计算填充图像和平铺卷积核空域卷积
result_1 = torch.conv2d(img_93x93_1.unsqueeze(0).unsqueeze(0), kernel_93x93_1.unsqueeze(0).unsqueeze(0),
                        padding=kernel_93x93_1.shape[-1] // 2).squeeze(0).squeeze(0)
result_2 = torch.conv2d(img_93x93_1.unsqueeze(0).unsqueeze(0), kernel_93x93_2.unsqueeze(0).unsqueeze(0),
                        padding=kernel_93x93_1.shape[-1] // 2).squeeze(0).squeeze(0)

# 可视化空域卷积运算后的图像
if switch5:
    fig14 = plot.figure(14)
    plot.imshow(result_1, cmap='gray')
    plot.axis('off')
    plot.show()

    fig15 = plot.figure(15)
    plot.imshow(result_2, cmap='gray')
    plot.axis('off')
    plot.show()

pass
