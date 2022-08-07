import torch
import time
import matplotlib.pyplot as plot
from PIL import Image
from torchvision import transforms

# 1. 基础参数设置
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
"""
sc：图像的尺寸，不论输入图像尺寸多大一律调整为 sc*sc 
scale：平铺的比例，比如 scale=3, 3*3 -> 9*9 
ke_sc：卷积核的尺寸，比如 ke_sc=3*3
ke_pad_sc：满足卷积不重叠下的卷积核尺寸，外圈补零，比如 ke_sc=3*3,ke_pad_sc=5*5
"""
sc = 101
scale = 3
ke_sc = 11
ke_pad_sc = sc + 2 * (ke_sc // 2)

# 2. 以输入图像中心在周围填充空白
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
"""
true：周围补零的输入图像
k：频域
"""
path = r'E:\桌面\0303.png'
img = Image.open(path).convert('L')
img = transforms.Compose([transforms.ToTensor(), transforms.Resize((sc, sc))])(img).squeeze(0)

img_ture_1 = torch.zeros((scale * (sc + 2 * (ke_sc // 2)), scale * (sc + 2 * (ke_sc // 2))))
img_ture_1[(scale // 2) * (sc + 2 * (ke_sc // 2)) + (ke_sc // 2):
           (scale // 2 + 1) * (sc + 2 * (ke_sc // 2)) - (ke_sc // 2),
(scale // 2) * (sc + 2 * (ke_sc // 2)) + (ke_sc // 2):
(scale // 2 + 1) * (sc + 2 * (ke_sc // 2)) - (ke_sc // 2)] \
    = img
img_ture_1_k = torch.fft.fftshift(torch.fft.fft2(img_ture_1), dim=(-2, -1))
img_ture_1_k_abs = torch.abs(img_ture_1_k)
img_ture_1_k_phase = img_ture_1_k.angle()

# 3. 设计卷积核
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# list_kernel = []
# kernel_1 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# list_kernel.append(kernel_1)
# kernel_2 = torch.tensor([[2, 0, 0], [0, -1, 0], [0, 0, -1]])
# list_kernel.append(kernel_2)
# kernel_3 = torch.tensor([[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]])*2
# list_kernel.append(kernel_3)
# kernel_4 = torch.tensor([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
# list_kernel.append(kernel_4)
# kernel_5 = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# list_kernel.append(kernel_5)
# kernel_6 = torch.tensor([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
# list_kernel.append(kernel_6)
# kernel_7 = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
# list_kernel.append(kernel_7)
# kernel_8 = torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
# list_kernel.append(kernel_8)
# kernel_9 = torch.tensor([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
# list_kernel.append(kernel_9)


# list_kernel = []
# kernel_1 = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# list_kernel.append(kernel_1)
# kernel_2 = torch.tensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]])*5
# list_kernel.append(kernel_2)
# kernel_3 = torch.tensor([[0, 0, -1], [0, 1, 0], [0, 0, 0]])*5
# list_kernel.append(kernel_3)
# kernel_4 = torch.tensor([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
# list_kernel.append(kernel_4)
# kernel_5 = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# list_kernel.append(kernel_5)
# kernel_6 = torch.tensor([[1, 1, 1], [1, -7, 1], [1, 1, 1]])
# list_kernel.append(kernel_6)
# kernel_7 = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
# list_kernel.append(kernel_7)
# kernel_8 = torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
# list_kernel.append(kernel_8)
# kernel_9 = torch.tensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])*5
# list_kernel.append(kernel_9)

list_kernel = []
kernel_1 = torch.tensor([[-1., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [-0., -1., -0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., -0., -1., -0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., -0., -1., -0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., -0., -1., -0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., -0., 6., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
list_kernel.append(kernel_1)

kernel_2 = torch.tensor([[0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 6., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
list_kernel.append(kernel_2)

kernel_3 = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., -0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., -0., -1., -0.],
                         [0., 0., 0., 0., 0., 0., 0., -0., -1., -0., 0.],
                         [0., 0., 0., 0., 0., 0., -0., -1., -0., 0., 0.],
                         [0., 0., 0., 0., 0., -0., -1., -0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 6., -0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
list_kernel.append(kernel_3)

kernel_4 = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [-1., -1., -1., -1., -1., 6., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
list_kernel.append(kernel_4)

kernel_9 = torch.tensor([[-0.1000, -0.0976, -0.0882, -0.0690, -0.0385, 0.0000, 0.0385, 0.0690,
                          0.0882, 0.0976, 0.1000],
                         [-0.1220, -0.1250, -0.1200, -0.1000, -0.0588, 0.0000, 0.0588, 0.1000,
                          0.1200, 0.1250, 0.1220],
                         [-0.1471, -0.1600, -0.1667, -0.1538, -0.1000, 0.0000, 0.1000, 0.1538,
                          0.1667, 0.1600, 0.1471],
                         [-0.1724, -0.2000, -0.2308, -0.2500, -0.2000, 0.0000, 0.2000, 0.2500,
                          0.2308, 0.2000, 0.1724],
                         [-0.1923, -0.2353, -0.3000, -0.4000, -0.5000, 0.0000, 0.5000, 0.4000,
                          0.3000, 0.2353, 0.1923],
                         [-0.2000, -0.2500, -0.3333, -0.5000, -1.0000, 0.0000, 1.0000, 0.5000,
                          0.3333, 0.2500, 0.2000],
                         [-0.1923, -0.2353, -0.3000, -0.4000, -0.5000, 0.0000, 0.5000, 0.4000,
                          0.3000, 0.2353, 0.1923],
                         [-0.1724, -0.2000, -0.2308, -0.2500, -0.2000, 0.0000, 0.2000, 0.2500,
                          0.2308, 0.2000, 0.1724],
                         [-0.1471, -0.1600, -0.1667, -0.1538, -0.1000, 0.0000, 0.1000, 0.1538,
                          0.1667, 0.1600, 0.1471],
                         [-0.1220, -0.1250, -0.1200, -0.1000, -0.0588, 0.0000, 0.0588, 0.1000,
                          0.1200, 0.1250, 0.1220],
                         [-0.1000, -0.0976, -0.0882, -0.0690, -0.0385, 0.0000, 0.0385, 0.0690,
                          0.0882, 0.0976, 0.1000]])
list_kernel.append(kernel_9)

kernel_5 = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 6., -1., -1., -1., -1., -1.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
list_kernel.append(kernel_5)

kernel_6 = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., -0., 6., 0., 0., 0., 0., 0.],
                         [0., 0., 0., -0., -1., -0., 0., 0., 0., 0., 0.],
                         [0., 0., -0., -1., -0., 0., 0., 0., 0., 0., 0.],
                         [0., -0., -1., -0., 0., 0., 0., 0., 0., 0., 0.],
                         [-0., -1., -0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [-1., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
list_kernel.append(kernel_6)

kernel_7 = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 6., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., -1., 0., 0., 0., 0., 0.]])
list_kernel.append(kernel_7)

kernel_8 = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 6., -0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., -0., -1., -0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., -0., -1., -0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., -0., -1., -0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., -0., -1., -0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., -0., -1.]])
list_kernel.append(kernel_8)

# 4. 组合卷积核
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
"""
small：紧凑平铺的小的卷积核 
tiny：最小的卷积核
true：间隔来平铺的大的卷积核
"""
# 4.1. 拼接小卷积核成大核再整体与扩充输入图像中心对齐
# -------------------------------------------------------------------------------------------------------------------- #
kernel_small_temp_1 = torch.zeros((scale * ke_sc, scale * ke_sc))
k = 0
for c in range(scale):
    for r in range(scale):
        kernel_small_temp_1[c * ke_sc:(c + 1) * ke_sc, r * ke_sc:(r + 1) * ke_sc] = list_kernel[k]
        k += 1
kernel_small_1 = torch.zeros((scale * ke_pad_sc, scale * ke_pad_sc))

kernel_small_1[(scale * ke_pad_sc) // 2 - (scale * ke_sc) // 2:(scale * ke_pad_sc) // 2 + ((scale * ke_sc) // 2 + 1),
(scale * ke_pad_sc) // 2 - (scale * ke_sc) // 2:(scale * ke_pad_sc) // 2 + ((scale * ke_sc) // 2 + 1)] \
    = kernel_small_temp_1

fig22 = plot.figure(22, figsize=(6, 6), dpi=300)
plot.imshow(kernel_small_temp_1, cmap='gray')
# plot.axis('off')
plot.colorbar()
plot.show()

# 4.2. 拼接小卷积核成大核再整体与扩充输入图像中心对齐
# -------------------------------------------------------------------------------------------------------------------- #
kernel_tiny_temp_1 = torch.zeros((ke_sc, ke_sc))
for idx in range(scale ** 2):
    kernel_tiny_temp_1 += list_kernel[idx]
kernel_tiny_1 = torch.zeros((scale * ke_pad_sc, scale * ke_pad_sc))
kernel_tiny_1[(scale * ke_pad_sc) // 2 - ke_sc // 2:(scale * ke_pad_sc) // 2 + (ke_sc // 2 + 1),
(scale * ke_pad_sc) // 2 - ke_sc // 2:(scale * ke_pad_sc) // 2 + (ke_sc // 2 + 1)] \
    = kernel_tiny_temp_1

# 4.3. 使每个小卷积核与扩充输入图像的子图像依次中心对齐
# -------------------------------------------------------------------------------------------------------------------- #
kernel_true_1 = torch.zeros((scale * ke_pad_sc, scale * ke_pad_sc))
k = 0
for r in range(scale):
    for c in range(scale):
        kernel_true_1[(ke_pad_sc // 2 - ke_sc // 2) + ke_pad_sc * r:(ke_pad_sc // 2 + ke_sc // 2 + 1) + ke_pad_sc * r,
        (ke_pad_sc // 2 - ke_sc // 2) + ke_pad_sc * c:(ke_pad_sc // 2 + ke_sc // 2 + 1) + ke_pad_sc * c] \
            = list_kernel[k]
        k += 1

# 5. 通过傅里叶变换实现卷积计算以及原卷积计算
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# 5.1. fft small
# -------------------------------------------------------------------------------------------------------------------- #
kernel_small_1_k = []
n = 0
time_start = time.time_ns()
while n <= 0:
    # 计算填充图像和平铺卷积核频域乘积
    # 序列大小为奇数时 fftshift 和 ifftshift 才能构成一个闭合的移位圈，偶数时任意两个都能构成一个闭合的移位圈
    kernel_small_1_k = torch.fft.fftshift(torch.fft.fft2(kernel_small_1), dim=(-2, -1))
    kernel_small_1_k.imag *= -1
    n += 1
time_end = time.time_ns()
time_fft = time_end - time_start
print(time_fft)

# 4f系统的傅里叶面
result_small_1_k = img_ture_1_k * kernel_small_1_k
result_small_1_k_phase = result_small_1_k.angle()
result_small_1_k_abs = torch.abs(result_small_1_k)

# 4f系统的右焦面
result_small_1_abs = torch.abs(torch.fft.fftshift(torch.fft.ifft2(result_small_1_k), dim=(-2, -1)))

# 5.2. fft tiny
# -------------------------------------------------------------------------------------------------------------------- #
kernel_tiny_1_k = []
n = 0
time_start = time.time_ns()
while n <= 0:
    kernel_tiny_1_k = torch.fft.fftshift(torch.fft.fft2(kernel_tiny_1), dim=(-2, -1))
    kernel_tiny_1_k.imag *= -1
    n += 1
time_end = time.time_ns()
time_fft = time_end - time_start
print(time_fft)

result_tiny_1_k = img_ture_1_k * kernel_tiny_1_k
result_tiny_1_k_phase = result_tiny_1_k.angle()
result_tiny_1_k_abs = torch.abs(result_tiny_1_k)

result_tiny_1_abs = torch.abs(torch.fft.fftshift(torch.fft.ifft2(result_tiny_1_k), dim=(-2, -1)))

# 5.3. fft true
# -------------------------------------------------------------------------------------------------------------------- #
kernel_true_1_k = []
n = 0
time_start = time.time_ns()
while n <= 0:
    kernel_true_1_k = torch.fft.fftshift(torch.fft.fft2(kernel_true_1), dim=(-2, -1))
    kernel_true_1_k.imag *= -1
    n += 1
time_end = time.time_ns()
time_fft = time_end - time_start
print(time_fft)

result_true_1_k = img_ture_1_k * kernel_true_1_k
result_true_1_k_abs = torch.abs(result_true_1_k)
result_true_1_k_phase = result_true_1_k.angle()

result_true_1_abs = torch.abs(torch.fft.fftshift(torch.fft.ifft2(result_true_1_k), dim=(-2, -1)))

# 5.4. 卷积
# -------------------------------------------------------------------------------------------------------------------- #
result_small_1_conv = []
result_tiny_1_conv = []
result_true_1_conv = []
n = 0
time_start = time.time_ns()
while n <= 0:
    result_small_1_conv = torch.conv2d(img_ture_1.unsqueeze(0).unsqueeze(0), kernel_small_1.unsqueeze(0).unsqueeze(0),
                                       padding=kernel_small_1.shape[-1] // 2).squeeze(0).squeeze(0)

    result_tiny_1_conv = torch.conv2d(img_ture_1.unsqueeze(0).unsqueeze(0), kernel_tiny_1.unsqueeze(0).unsqueeze(0),
                                      padding=kernel_tiny_1.shape[-1] // 2).squeeze(0).squeeze(0)

    result_true_1_conv = torch.conv2d(img_ture_1.unsqueeze(0).unsqueeze(0), kernel_true_1.unsqueeze(0).unsqueeze(0),
                                      padding=kernel_true_1.shape[-1] // 2).squeeze(0).squeeze(0)
    n += 1
time_end = time.time_ns()
time_conv = time_end - time_start
print(time_conv)

# 6. 可视化
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# 6.1. 可视化开关
# -------------------------------------------------------------------------------------------------------------------- #
flag = True
if flag:
    [switch1, switch2, switch3, switch4] = [True, True, True, True]
else:
    [switch1, switch2, switch3, switch4] = [False, False, False, False]

# 6.2. 可视化输入图像填充前后及其频谱相谱
# -------------------------------------------------------------------------------------------------------------------- #
if switch1:
    fig1 = plot.figure(1, figsize=(6, 6), dpi=300)
    plot.subplot(2, 2, 1)
    plot.imshow(img, cmap='gray')
    # plot.axis('off')
    plot.title('raw')
    plot.colorbar()

    plot.subplot(2, 2, 2)
    plot.imshow(img_ture_1, cmap='gray')
    # plot.axis('off')
    plot.title('raw by pad')
    plot.colorbar()

    plot.subplot(2, 2, 3)
    plot.imshow(img_ture_1_k_abs, cmap='gray')
    # plot.axis('off')
    plot.title('abs')
    plot.colorbar()

    plot.subplot(2, 2, 4)
    plot.imshow(img_ture_1_k_phase, cmap='gray')
    # plot.axis('off')
    plot.title('phase')
    plot.colorbar()

    plot.show()

# 6.3. 可视化平铺卷积核与其频谱相谱
# -------------------------------------------------------------------------------------------------------------------- #

if switch2:
    fig2 = plot.figure(2, figsize=(6, 6), dpi=300)
    plot.subplot(3, 3, 1)
    plot.imshow(kernel_small_1, cmap='gray')
    # plot.axis('off')
    plot.title('raw of small')
    plot.colorbar()

    plot.subplot(3, 3, 4)
    plot.imshow(torch.abs(kernel_small_1_k), cmap='gray')
    # plot.axis('off')
    plot.title('abs')
    plot.colorbar()

    plot.subplot(3, 3, 7)
    plot.imshow(kernel_small_1_k.angle(), cmap='gray')
    # plot.axis('off')
    plot.title('phase')
    plot.colorbar()

    plot.subplot(3, 3, 2)
    plot.imshow(kernel_tiny_1, cmap='gray')
    # plot.axis('off')
    plot.title('raw of tiny')
    plot.colorbar()

    plot.subplot(3, 3, 5)
    plot.imshow(torch.abs(kernel_tiny_1_k), cmap='gray')
    # plot.axis('off')
    plot.title('abs')
    plot.colorbar()

    plot.subplot(3, 3, 8)
    plot.imshow(kernel_tiny_1_k.angle(), cmap='gray')
    # plot.axis('off')
    plot.title('phase')
    plot.colorbar()

    plot.subplot(3, 3, 3)
    plot.imshow(kernel_true_1, cmap='gray')
    # plot.axis('off')
    plot.title('raw of true')
    plot.colorbar()

    plot.subplot(3, 3, 6)
    plot.imshow(torch.abs(kernel_true_1_k), cmap='gray')
    # plot.axis('off')
    plot.title('abs')
    plot.colorbar()

    plot.subplot(3, 3, 9)
    plot.imshow(kernel_true_1_k.angle(), cmap='gray')
    # plot.axis('off')
    plot.title('phase')
    plot.colorbar()
    plot.tight_layout()

    plot.show()

# 6.4. 可视化4f系统傅里叶面及其傅里叶变换输出结果与卷积结果
# -------------------------------------------------------------------------------------------------------------------- #
if switch3:
    fig3 = plot.figure(3, figsize=(6, 6), dpi=300)
    plot.subplot(4, 3, 1)
    plot.imshow(torch.abs(result_small_1_k_abs), cmap='gray')
    plot.axis('off')
    plot.title('fft plane abs of small')
    plot.colorbar()

    plot.subplot(4, 3, 4)
    plot.imshow(torch.abs(result_small_1_k_phase), cmap='gray')
    plot.axis('off')
    plot.title('fft plane phase of small')
    plot.colorbar()

    plot.subplot(4, 3, 7)
    plot.imshow(torch.abs(result_small_1_abs), cmap='gray')
    plot.axis('off')
    plot.title('small result by fft')
    plot.colorbar()

    plot.subplot(4, 3, 10)
    plot.imshow(torch.abs(result_small_1_conv), cmap='gray')
    plot.axis('off')
    plot.title('small result by conv')
    plot.colorbar()

    plot.subplot(4, 3, 2)
    plot.imshow(torch.abs(result_tiny_1_k_abs), cmap='gray')
    plot.axis('off')
    plot.title('fft plane abs of tiny')
    plot.colorbar()

    plot.subplot(4, 3, 5)
    plot.imshow(torch.abs(result_tiny_1_k_phase), cmap='gray')
    plot.axis('off')
    plot.title('fft plane phase of tiny')
    plot.colorbar()

    plot.subplot(4, 3, 8)
    plot.imshow(torch.abs(result_tiny_1_abs), cmap='gray')
    plot.axis('off')
    plot.title('tiny result by fft')
    plot.colorbar()

    plot.subplot(4, 3, 11)
    plot.imshow(torch.abs(result_tiny_1_conv), cmap='gray')
    plot.axis('off')
    plot.title('tiny result by conv')
    plot.colorbar()

    plot.subplot(4, 3, 3)
    plot.imshow(torch.abs(result_true_1_k_abs), cmap='gray')
    plot.axis('off')
    plot.title('fft plane abs of true')
    plot.colorbar()

    plot.subplot(4, 3, 6)
    plot.imshow(torch.abs(result_true_1_k_phase), cmap='gray')
    plot.axis('off')
    plot.title('fft plane phase of true')
    plot.colorbar()

    plot.subplot(4, 3, 9)
    plot.imshow(torch.abs(result_true_1_abs), cmap='gray')
    plot.axis('off')
    plot.title('true result by fft')
    plot.colorbar()

    plot.subplot(4, 3, 12)
    plot.imshow(torch.abs(result_true_1_conv), cmap='gray')
    plot.axis('off')
    plot.title('true result by conv')
    plot.colorbar()

    plot.show()

# 6.5. 可视化fft与conv之间残差以及tiny和true叠放一起后的残差
# -------------------------------------------------------------------------------------------------------------------- #
if switch4:
    fig4 = plot.figure(4, figsize=(6, 6), dpi=300)
    plot.subplot(1, 2, 1)
    plot.imshow(result_true_1_abs - torch.abs(result_true_1_conv), cmap='gray')
    # plot.axis('off')
    plot.title('fft and conv with error of true')
    plot.colorbar()

    # 把结果都叠放到中心？？？这里不是写了第一个4f系统输出的结果都叠起在中心不就是第二个4f系统的输出结果吗
    result_true_tiny_1 = torch.zeros((scale * (sc + 2 * (ke_sc // 2)), scale * (sc + 2 * (ke_sc // 2))))
    for r in range(scale):
        for c in range(scale):
            result_true_tiny_1[(scale // 2) * (sc + 2 * (ke_sc // 2)) + (ke_sc // 2):
                               (scale // 2 + 1) * (sc + 2 * (ke_sc // 2)) - (ke_sc // 2),
            (scale // 2) * (sc + 2 * (ke_sc // 2)) + (ke_sc // 2):
            (scale // 2 + 1) * (sc + 2 * (ke_sc // 2)) - (ke_sc // 2)] \
                += result_true_1_abs[r * (sc + 2 * (ke_sc // 2)) + (ke_sc // 2):
                                     (r + 1) * (sc + 2 * (ke_sc // 2)) - (ke_sc // 2),
                   c * (sc + 2 * (ke_sc // 2)) + (ke_sc // 2):
                   (c + 1) * (sc + 2 * (ke_sc // 2)) - (ke_sc // 2)]
    plot.subplot(1, 2, 2)
    plot.imshow(result_true_tiny_1 - result_tiny_1_abs, cmap='gray')
    # plot.axis('off')
    plot.title('fft of tiny and fft of true with error')
    plot.colorbar()

    # 这个不就是第一个4f系统的输出的频谱吗
    fig5 = plot.figure(5, figsize=(6, 6), dpi=300)
    plot.imshow(result_true_1_abs, cmap='gray')
    plot.axis('off')
    # plot.title('true result by fft')
    plot.colorbar()
    plot.show()

    # 这个不就是第一个4f系统输入直接卷积中间卷积核实际样子的结果吗
    fig6 = plot.figure(6, figsize=(6, 6), dpi=300)
    plot.imshow(result_true_1_conv, cmap='gray')
    plot.axis('off')
    # plot.title('true result by conv')
    plot.colorbar()

    # 这个不就是第一个4f系统的卷积核的相位谱吗，不就是放在4f中间的部分吗
    fig9 = plot.figure(9, figsize=(6, 6), dpi=300)
    plot.imshow(kernel_true_1_k.angle(), cmap='gray')
    plot.axis('off')
    # plot.title('phase')
    plot.colorbar()

    # 这个不就是第一个卷积核的相位谱对应的频谱吗
    fig10 = plot.figure(10, figsize=(6, 6), dpi=300)
    plot.imshow(torch.abs(kernel_true_1_k), cmap='gray')
    plot.axis('off')
    # plot.title('phase')
    plot.colorbar()

    plot.show()

# 这个不就是第二个4f系统的卷积核的实际样子吗
kernel_true_2 = torch.zeros((scale * ke_pad_sc, scale * ke_pad_sc))
for r in range(scale):
    for c in range(scale):
        kernel_true_2[(ke_pad_sc // 2) + ke_pad_sc * r,
                      (ke_pad_sc // 2) + ke_pad_sc * c] \
            = 1

# 这个不就是他的频域的复值样子吗
kernel_true_2_k = torch.fft.fftshift(torch.fft.fft2(kernel_true_2), dim=(-2, -1))
kernel_true_2_k.imag *= -1

fig11 = plot.figure(11, figsize=(6, 6), dpi=300)
plot.imshow(kernel_true_2, cmap='gray')
plot.axis('off')
# plot.title('phase')
plot.colorbar()

# 这个不就是频谱吗
fig12 = plot.figure(12, figsize=(6, 6), dpi=300)
plot.imshow(torch.abs(kernel_true_2_k), cmap='gray')
plot.axis('off')
# plot.title('phase')
plot.colorbar()

# 这个不会是他的相位谱吗
fig13 = plot.figure(13, figsize=(6, 6), dpi=300)
plot.imshow(kernel_true_2_k.angle(), cmap='gray')
plot.axis('off')
# plot.title('phase')
plot.colorbar()

plot.show()

fig14 = plot.figure(14, figsize=(6, 6), dpi=300)
plot.imshow(result_true_tiny_1, cmap='gray')
plot.axis('off')
plot.colorbar()
plot.show()

# 示意图
kernel_true_3 = torch.zeros((scale * 11, scale * 11))
for r in range(scale):
    for c in range(scale):
        kernel_true_3[(11 // 2) + 11 * r,
                      (11 // 2) + 11 * c] \
            = 1

fig15 = plot.figure(15, figsize=(6, 6), dpi=300)
plot.imshow(kernel_true_3, cmap='gray')
plot.axis('off')
plot.colorbar()
plot.show()

pass
