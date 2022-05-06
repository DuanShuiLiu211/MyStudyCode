from pickletools import optimize
import sys
import time
from sklearn.metrics import log_loss
import torch
from torch import pi
from torch import nn
from torch.nn.functional import pad
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plot
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


def initialize_weight(weight):
    """
    初始化模型的权重参数
    """
    nn.init.kaiming_uniform_(weight, a=0, mode='fan_in', nonlinearity='relu')


def fourier_transform_right(inputs):
    """
    对xy平面的空间域做傅立叶正变换并将低频移到中心
    """
    outputs = torch.fft.fftshift(torch.fft.fft2(inputs, dim=(-2, -1)), dim=(-2, -1))

    return outputs


def fourier_transform_left(inputs):
    """
    对xy平面的频域做傅立叶逆变换，还原xy平面空间域结果
    """
    outputs = torch.fft.ifftshift(torch.fft.ifft2(inputs, dim=(-2, -1)), dim=(-2, -1))

    return outputs


def phase_shifts_heightmap(heightmaps, wave_lengths=638e-9, refractive_index=1.518):
    """
    计算由高度图产生的相位偏移，自由传播中的空气折射率是1.000277
    """
    delta_n = refractive_index - 1.000277
    wave_vector = 2. * pi / wave_lengths
    phase = wave_vector * delta_n * heightmaps
    outputs = torch.exp(1.j * phase)

    return outputs


def pad_inputs(inputs, input_kernel_size, kernel_size, scale):
    """
    对输入数据四周有序补零
    """
    _, _, h, w = inputs.shape
    assert (h < input_kernel_size[0]) & (w < input_kernel_size[1])
    assert (kernel_size[0] == kernel_size[1]) & (scale % 2 == 1)
    dh = kernel_size[0] // 2
    dw = kernel_size[1] // 2
    outputs = pad(inputs, [int(input_kernel_size[0] * (scale // 2) + dh), int(input_kernel_size[1] * (scale // 2) + dw),
                           int(input_kernel_size[0] * (scale // 2) + dh), int(input_kernel_size[1] * (scale // 2) + dw)], value=0.)

    return outputs


def unpad_inputs(inputs, input_kernel_size, kernel_size, scale):
    """
    取出输入数据中心有效区域
    """
    _, _, h, w = inputs.size()
    dh = kernel_size[0] // 2
    dw = kernel_size[1] // 2
    outputs = inputs[:, :, input_kernel_size[0] * (scale // 2) + dh:input_kernel_size[0] * (scale // 2 + 1) - dh,
                     input_kernel_size[1] * (scale // 2) + dw:input_kernel_size[1] * (scale // 2 + 1) - dw]

    return outputs


class FourOptConv(nn.Module):
    def __init__(self, input_size=(100, 100), kernel_size=(3, 3), scale=3, weight_mode="kernel", visual=False):
        """
        input_size：输入数据的尺寸
        kernel_size：每一个卷积核的尺寸，
        scale：卷积核行列平铺个数，比如 scale=3, kernel_size 3*3 -> 9*(3*3)
        weight_mode：模型权重的类型，比如 kernel 与 plane
        """
        super(FourOptConv, self).__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.scale = scale
        self.weight_mode = weight_mode
        self.visual = visual
        self.input_kernel_size = np.array(input_size) + 2 * (np.array(kernel_size) // 2)
        self.kernel_add_softmax = torch.nn.Softmax(-1)(torch.rand(scale**2))
        self.phase_bn_input = torch.nn.BatchNorm2d(1)
        self.phase_bn_down = torch.nn.BatchNorm2d(1)
        self.phase_bn_up = torch.nn.BatchNorm2d(1)
        self.phase_bn_output = torch.nn.BatchNorm2d(1)

        # 0426 增加 plane
        if self.weight_mode == "kernel":
            self.kernel_fetch, kernel_fetch_mask, self.kernel_add, kernel_add_mask = self.create_kernel()
            self.register_buffer("kernel_fetch_mask", kernel_fetch_mask)
            self.register_buffer("kernel_add_mask", kernel_add_mask)
        elif self.weight_mode == "plane":
            self.plane = self.create_plane()
        else:
            print('权重模式错误')
            sys.exit()

    def create_kernel(self):
        kernel_list = []
        for _ in range(self.scale ** 2):
            kernel_valid = torch.rand(size=self.kernel_size)
            initialize_weight(kernel_valid)
            kernel_list.append(kernel_valid)

        kernel_fetch = torch.zeros(size=list(self.scale * self.input_kernel_size))
        kernel_fetch_mask = torch.zeros(size=list(self.scale * self.input_kernel_size))
        kernel_add = torch.zeros(size=list(self.scale * self.input_kernel_size))
        kernel_add_mask = torch.zeros(size=list(self.scale * self.input_kernel_size))

        k = 0
        for i in range(self.scale):
            for j in range(self.scale):
                """
                生成平铺核共有四种情况
                1. 输入尺寸为奇数而核尺寸为奇数则整体尺寸为奇数，核位于整体的中心
                2. 输入尺寸为奇数而核尺寸为偶数则整体尺寸为奇数，核位于整体的中心偏上左一个元素
                3. 输入尺寸为偶数而核尺寸为奇数则整体尺寸为偶数，核位于整体的中心偏下右一个元素
                4. 输入尺寸为偶数而核尺寸为偶数则整体尺寸为偶数，核位于整体的中心
                由于离散性，平铺的傅立叶卷积能够与平铺的PyTorch卷积的输出结果完全对应的是 1 与 2 整体尺寸为奇数，其中 3 与 4 整体尺寸为偶数有细微差别
                3. 平铺的PyTorch卷积的输出结果比平铺的傅立叶卷积的输出结果下与右多出一行与一列额外元素
                4. 平铺的PyTorch卷积的输出结果比平铺的傅立叶卷积的输出结果下与右多出一行与一列额外元素
                另外，如果核尺寸为奇数则每一个子区域将紧密相连，如果核尺寸为偶数则每一个子区域间将有一个元素间隔            
                但以上四种情况输出的有效元素位置都与输入有效元素位置相同
                """
                # 核尺寸为奇数时生成平铺卷积核
                if (self.kernel_size[0] % 2 == 1) & (self.kernel_size[1] % 2 == 1):
                    kernel_fetch[
                    self.input_kernel_size[0] * i + (self.input_kernel_size[0] // 2 - self.kernel_size[0] // 2):
                    self.input_kernel_size[0] * i + (self.input_kernel_size[0] // 2 + self.kernel_size[0] // 2 + 1),
                    self.input_kernel_size[1] * j + (self.input_kernel_size[1] // 2 - self.kernel_size[1] // 2):
                    self.input_kernel_size[1] * j + (self.input_kernel_size[1] // 2 + self.kernel_size[1] // 2 + 1)] \
                        = kernel_list[k]
                    kernel_fetch_mask[
                    self.input_kernel_size[0] * i + (self.input_kernel_size[0] // 2 - self.kernel_size[0] // 2):
                    self.input_kernel_size[0] * i + (self.input_kernel_size[0] // 2 + self.kernel_size[0] // 2 + 1),
                    self.input_kernel_size[1] * j + (self.input_kernel_size[1] // 2 - self.kernel_size[1] // 2):
                    self.input_kernel_size[1] * j + (self.input_kernel_size[1] // 2 + self.kernel_size[1] // 2 + 1)] \
                        = torch.ones(size=self.kernel_size)
                # 核尺寸为偶数时生成平铺卷积核
                elif (self.kernel_size[0] % 2 == 0) & (self.kernel_size[1] % 2 == 0):
                    kernel_fetch[
                    self.input_kernel_size[0] * i + (self.input_kernel_size[0] // 2 - self.kernel_size[0] // 2):
                    self.input_kernel_size[0] * i + (self.input_kernel_size[0] // 2 + self.kernel_size[0] // 2),
                    self.input_kernel_size[1] * j + (self.input_kernel_size[1] // 2 - self.kernel_size[1] // 2):
                    self.input_kernel_size[1] * j + (self.input_kernel_size[1] // 2 + self.kernel_size[1] // 2)] \
                        = kernel_list[k]
                    kernel_fetch_mask[
                    self.input_kernel_size[0] * i + (self.input_kernel_size[0] // 2 - self.kernel_size[0] // 2):
                    self.input_kernel_size[0] * i + (self.input_kernel_size[0] // 2 + self.kernel_size[0] // 2),
                    self.input_kernel_size[1] * j + (self.input_kernel_size[1] // 2 - self.kernel_size[1] // 2):
                    self.input_kernel_size[1] * j + (self.input_kernel_size[1] // 2 + self.kernel_size[1] // 2)] \
                        = torch.ones(size=self.kernel_size)
                else:
                    print("请保证卷积核有效区是正方形")
                    sys.exit()

                kernel_add[
                    self.input_kernel_size[0] * i + self.input_kernel_size[0] // 2,
                    self.input_kernel_size[1] * j + self.input_kernel_size[1] // 2] \
                    = self.kernel_add_softmax[k]
                kernel_add_mask[
                    self.input_kernel_size[0] * i + self.input_kernel_size[0] // 2,
                    self.input_kernel_size[1] * j + self.input_kernel_size[1] // 2] \
                    = torch.ones(1)
                k += 1

        return nn.Parameter(kernel_fetch), kernel_fetch_mask, nn.Parameter(kernel_add), kernel_add_mask

    def create_plane(self):
        plane = torch.rand(size=self.input_kernel_size)
        initialize_weight(plane)

        return nn.Parameter(plane)

    def forward(self, inputs, factor=2, sample_mode='input'):
        """现阶段实验可实现的是纯相位调制，即使用平铺卷积核的相谱制作纯相位的衍射板，相应的只能进行退化版的傅立叶卷积，计算时仅使用 angle()"""
        if self.weight_mode == "kernel":
            if sample_mode == 'input':
                # 构造输入
                inputs = pad_inputs(inputs, self.input_kernel_size, self.kernel_size, self.scale)
                if self.visual:
                    plot.figure(1)
                    plot.subplot(121)
                    plot.imshow(np.abs(inputs.detach().numpy().squeeze(0).squeeze(0)))
                    plot.colorbar()
                    plot.title("Amplitude")
                    plot.subplot(122)
                    plot.imshow(inputs.detach().angle().numpy().squeeze(0).squeeze(0))
                    plot.colorbar()
                    plot.title("Phase")
                    plot.show()

                # 特征提取的4f系统
                kernel_fetch_k = fourier_transform_right(self.kernel_fetch * self.kernel_fetch_mask)
                # phase_plane = torch.exp(1j * (2 * pi * kernel_fetch_k.angle()))
                phase_plane = torch.exp(1j * (2 * pi * torch.sigmoid(kernel_fetch_k.angle())))
                outputs = fourier_transform_right(inputs) * phase_plane
                outputs = fourier_transform_left(outputs)
                if self.visual:
                    plot.figure(2)
                    plot.subplot(121)
                    plot.imshow(np.abs(outputs.detach().numpy().squeeze(0).squeeze(0)))
                    plot.colorbar()
                    plot.title("Amplitude")
                    plot.subplot(122)
                    plot.imshow(outputs.detach().angle().numpy().squeeze(0).squeeze(0))
                    plot.colorbar()
                    plot.title("Phase")
                    plot.show()

                # 特征融合的4f系统
                kernel_add_k = fourier_transform_right(self.kernel_add * self.kernel_add_mask)
                # phase_plane = torch.exp(1j * (2 * pi * kernel_add_k.angle()))
                phase_plane = torch.exp(1j * (2 * pi * torch.sigmoid(kernel_add_k.angle())))
                outputs = fourier_transform_right(outputs) * phase_plane
                outputs = fourier_transform_left(outputs)
                if self.visual:
                    plot.figure(3)
                    plot.subplot(121)
                    plot.imshow(np.abs(outputs.detach().numpy().squeeze(0).squeeze(0)))
                    plot.colorbar()
                    plot.title("Amplitude")
                    plot.subplot(122)
                    plot.imshow(outputs.detach().angle().numpy().squeeze(0).squeeze(0))
                    plot.colorbar()
                    plot.title("Phase")
                    plot.show()

                # 构造输出
                outputs = unpad_inputs(outputs, self.input_kernel_size, self.kernel_size, self.scale)
                if self.visual:
                    plot.figure(4)
                    plot.subplot(121)
                    plot.imshow(np.abs(outputs.detach().numpy().squeeze(0).squeeze(0)))
                    plot.colorbar()
                    plot.title("Amplitude")
                    plot.subplot(122)
                    plot.imshow(outputs.detach().angle().numpy().squeeze(0).squeeze(0))
                    plot.colorbar()
                    plot.title("Phase")
                    plot.show()

            elif sample_mode == 'down':
                # 频域下采样，调整 NA 降低分辨率
                # _, _, h, w = inputs.shape
                # inputs_k = fourier_transform_right(inputs)[:, :, h // factor - h // (2 * factor):h // factor + h // (2 * factor) + 1,
                #            w // factor - w // (2 * factor):w // factor + w // (2 * factor) + 1]
                # inputs = torch.fft.ifft2(inputs_k)

                # 空域下采样，插值或池化
                inputs = torch.nn.UpsamplingNearest2d(scale_factor=0.5)(torch.abs(inputs))
                # inputs = torch.max_pool2d(torch.abs(inputs), kernel_size=(2, 2), stride=(2, 2))
                if self.visual:
                    plot.figure(1)
                    plot.subplot(121)
                    plot.imshow(np.abs(inputs.detach().numpy().squeeze(0).squeeze(0)))
                    plot.colorbar()
                    plot.title("Amplitude")
                    plot.subplot(122)
                    plot.imshow(np.abs(inputs.detach().angle().numpy().squeeze(0).squeeze(0)))
                    plot.colorbar()
                    plot.title("Phase")
                    plot.show()

                # 构造输入
                inputs = pad_inputs(inputs, self.input_kernel_size, self.kernel_size, self.scale)
                if self.visual:
                    plot.figure(2)
                    plot.subplot(121)
                    plot.imshow(np.abs(inputs.detach().numpy().squeeze(0).squeeze(0)))
                    plot.colorbar()
                    plot.title("Amplitude")
                    plot.subplot(122)
                    plot.imshow(np.abs(inputs.detach().angle().numpy().squeeze(0).squeeze(0)))
                    plot.colorbar()
                    plot.title("Phase")
                    plot.show()

                # 特征提取的4f系统
                kernel_fetch_k = fourier_transform_right(self.kernel_fetch * self.kernel_fetch_mask)
                # phase_plane = torch.exp(1j * (2 * pi * kernel_fetch_k.angle()))
                phase_plane = torch.exp(1j * (2 * pi * torch.sigmoid(kernel_fetch_k.angle())))
                outputs = fourier_transform_right(inputs) * phase_plane
                outputs = fourier_transform_left(outputs)
                if self.visual:
                    plot.figure(3)
                    plot.subplot(121)
                    plot.imshow(np.abs(outputs.detach().numpy().squeeze(0).squeeze(0)))
                    plot.colorbar()
                    plot.title("Amplitude")
                    plot.subplot(122)
                    plot.imshow(outputs.detach().angle().numpy().squeeze(0).squeeze(0))
                    plot.colorbar()
                    plot.title("Phase")
                    plot.show()

                # 特征融合的4f系统
                kernel_add_k = fourier_transform_right(self.kernel_add * self.kernel_add_mask)
                # phase_plane = torch.exp(1j * (2 * pi * kernel_add_k.angle()))
                phase_plane = torch.exp(1j * (2 * pi * torch.sigmoid(kernel_add_k.angle())))
                outputs = fourier_transform_right(outputs) * phase_plane
                outputs = fourier_transform_left(outputs)
                if self.visual:
                    plot.figure(4)
                    plot.subplot(121)
                    plot.imshow(np.abs(outputs.detach().numpy().squeeze(0).squeeze(0)))
                    plot.colorbar()
                    plot.title("Amplitude")
                    plot.subplot(122)
                    plot.imshow(outputs.detach().angle().numpy().squeeze(0).squeeze(0))
                    plot.colorbar()
                    plot.title("Phase")
                    plot.show()

                # 构造输出
                outputs = unpad_inputs(outputs, self.input_kernel_size, self.kernel_size, self.scale)
                if self.visual:
                    plot.figure(5)
                    plot.subplot(121)
                    plot.imshow(np.abs(outputs.detach().numpy().squeeze(0).squeeze(0)))
                    plot.colorbar()
                    plot.title("Amplitude")
                    plot.subplot(122)
                    plot.imshow(outputs.detach().angle().numpy().squeeze(0).squeeze(0))
                    plot.colorbar()
                    plot.title("Phase")
                    plot.show()

            elif sample_mode == 'up':
                # 空域上采样
                _, _, h, w = inputs.shape
                if (h % 2 == 0) & (w % 2 == 0):
                    inputs = torch.nn.UpsamplingNearest2d(size=(h * factor, w * factor))(torch.abs(inputs))
                elif (h % 2 == 1) & (w % 2 == 1):
                    inputs = torch.nn.UpsamplingNearest2d(size=((h + 1) * factor - 1, (w + 1) * factor - 1))(torch.abs(inputs))
                else:
                    print("保证输入为方形")
                    sys.exit()
                if self.visual:
                    plot.figure(1)
                    plot.subplot(121)
                    plot.imshow(np.abs(inputs.detach().numpy().squeeze(0).squeeze(0)))
                    plot.colorbar()
                    plot.title("Amplitude")
                    plot.subplot(122)
                    plot.imshow(np.abs(inputs.detach().angle().numpy().squeeze(0).squeeze(0)))
                    plot.colorbar()
                    plot.title("Phase")
                    plot.show()

                # 构造输入
                inputs = pad_inputs(inputs, self.input_kernel_size, self.kernel_size, self.scale)
                if self.visual:
                    plot.figure(2)
                    plot.subplot(121)
                    plot.imshow(np.abs(inputs.detach().numpy().squeeze(0).squeeze(0)))
                    plot.colorbar()
                    plot.title("Amplitude")
                    plot.subplot(122)
                    plot.imshow(np.abs(inputs.detach().angle().numpy().squeeze(0).squeeze(0)))
                    plot.colorbar()
                    plot.title("Phase")
                    plot.show()

                # 特征提取的4f系统
                kernel_fetch_k = fourier_transform_right(self.kernel_fetch * self.kernel_fetch_mask)
                # phase_plane = torch.exp(1j * (2 * pi * kernel_fetch_k.angle()))
                phase_plane = torch.exp(1j * (2 * pi * torch.sigmoid(kernel_fetch_k.angle())))
                outputs = fourier_transform_right(inputs) * phase_plane
                outputs = fourier_transform_left(outputs)
                if self.visual:
                    plot.figure(3)
                    plot.subplot(121)
                    plot.imshow(np.abs(outputs.detach().numpy().squeeze(0).squeeze(0)))
                    plot.colorbar()
                    plot.title("Amplitude")
                    plot.subplot(122)
                    plot.imshow(outputs.detach().angle().numpy().squeeze(0).squeeze(0))
                    plot.colorbar()
                    plot.title("Phase")
                    plot.show()

                # 特征融合的4f系统
                kernel_add_k = fourier_transform_right(self.kernel_add * self.kernel_add_mask)
                # phase_plane = torch.exp(1j * (2 * pi * kernel_add_k.angle()))
                phase_plane = torch.exp(1j * (2 * pi * torch.sigmoid(kernel_add_k.angle())))
                outputs = fourier_transform_right(outputs) * phase_plane
                outputs = fourier_transform_left(outputs)
                if self.visual:
                    plot.figure(4)
                    plot.subplot(121)
                    plot.imshow(np.abs(outputs.detach().numpy().squeeze(0).squeeze(0)))
                    plot.colorbar()
                    plot.title("Amplitude")
                    plot.subplot(122)
                    plot.imshow(outputs.detach().angle().numpy().squeeze(0).squeeze(0))
                    plot.colorbar()
                    plot.title("Phase")
                    plot.show()

                # 构造输出
                outputs = unpad_inputs(outputs, self.input_kernel_size, self.kernel_size, self.scale)
                if self.visual:
                    plot.figure(5)
                    plot.subplot(121)
                    plot.imshow(np.abs(outputs.detach().numpy().squeeze(0).squeeze(0)))
                    plot.colorbar()
                    plot.title("Amplitude")
                    plot.subplot(122)
                    plot.imshow(outputs.detach().angle().numpy().squeeze(0).squeeze(0))
                    plot.colorbar()
                    plot.title("Phase")
                    plot.show()

            elif sample_mode == 'output':
                # 构造输入
                inputs = pad_inputs(inputs, self.input_kernel_size, self.kernel_size, self.scale)
                if self.visual:
                    plot.figure(1)
                    plot.subplot(121)
                    plot.imshow(np.abs(inputs.detach().numpy().squeeze(0).squeeze(0)))
                    plot.colorbar()
                    plot.title("Amplitude")
                    plot.subplot(122)
                    plot.imshow(inputs.detach().angle().numpy().squeeze(0).squeeze(0))
                    plot.colorbar()
                    plot.title("Phase")
                    plot.show()

                # 特征提取的4f系统
                kernel_fetch_k = fourier_transform_right(self.kernel_fetch * self.kernel_fetch_mask)
                # phase_plane = torch.exp(1j * (2 * pi * kernel_fetch_k.angle()))
                phase_plane = torch.exp(1j * (2 * pi * torch.sigmoid(kernel_fetch_k.angle())))
                outputs = fourier_transform_right(inputs) * phase_plane
                outputs = fourier_transform_left(outputs)
                if self.visual:
                    plot.figure(2)
                    plot.subplot(121)
                    plot.imshow(np.abs(outputs.detach().numpy().squeeze(0).squeeze(0)))
                    plot.colorbar()
                    plot.title("Amplitude")
                    plot.subplot(122)
                    plot.imshow(outputs.detach().angle().numpy().squeeze(0).squeeze(0))
                    plot.colorbar()
                    plot.title("Phase")
                    plot.show()

                # 特征融合的4f系统
                kernel_add_k = fourier_transform_right(self.kernel_add * self.kernel_add_mask)
                # phase_plane = torch.exp(1j * (2 * pi * kernel_add_k.angle()))
                phase_plane = torch.exp(1j * (2 * pi * torch.sigmoid(kernel_add_k.angle())))
                outputs = fourier_transform_right(outputs) * phase_plane
                outputs = fourier_transform_left(outputs)
                if self.visual:
                    plot.figure(3)
                    plot.subplot(121)
                    plot.imshow(np.abs(outputs.detach().numpy().squeeze(0).squeeze(0)))
                    plot.colorbar()
                    plot.title("Amplitude")
                    plot.subplot(122)
                    plot.imshow(outputs.detach().angle().numpy().squeeze(0).squeeze(0))
                    plot.colorbar()
                    plot.title("Phase")
                    plot.show()

                # 构造输出
                outputs = unpad_inputs(outputs, self.input_kernel_size, self.kernel_size, self.scale)
                if self.visual:
                    plot.figure(4)
                    plot.subplot(121)
                    plot.imshow(np.abs(outputs.detach().numpy().squeeze(0).squeeze(0)))
                    plot.colorbar()
                    plot.title("Amplitude")
                    plot.subplot(122)
                    plot.imshow(outputs.detach().angle().numpy().squeeze(0).squeeze(0))
                    plot.colorbar()
                    plot.title("Phase")
                    plot.show()

            else:
                print('采样模式错误')
                sys.exit()
        elif self.weight_mode == "plane":
            if sample_mode == 'normal':
                outputs = fourier_transform_right(inputs) * phase_shifts_heightmap(self.plane)
            elif sample_mode == 'down':
                _, _, h, w = inputs.shape
                inputs_k = fourier_transform_right(inputs)[:, :, h // factor - h // (2 * factor):h // factor + h // (2 * factor),
                           w // factor - w // (2 * factor):w // factor + w // (2 * factor)]
                outputs = inputs_k * phase_shifts_heightmap(self.plane)
            elif sample_mode == 'up':
                inputs_k = fourier_transform_right(torch.nn.UpsamplingNearest2d(scale_factor=factor)(inputs))
                outputs = inputs_k * phase_shifts_heightmap(self.plane)
            else:
                print('采样模式设置错误')
                sys.exit()
        else:
            print('权重模式设置错误')
            sys.exit()

        return outputs


class OptConvNet(nn.Module):
    def __init__(self, input_size=(256, 256), output_size=(256, 256), down_factor=4, up_factor=4, visual=False):
        super(OptConvNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.down_factor = down_factor
        self.up_factor = up_factor
        self.visual = visual
        self.layer1_size = np.array(self.input_size)
        if (self.layer1_size[0] % 2 == 0) & (self.layer1_size[0] % 2 == 0):
            self.layer2_size = self.layer1_size // self.down_factor
            self.layer3_size = self.layer2_size // self.down_factor
            self.layer4_size = self.layer3_size * self.up_factor
            self.layer5_size = self.layer4_size * self.up_factor
            self.layer6_size = np.array(self.output_size)
        elif (self.layer1_size[0] % 2 == 1) & (self.layer1_size[0] % 2 == 1):
            self.layer2_size = self.layer1_size // self.down_factor
            self.layer3_size = self.layer2_size // self.down_factor
            self.layer4_size = (self.layer3_size + 1) * self.up_factor - 1
            self.layer5_size = (self.layer4_size + 1) * self.up_factor - 1
            self.layer6_size = np.array(self.output_size)
        else:
            print("保证输入层为方")
            sys.exit()
        self.layer1 = FourOptConv(self.layer1_size, (3, 3), 5, "kernel", self.visual)
        self.drop1 = nn.Dropout2d(p=0.3)
        self.layer2 = FourOptConv(self.layer2_size, (3, 3), 9, "kernel", self.visual)
        self.drop2 = nn.Dropout2d(p=0.3)
        self.layer3 = FourOptConv(self.layer3_size, (3, 3), 17, "kernel", self.visual)
        self.drop3 = nn.Dropout2d(p=0.3)
        self.layer4 = FourOptConv(self.layer4_size, (3, 3), 9, "kernel", self.visual)
        self.drop4 = nn.Dropout2d(p=0.3)
        self.layer5 = FourOptConv(self.layer5_size, (3, 3), 5, "kernel", self.visual)
        self.drop5 = nn.Dropout2d(p=0.3)
        self.layer6 = FourOptConv(self.layer6_size, (3, 3), 1, "kernel", self.visual)
        self.drop6 = nn.Dropout2d(p=0.3)
        self.bn_output = torch.nn.BatchNorm2d(1)

    def forward(self, inputs):
        x = self.layer1(inputs, sample_mode='input')
        # plot.figure(1)
        # plot.imshow(inputs.detach().numpy().squeeze(0).squeeze(0))
        # plot.colorbar()
        # plot.title("Amplitude")
        # plot.show()
        x = self.layer2(x, self.down_factor, 'down')
        x = self.layer3(x, self.down_factor, 'down')
        x = self.layer4(x, self.up_factor, 'up')
        x = self.layer5(x, self.up_factor, 'up')
        x = self.layer6(x, sample_mode='output')
        # outputs = self.bn_output(torch.abs(x))
        outputs = torch.sigmoid(torch.abs(x))
        # plot.figure(2)
        # plot.imshow(outputs.detach().numpy().squeeze(0).squeeze(0))
        # plot.colorbar()
        # plot.title("Amplitude")
        # plot.show()
        return outputs


if __name__ == '__main__':
    """
    path = r'/Users/WangHao/工作/纳米光子中心/全光相关/实验-0303/0303.png'
    img = Image.open(path).convert('L')
    img_01 = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])(img).squeeze(0)
    img_02 = transforms.Compose([transforms.ToTensor(), transforms.Resize((255, 255))])(img).squeeze(0)
    # 1. 读入图片
    plot.figure(1)
    plot.imshow(img_01)
    plot.figure(2)
    plot.imshow(img_02)

    # # 2. 生成卷积核
    # kernels = torch.rand((3, 3))
    # plot.figure(2)
    # plot.imshow(kernels)
    #
    # # 3. 进行卷积神经网络的空域卷积运算（实质是相关运算）
    # img_conv = torch.conv2d(img_01.unsqueeze(0).unsqueeze(0), kernels.unsqueeze(0).unsqueeze(0),
    #                         padding=kernels.shape[-1] // 2).squeeze(0).squeeze(0)
    # plot.figure(3)
    # plot.imshow(img_conv)

    # 对输入数据与卷积核做傅立叶变换并进行反褶将低频移至中心
    img_11 = fourier_transform_right(img_01)
    # kernels_11 = fourier_transform_right(pad(kernels, [127, 126, 127, 126], 'constant', 0))

    img_12 = fourier_transform_right(img_02)
    # kernels_12 = fourier_transform_right(pad(kernels, [126, 126, 126, 126], 'constant', 0))
    # 4. 在频域完成3中等价的卷积后反变换回去并进行反褶
    # img_21 = fourier_transform_left(img_11 * kernels_11)
    # plot.figure(4)
    # plot.imshow(torch.abs(img_21))

    # img_22 = fourier_transform_left(img_12 * kernels_12)
    # plot.figure(5)
    # plot.imshow(torch.abs(img_22))
    # 取输入数据频谱的一半 -> 降低一倍空间分辨率
    img_31 = img_11[256 // 2 - 256 // 4:256 // 2 + 256 // 4, 256 // 2 - 256 // 4:256 // 2 + 256 // 4]
    img_32 = img_12[255 // 2 - 255 // 4:255 // 2 + 255 // 4, 255 // 2 - 255 // 4:255 // 2 + 255 // 4]

    # 5. 傅立叶反变换后没有进行反褶，图片正常还原
    img_41 = torch.fft.ifft2(img_31)
    plot.figure(5)
    plot.imshow(torch.abs(img_41))

    img_42 = torch.fft.ifft2(img_32)
    plot.figure(6)
    plot.imshow(torch.abs(img_42))

    img_43 = torch.nn.UpsamplingNearest2d(scale_factor=0.5)(torch.abs(img_01.unsqueeze(0).unsqueeze(0))).squeeze(0).squeeze(0)
    plot.figure(7)
    plot.imshow(torch.abs(img_43))

    img_44 = torch.nn.UpsamplingNearest2d(scale_factor=0.5)(torch.abs(img_02.unsqueeze(0).unsqueeze(0))).squeeze(0).squeeze(0)
    plot.figure(8)
    plot.imshow(torch.abs(img_44))
    # # 6. 傅立叶反变换后进行反褶，图片发生颠倒
    # img_5 = fourier_transform_left(img_13)
    # plot.figure(6)
    # plot.imshow(torch.abs(img_5))

    # # 取输入数据在频域完成3中等价的卷积后的频谱的一半 -> 降低一倍空间分辨率
    # img_6 = (img_1 * kernels_1)[255 // 2 - 255 // 4:255 // 2 + 255 // 4, 255 // 2 - 255 // 4:255 // 2 + 255 // 4]
    #
    # # 7. 傅立叶反变换后不进行反褶，图片发生颠倒
    # img_7 = torch.fft.ifft2(img_6)
    # plot.figure(7)
    # plot.imshow(torch.abs(img_7))
    #
    # # 8. 傅立叶反变换后进行反褶，图片正常还原
    # img_8 = fourier_transform_left(img_6)
    # plot.figure(8)
    # plot.imshow(torch.abs(img_8))

    # # 频域补一圈一倍的零 -> 提高一倍空间分辨率
    # img_9 = torch.zeros((512, 512), dtype=torch.complex64)
    # img_9[512 // 2 - 512 // 4:512 // 2 + 512 // 4, 512 // 2 - 512 // 4:512 // 2 + 512 // 4] = (img_1 * kernels_1)

    # # 9. 傅立叶反变换后进行反褶
    # img_10 = fourier_transform_left(img_9)
    # plot.figure(9)
    # plot.imshow(torch.abs(img_10))

    plot.show()
    """

    """
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.layer1 = nn.Parameter(torch.tensor(1.))
                self.register_parameter("layer11", self.layer1)
                self.layer1.data = torch.tensor(2.)

                self.layer2 = torch.tensor(1.)
                self.register_buffer("layer22", self.layer2)
                self.register_buffer("layer222", self.layer2)
                self.layer2 += 1


        nets = Net()
        for parameters in nets.parameters():
            print(parameters, id(parameters))
        for parameters in nets.named_parameters():
            print(parameters, id(parameters))
        for k, v in nets._parameters.items():
            print(k, v, id(v))
        for buffers in nets.buffers():
            print(buffers, id(buffers))
        for buffers in nets.named_buffers():
            print(buffers, id(buffers))
        for k, v in nets._buffers.items():
            print(k, v, id(v))
        """

    models1 = OptConvNet(input_size=(255, 255), output_size=(255, 255), down_factor=2, up_factor=2, visual=False)
    adam_optimize = optim.Adam(models1.parameters(), lr=1e-4, weight_decay=1e-5)
    ce_loss =nn.BCELoss()
    visual_writer = SummaryWriter('visualization')
    path = r'/Users/WangHao/工作/纳米光子中心/全光相关/实验-0303/0303.png'
    inputs1 = Image.open(path).convert('L')
    inputs1 = transforms.Compose([transforms.ToTensor(), transforms.Resize((255, 255))])(inputs1).unsqueeze(0)
    epoch = 1000
    for idx in range(epoch):
        models1.train()
        adam_optimize.zero_grad()
        outputs1 = models1(inputs1)
        loss_result = ce_loss(outputs1, inputs1)
        loss_result.backward()
        adam_optimize.step()
        if idx == 0:
            visual_writer.add_graph(models1, inputs1)
        visual_writer.add_scalars('trainloss', {'ce_loss': loss_result.data.item(), }, idx)
        visual_writer.add_image('0303_1', inputs1.squeeze(0), idx)
        visual_writer.add_image('0303_2', outputs1.squeeze(0), idx)
    
    visual_writer.close()

    for parameters in models1.named_parameters():
        plot.figure(parameters[0])
        plot.imshow(parameters[1].detach().numpy())
        plot.colorbar()
    plot.show()

    pass
