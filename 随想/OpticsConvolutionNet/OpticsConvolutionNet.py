import sys
import torch
from torch import pi
from torch import nn
from torch.nn.functional import pad
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plot


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


def pad_inputs(inputs, pad_size, scale):
    """
    对输入数据四周有序补零
    """
    _, _, h, w = inputs.size()
    dh = (pad_size[0] - h) // 2
    dw = (pad_size[1] - w) // 2
    if scale % 2 == 0:
        outputs = pad(inputs, [int(pad_size[0] * (scale // 2 - 1) + dh), int(pad_size[1] * (scale // 2) + dw),
                               int(pad_size[0] * (scale // 2 - 1) + dh), int(pad_size[1] * (scale // 2) + dw)], value=0)
    else:
        outputs = pad(inputs, [int(pad_size[0] * (scale // 2) + dh), int(pad_size[1] * (scale // 2) + dw),
                               int(pad_size[0] * (scale // 2) + dh), int(pad_size[1] * (scale // 2) + dw)], value=0)
    return outputs


# TODO: 0427 未完成
def unpad_inputs(inputs, pad_size, scale):
    """
    取出输入数据未补零的部分
    """
    _, _, h, w = inputs.size()
    dh = (pad_size[0] - h) // 2
    dw = (pad_size[1] - w) // 2
    if scale % 2 == 0:
        outputs = pad(inputs, [int(pad_size[0] * (scale // 2 - 1) + dh), int(pad_size[1] * (scale // 2) + dw),
                               int(pad_size[0] * (scale // 2 - 1) + dh), int(pad_size[1] * (scale // 2) + dw)], value=0)
    else:
        outputs = pad(inputs, [int(pad_size[0] * (scale // 2) + dh), int(pad_size[1] * (scale // 2) + dw),
                               int(pad_size[0] * (scale // 2) + dh), int(pad_size[1] * (scale // 2) + dw)], value=0)
    return outputs


class FourOptConv(nn.Module):
    def __init__(self, input_size=(100, 100), kernel_size=(3, 3), scale=3, weight_mode="kernel"):
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
        self.input_kernel_size = np.array(input_size) + 2 * (np.array(kernel_size) // 2)

        # 0426
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
                kernel_fetch[
                self.input_kernel_size[0] * i + (self.input_kernel_size[0] // 2 - self.kernel_size[0] // 2):
                self.input_kernel_size[0] * i + (self.input_kernel_size[0] // 2 + self.kernel_size[0] // 2 + 1),
                self.input_kernel_size[1] * j + (self.input_kernel_size[1] // 2 - self.kernel_size[1] // 2):
                self.input_kernel_size[1] * j + (self.input_kernel_size[1] // 2 + self.kernel_size[1] // 2 + 1)] \
                    = kernel_list[k]
                k += 1
                kernel_fetch_mask[
                self.input_kernel_size[0] * i + (self.input_kernel_size[0] // 2 - self.kernel_size[0] // 2):
                self.input_kernel_size[0] * i + (self.input_kernel_size[0] // 2 + self.kernel_size[0] // 2 + 1),
                self.input_kernel_size[1] * j + (self.input_kernel_size[1] // 2 - self.kernel_size[1] // 2):
                self.input_kernel_size[1] * j + (self.input_kernel_size[1] // 2 + self.kernel_size[1] // 2 + 1)] \
                    = torch.ones(size=self.kernel_size)
                kernel_add[
                    self.input_kernel_size[0] * i + self.input_kernel_size[0] // 2,
                    self.input_kernel_size[1] * j + self.input_kernel_size[1] // 2] \
                    = 1
                kernel_add_mask[
                    self.input_kernel_size[0] * i + self.input_kernel_size[0] // 2,
                    self.input_kernel_size[1] * j + self.input_kernel_size[1] // 2] \
                    = 1

        return nn.Parameter(kernel_fetch), kernel_fetch_mask, nn.Parameter(kernel_add), kernel_add_mask

    def create_plane(self):
        plane = torch.rand(size=self.input_kernel_size)
        initialize_weight(plane)

        return nn.Parameter(plane)

    def forward(self, inputs, factor=2, sample_mode='normal'):
        inputs = pad_inputs(inputs, self.input_kernel_size, self.scale)
        if self.weight_mode == "kernel":
            if sample_mode == 'normal':
                outputs = fourier_transform_right(inputs) * fourier_transform_right(self.kernel_fetch * self.kernel_fetch_mask)
                outputs = outputs * fourier_transform_right(self.kernel_add * self.kernel_add_mask)
                outputs = fourier_transform_left(outputs)
                outputs = 1
            elif sample_mode == 'down':
                _, _, h, w = inputs.shape
                inputs_k = fourier_transform_right(inputs)[:, :, h // factor - h // (2*factor):h // factor + h // (2*factor),
                                                           w // factor - w // (2*factor):w // factor + w // (2*factor)]
                outputs = inputs_k * fourier_transform_right(self.kernel_fetch * self.kernel_fetch_mask)
                outputs = outputs * fourier_transform_right(self.kernel_add * self.kernel_add_mask)
            elif sample_mode == 'up':
                inputs_k = fourier_transform_right(torch.nn.UpsamplingNearest2d(scale_factor=factor)(inputs))
                outputs = inputs_k * fourier_transform_right(self.kernel_fetch * self.kernel_fetch_mask)
                outputs = outputs * fourier_transform_right(self.kernel_add * self.kernel_add_mask)
            else:
                print('采样模式错误')
                sys.exit()
        elif self.weight_mode == "plane":
            if sample_mode == 'normal':
                outputs = fourier_transform_right(inputs) * phase_shifts_heightmap(self.plane)
            elif sample_mode == 'down':
                _, _, h, w = inputs.shape
                inputs_k = fourier_transform_right(inputs)[:, :, h // factor - h // (2*factor):h // factor + h // (2*factor),
                                                           w // factor - w // (2*factor):w // factor + w // (2*factor)]
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
    def __init__(self, input_size=(256, 256), output_size=(256, 256), down_factor=4, up_factor=4):
        super(OptConvNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.down_factor = down_factor
        self.up_factor = up_factor
        self.layer1_size = np.array(self.input_size)
        self.layer2_size = self.layer1_size // self.down_factor
        self.layer3_size = self.layer2_size // self.down_factor
        self.layer4_size = self.layer3_size * self.up_factor
        self.layer5_size = self.layer4_size * self.up_factor
        self.layer6_size = np.array(self.output_size)
        self.layer1 = FourOptConv(self.layer1_size, (3, 3), 16)
        self.drop1 = nn.Dropout2d(p=0.3)
        self.layer2 = FourOptConv(self.layer2_size, (3, 3), 32)
        self.drop2 = nn.Dropout2d(p=0.3)
        self.layer3 = FourOptConv(self.layer3_size, (3, 3), 64)
        self.drop3 = nn.Dropout2d(p=0.3)
        self.layer4 = FourOptConv(self.layer4_size, (3, 3), 32)
        self.drop4 = nn.Dropout2d(p=0.3)
        self.layer5 = FourOptConv(self.layer5_size, (3, 3), 16)
        self.drop5 = nn.Dropout2d(p=0.3)
        self.layer6 = FourOptConv(self.layer6_size, (3, 3), 1)
        self.drop6 = nn.Dropout2d(p=0.3)

    def forward(self, inputs):
        x = self.layer1(inputs, 'normal')
        x = self.layer2(x, self.down_factor, 'down')
        x = self.layer3(x, self.down_factor, 'down')
        x = self.layer4(x, self.up_factor, 'up')
        x = self.layer5(x, self.up_factor, 'up')
        x = self.layer6(x, 'normal')

        return x


if __name__ == '__main__':
    """
    path = r'/Users/WangHao/工作/纳米光子中心/全光相关/实验-0303/0303.png'
    img = Image.open(path).convert('L')
    img = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])(img).squeeze(0)
    # 1. 读入图片
    plot.figure(1)
    plot.imshow(img)

    # 2. 生成卷积核
    kernels = torch.rand((3, 3))
    plot.figure(2)
    plot.imshow(kernels)

    # 3. 进行卷积神经网络的空域卷积运算（实质是相关运算）
    img_conv = torch.conv2d(img.unsqueeze(0).unsqueeze(0), kernels.unsqueeze(0).unsqueeze(0),
                            padding=kernels.shape[-1] // 2).squeeze(0).squeeze(0)
    plot.figure(3)
    plot.imshow(img_conv)

    # 对输入数据与卷积核做傅立叶变换并进行反褶将低频移至中心
    img_1 = fourier_transform_right(img)
    kernels_1 = fourier_transform_right(pad(kernels, [127, 126, 127, 126], 'constant', 0))

    # 4. 在频域完成3中等价的卷积后反变换回去并进行反褶
    img_2 = fourier_transform_left(img_1 * kernels_1)
    plot.figure(4)
    plot.imshow(torch.abs(img_2))

    # 取输入数据频谱的一半 -> 降低一倍空间分辨率
    img_3 = img_1[256 // 2 - 256 // 4:256 // 2 + 256 // 4, 256 // 2 - 256 // 4:256 // 2 + 256 // 4]

    # 5. 傅立叶反变换后没有进行反褶，图片正常还原
    img_4 = torch.fft.ifft2(img_3)
    plot.figure(5)
    plot.imshow(torch.abs(img_4))

    # 6. 傅立叶反变换后进行反褶，图片发生颠倒
    img_5 = fourier_transform_left(img_3)
    plot.figure(6)
    plot.imshow(torch.abs(img_5))

    # 取输入数据在频域完成3中等价的卷积后的频谱的一半 -> 降低一倍空间分辨率
    img_6 = (img_1 * kernels_1)[256 // 2 - 256 // 4:256 // 2 + 256 // 4, 256 // 2 - 256 // 4:256 // 2 + 256 // 4]

    # 7. 傅立叶反变换后不进行反褶，图片发生颠倒
    img_7 = torch.fft.ifft2(img_6)
    plot.figure(7)
    plot.imshow(torch.abs(img_7))

    # 8. 傅立叶反变换后进行反褶，图片正常还原
    img_8 = fourier_transform_left(img_6)
    plot.figure(8)
    plot.imshow(torch.abs(img_8))

    # 频域补一圈一倍的零 -> 提高一倍空间分辨率
    img_9 = torch.zeros((512, 512), dtype=torch.complex64)
    img_9[512 // 2 - 512 // 4:512 // 2 + 512 // 4, 512 // 2 - 512 // 4:512 // 2 + 512 // 4] = (img_1 * kernels_1)

    # 9. 傅立叶反变换后进行反褶
    img_10 = fourier_transform_left(img_9)
    plot.figure(9)
    plot.imshow(torch.abs(img_10))

    plot.show()
    """
    models = OptConvNet()
    inputs1 = torch.rand((2, 1, 256, 256))
    outputs1 = models(inputs1)
    for parameters in models.named_parameters():
        print(parameters)


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
    pass
