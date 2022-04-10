import sys
import torch
from torch import nn
from torch.nn.functional import pad
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plot


def initialize_weight(weight):
    nn.init.kaiming_uniform_(weight, a=0, mode='fan_in', nonlinearity='relu')


def fourier_transform_right(inputs):
    outputs = torch.fft.fftshift(torch.fft.fft2(inputs, dim=(-2, -1)), dim=(-2, -1))

    return outputs


def fourier_transform_left(inputs):
    outputs = torch.fft.ifftshift(torch.fft.ifft2(inputs, dim=(-2, -1)), dim=(-2, -1))

    return outputs


class FourOptConv(nn.Module):
    def __init__(self, input_size=(100, 100), kernel_size=(3, 3), scale=3):
        """
        input_size：输入数据的尺寸
        kernel_size：每一个卷积核的尺寸，
        scale：卷积核行列平铺个数，比如 scale=3, kernel_size 3*3 -> 9*(3*3)
        """
        super(FourOptConv, self).__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.scale = scale
        self.input_pad_size = torch.tensor(input_size) + 2 * torch.div(torch.tensor(kernel_size),
                                                                       2, rounding_mode='floor')
        self.kernel_pad_size = torch.tensor(kernel_size) + 2 * torch.div(torch.tensor(kernel_size),
                                                                         2, rounding_mode='floor')
        self.kernel_fetch, self.kernel_mask, self.kernel_add = self.create_kernel()

    def create_kernel(self):
        kernel_zero_pad_list = []
        pad_size_list = [(self.kernel_size[0] // 2), (self.kernel_size[0] // 2),
                         (self.kernel_size[1] // 2), (self.kernel_size[1] // 2)]
        for _ in range(self.scale ** 2):
            kernel_valid = torch.rand(size=self.kernel_size)
            initialize_weight(kernel_valid)
            kernel_zero_pad = pad(kernel_valid, pad_size_list, 'constant', 0)
            kernel_zero_pad_list.append(kernel_zero_pad)
        mask_zero_pad = pad(torch.ones(size=self.kernel_size), pad_size_list, 'constant', 0)

        kernel_fetch = torch.zeros(size=list(self.scale * self.input_pad_size))
        kernel_mask = torch.zeros(size=list(self.scale * self.input_pad_size))
        kernel_add = torch.zeros(size=list(self.scale * self.input_pad_size))
        k = 0
        for i in range(self.scale):
            for j in range(self.scale):
                kernel_fetch[
                    self.input_pad_size[0] * i + (self.input_pad_size[0] // 2 - self.kernel_pad_size[0] // 2):
                    self.input_pad_size[0] * i + (self.input_pad_size[0] // 2 + self.kernel_pad_size[0] // 2) + 1,
                    self.input_pad_size[1] * j + (self.input_pad_size[1] // 2 - self.kernel_pad_size[1] // 2):
                    self.input_pad_size[1] * j + (self.input_pad_size[1] // 2 + self.kernel_pad_size[1] // 2) + 1] \
                    = kernel_zero_pad_list[k]
                k += 1
                kernel_mask[
                    self.input_pad_size[0] * i + (self.input_pad_size[0] // 2 - self.kernel_pad_size[0] // 2):
                    self.input_pad_size[0] * i + (self.input_pad_size[0] // 2 + self.kernel_pad_size[0] // 2) + 1,
                    self.input_pad_size[1] * j + (self.input_pad_size[1] // 2 - self.kernel_pad_size[1] // 2):
                    self.input_pad_size[1] * j + (self.input_pad_size[1] // 2 + self.kernel_pad_size[1] // 2) + 1] \
                    = mask_zero_pad
                kernel_add[
                    self.input_pad_size[0] * i + self.input_pad_size[0] // 2,
                    self.input_pad_size[1] * j + self.input_pad_size[1] // 2] \
                    = 1

        return nn.Parameter(kernel_fetch), kernel_mask, kernel_add

    def forward(self, inputs, mode='normal'):
        if mode == 'normal':
            outputs = fourier_transform_right(inputs) * fourier_transform_right(self.kernel_fetch * self.kernel_mask)
            outputs = outputs * fourier_transform_right(self.kernel_add)
        elif mode == 'down':
            _, _, h, w = inputs.shape
            inputs_k = fourier_transform_right(inputs)[h//2-h//4:h//2+h//4, w//2-w//4:w//2+w//4]
            outputs = inputs_k * fourier_transform_right(self.kernel_fetch * self.kernel_mask)
            outputs = outputs * fourier_transform_right(self.kernel_add)
        elif mode == 'up':
            _, _, h, w = inputs.shape
            inputs_k = pad(fourier_transform_right(inputs), [h, h, w, w], 'constant', 0.+0.j)
            outputs = inputs_k * fourier_transform_right(self.kernel_fetch * self.kernel_mask)
            outputs = outputs * fourier_transform_right(self.kernel_add)
        else:
            print('模式错误')
            sys.exit()

        return fourier_transform_left(outputs)


class FourOptNet(nn.Module):
    def __init__(self):
        super(FourOptNet, self).__init__()
        self.layer1 = FourOptConv((256, 256), (3, 3), 16)
        self.drop1 = nn.Dropout2d(p=0.3)
        self.layer2 = FourOptConv((64, 64), (3, 3), 32)
        self.drop2 = nn.Dropout2d(p=0.3)
        self.layer3 = FourOptConv((16, 16), (3, 3), 64)
        self.drop3 = nn.Dropout2d(p=0.3)
        self.layer4 = FourOptConv((64, 64), (3, 3), 32)
        self.drop4 = nn.Dropout2d(p=0.3)
        self.layer5 = FourOptConv((256, 256), (3, 3), 16)
        self.drop5 = nn.Dropout2d(p=0.3)
        self.layer6 = FourOptConv((256, 256), (3, 3), 1)
        self.drop6 = nn.Dropout2d(p=0.3)

    def forward(self, inputs):
        x = self.layer1(inputs, 'normal')
        x = self.layer2(x, 'down')
        x = self.layer3(x, 'down')
        x = self.layer4(x, 'up')
        x = self.layer5(x, 'up')
        x = self.layer6(x, 'normal')

        return x


if __name__ == '__main__':
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
    img_3 = img_1[256//2-256//4:256//2+256//4, 256//2-256//4:256//2+256//4]

    # 5. 傅立叶反变换后没有进行反褶，图片正常还原
    img_4 = torch.fft.ifft2(img_3)
    plot.figure(5)
    plot.imshow(torch.abs(img_4))

    # 6. 傅立叶反变换后进行反褶，图片发生颠倒
    img_5 = fourier_transform_left(img_3)
    plot.figure(6)
    plot.imshow(torch.abs(img_5))

    # 取输入数据在频域完成3中等价的卷积后的频谱的一半 -> 降低一倍空间分辨率
    img_6 = (img_1 * kernels_1)[256//2-256//4:256//2+256//4, 256//2-256//4:256//2+256//4]

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
    img_9[512//2-512//4:512//2+512//4, 512//2-512//4:512//2+512//4] = (img_1 * kernels_1)

    # 9. 傅立叶反变换后进行反褶
    img_10 = fourier_transform_left(img_9)
    plot.figure(9)
    plot.imshow(torch.abs(img_10))

    plot.show()
