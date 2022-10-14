import os
import tifffile
import cv2 as cv
import numpy as np


def present(e):
    if isinstance(e, BaseException):
        raise e
    else:
        raise ValueError(e)


class Noises:
    def __init__(self, image, mean, sigma, snr, rg=None):
        super().__init__()
        self.image = image
        self.mean = mean
        self.sigma = sigma
        self.snr = snr
        self.rg = rg

    # 添加高斯噪声
    def add_normal_noise(self):
        noise = (np.random.normal(self.mean, self.sigma, self.image.shape) * 255).astype(np.uint8)
        return noise

    # 添加泊松噪声
    def add_poisson_noise(self):
        noise = np.random.poisson(
            np.maximum(1, self.image * self.snr + 1).astype(int)).astype(np.uint8)
        return noise

    # 添加组合的高斯噪声和泊松噪声
    def add_normal_poisson_noise(self):
        noise = (self.add_normal_noise() + self.add_poisson_noise()).astype(
            np.uint8)
        return noise

    def add_random_noise(self):
        if self.rg is None:
            self.rg = np.random.uniform
        np.isscalar(self.snr) or present(
            ValueError("please give a snr value range"))
        np.isscalar(self.mean) or present(
            ValueError("please give a mean value range"))
        np.isscalar(self.sigma) or present(
            ValueError("please give a sigma value range"))
        all(v[0] <= v[1]
            for v in [self.snr, self.mean, self.sigma]) or present(
                ValueError(
                    "Lower bound is expected to be less than the upper bound"))
        all(v[0] >= 0 and v[1] >= 0
            for v in [self.snr, self.mean, self.sigma]) or present(
                ValueError(
                    "Noise's parameter is expected to be greater than 0"))

        self.mean = self.rg(*self.mean)
        self.sigma = self.rg(*self.sigma)
        self.snr = self.rg(*self.snr)
        self.image = (self.image - np.min(self.image)) / (np.max(self.image) +
                                                          np.min(self.image))

        noise = self.add_normal_poisson_noise()
        noise = np.maximum(0, noise)
        noise = np.minimum(1, noise)
        return noise


def cropper(image,
            crop_shape,
            anchor_shape=None,
            jitter=False,
            max_jitter=None,
            planes=None):
    half_crop_shape = tuple(_c // 2 for _c in crop_shape)
    if anchor_shape is None:
        anchor_shape = tuple(_i // 2 for _i in image.shape)
    assert all([_i - _c >= 0 for _c, _i in zip(half_crop_shape, anchor_shape)]), "Crop shape is bigger than image shape"

    if jitter:
        contrain_1 = tuple((_i - _c) // 4 for _c, _i in zip(half_crop_shape, anchor_shape))
        contrain_2 = tuple(c // 2 for c in half_crop_shape)
        if max_jitter is None:
            max_jitter = tuple([min(_ct2, _ct1) for _ct2, _ct1 in zip(contrain_1, contrain_2)])
        all([_i - _m >= 0 and _i + _m < 2 * _i for _m, _i in zip(max_jitter, anchor_shape)]) or present(ValueError("Jitter results in cropping outside border, please reduce max_jitter"))
        loc = tuple(_l - np.random.randint(-1 * max_jitter[_i], max_jitter[_i] + 1)
                    for _i, _l in enumerate(anchor_shape))
    else:
        loc = anchor_shape

    crop_image = image[loc[0] - half_crop_shape[0]:loc[0] + half_crop_shape[0],
                       loc[1] - half_crop_shape[1]:loc[1] + half_crop_shape[1],
                       loc[2] - half_crop_shape[2]:loc[2] + half_crop_shape[2]]

    if planes is not None:
        try:
            crop_image = crop_image[planes]
        except IndexError:
            present(ValueError("Plane does not exist"))

    return crop_image



path_1 = "/Users/WangHao/Desktop/Paper ImT UNet/data_0913/datasets/labels"
file_name_1 = os.listdir(path_1)
try:
    file_name_1.remove('.DS_Store')
except:
    pass
file_name_1 = sorted(file_name_1)

for i in range(len(file_name_1)):
    data_1 = tifffile.imread(os.path.join(path_1, file_name_1[i]))
    data_2 = cv.resize(data_1.transpose((1, 2, 0)), (64, 64)).transpose((2, 0, 1))
    data_2 = data_2 + Noises(data_2, 0, 0.006, 0.006).add_normal_poisson_noise()
    path_2 = f"/Users/WangHao/Desktop/Paper ImT UNet/data_0913/datasets/inputs/{file_name_1[i]}"
    tifffile.imsave(path_2, data_2)

print("运行完成")
