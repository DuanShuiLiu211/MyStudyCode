import sys

import cv2
import matplotlib.pyplot as plot
import numpy as np


def ContrastAlgorithm(rgb_img, contrast=0.5, threshold=0.5):
    img = rgb_img * 1.0
    img_out = img

    # 增量等于1，按灰度阈值最多调整成八种颜色：
    # 黑、红、绿、蓝、黄(255,255,0)、品红(255,0,255)、青(0,255,255)、白
    if contrast == 1:
        # newRGB = RGB >= Threshold? 255 : 0
        mask_1 = img >= threshold * 255.0
        rgb1 = 255.0
        rgb2 = 0
        img_out = rgb1 * mask_1 + rgb2 * (1 - mask_1)

    # 增量大于0小于1
    elif contrast >= 0:
        alpha = 1 - contrast
        alpha = 1 / alpha - 1
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - threshold * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - threshold * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - threshold * 255.0) * alpha

    # 增量小于0
    else:
        alpha = contrast
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - threshold * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - threshold * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - threshold * 255.0) * alpha

    img_out = img_out / 255.0
    return img_out


if __name__ == "__main__":
    """
    对比度调整算法
    主要是对RGB空间进行调整。设定合适的RGB阈值，并在此阈值基础上计算出合适的调整系数进行对比度调整。
    """
    path = "./resource/fruit.bmp"
    contrast = 0.5  # 范围：-1至1
    threshold = 0.5  # 范围：0至1
    len = len(sys.argv)
    if len >= 2:
        path = sys.argv[1]
        if len >= 3:
            contrast = float(sys.argv[2])
            if len >= 4:
                threshold = float(sys.argv[3])

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_new = ContrastAlgorithm(img, contrast, threshold)

    plot.figure("img_original")
    plot.imshow(img / 255.0)
    plot.axis("off")

    plot.figure("img_contrast")
    plot.imshow(img_new)
    plot.axis("off")

    plot.show()
