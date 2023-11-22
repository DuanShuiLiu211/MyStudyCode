import cv2
import matplotlib.pyplot as plt
import numpy as np


def myImread(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img


if __name__ == "__main__":
    path = "数据/图片1.png"
    img = myImread(path)
    plt.figure(0)
    plt.imshow(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.figure(1)
    plt.imshow(img)
    plt.show()
