import numpy as np
from cv2 import *
import matplotlib.pyplot as plt


def myImread(path):
    img = imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img

if __name__ == '__main__':
    path = 'X:\图片\Saved Pictures\雪山1.JPG'
    img = myImread(path)
    plt.figure(0)
    plt.imshow(img)
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.figure(1)
    plt.imshow(image)