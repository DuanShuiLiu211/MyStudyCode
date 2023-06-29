import numpy as np
import cv2
import matplotlib.pyplot as plt


def myImread(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img


if __name__ == '__main__':
<<<<<<< HEAD
    path = '数据/图片1.png'
    img = myImread(path)
    plt.figure(0)
    plt.imshow(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.figure(1)
    plt.imshow(img)
    plt.show()
    
=======
    path = 'X:\图片\Saved Pictures\雪山1.JPG'
    img = myImread(path)
    plt.figure(0)
    plt.imshow(img)
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.figure(1)
    plt.imshow(image)
>>>>>>> a98794fef118e4fbd47d0348edb5f8b3154dd000
