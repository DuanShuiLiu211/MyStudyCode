import matplotlib.pyplot as plt
from PIL import Image

a = 2

if a:  # a非0既运行
    print(a)


def xfx(b, c=1):
    d = b + c
    return d


print(xfx(1, 2))

plt.figure()  # 默认从1开始
plt.figure()
plt.figure()
plt.figure(-1)
plt.figure(2)
plt.figure(4)
plt.show()


def image_preprocessing():
    im = Image.open('UnprocessImage.jpeg')

    imbackground = im.resize((256, 256))

    imbackground.save('ProcessedImage.jpg', 'JPEG')


if __name__ == "__main__":
    image_preprocessing()
