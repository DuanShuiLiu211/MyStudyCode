import matplotlib.pyplot as plt
from PIL import Image

<<<<<<< HEAD

plt.figure()  # 标题从1开始计数
=======
a = 2

if a:  # a非0既运行
    print(a)


def xfx(b, c=1):
    d = b + c
    return d


print(xfx(1, 2))

plt.figure()  # 默认从1开始
>>>>>>> a98794fef118e4fbd47d0348edb5f8b3154dd000
plt.figure()
plt.figure()
plt.figure(-1)
plt.figure(2)
plt.figure(4)
plt.show()


def image_preprocessing():
<<<<<<< HEAD
    img = Image.open('数据/图片1.png')
    ground = img.resize((256, 256))
    ground.save('ProcessedImage.jpg', 'JPEG')
=======
    im = Image.open('UnprocessImage.jpeg')

    imbackground = im.resize((256, 256))

    imbackground.save('ProcessedImage.jpg', 'JPEG')
>>>>>>> a98794fef118e4fbd47d0348edb5f8b3154dd000


if __name__ == "__main__":
    image_preprocessing()
