import matplotlib.pyplot as plt
from PIL import Image

plt.figure()  # 标题从1开始计数
plt.figure()
plt.figure()
plt.figure(-1)
plt.figure(2)
plt.figure(4)
plt.show()


def image_preprocessing():
    img = Image.open("数据/图片1.png")
    ground = img.resize((256, 256))
    ground.save("ProcessedImage.jpg", "JPEG")


if __name__ == "__main__":
    image_preprocessing()
