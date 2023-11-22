import numpy as np


def otus_threshold(image, bins=255, ranges=(0, 255)):
    assert len(image.shape) == 2, print("image should be in gray only 2 dims")
    hist_array, bins_array = np.histogram(image.ravel(), bins, ranges)
    total_pixel = np.multiply(*image.shape)

    # 各个灰度级(取盒子右壁值)为阈值时的背景与前景的灰度均值
    w_b = []
    u_b = []
    w_f = []
    u_f = []
    for k in range(len(hist_array)):
        back_pixel = np.sum(hist_array[: k + 1])
        w_b.append(back_pixel / total_pixel)
        if back_pixel == 0:
            u_b.append(0)
        else:
            u_b.append((bins_array[1 : k + 2] * hist_array[: k + 1]).sum() / back_pixel)

        front_pixel = total_pixel - back_pixel
        w_f.append(front_pixel / total_pixel)
        if front_pixel == 0:
            u_f.append(0)
        else:
            u_f.append(
                (bins_array[k + 2 : bins] * hist_array[k + 1 : bins - 1]).sum()
                / front_pixel
            )

    # 最大类间方差的灰度
    variance_k = np.array(w_b) * np.array(w_f) * (np.array(u_b) - np.array(u_f)) ** 2
    gray_k = list(variance_k).index(np.max(variance_k))

    return gray_k


if __name__ == "__main__":
    data = np.random.rand(20, 20) * 255
    otus_threshold(data)

    print("运行完成")
