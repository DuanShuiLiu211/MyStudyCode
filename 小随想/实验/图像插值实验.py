import logging
import math

import cv2
import numpy as np


class ImageInterLines(object):
    """设置传入参数"""

    def __init__(self, img, modify_size=(3, 3), *, align="center"):
        self.img = img
        self.h_rate = modify_size[0]  # 高度缩放率
        self.w_rate = modify_size[1]  # 宽度缩放率
        self.align = align  # 设置居中模式，进而进行判断其对齐方式

        self.source_h = img.shape[0]  # 对应 i 列  -> x
        self.source_w = img.shape[1]  # 对饮 j 列  -> y
        self.goal_channel = img.shape[2]  # 通道数

        self.goal_h = round(
            self.source_h * self.h_rate
        )  # 将原图像的size进行按照传入进来的rate参数等比的放大
        self.goal_w = round(self.source_w * self.w_rate)

        if self.align not in ["center", "left"]:
            logging.exception(f"{self.align} is not a valid align parameter")
            self.align = "center"  # 如果传入的参数不是居中或者居左，则强制将其置为居中
            pass

    def set_rate(self, new_modify_size=(None, None)):
        self.h_rate = new_modify_size[0]
        self.w_rate = new_modify_size[1]

    def convert2src_axes(self, des_x, des_y):
        if self.align == "left":  # 左对齐
            src_x = float(des_x * (self.source_w / self.goal_w))
            src_y = float(des_y * (self.source_h / self.goal_h))

            src_x = min((self.source_h - 1), src_x)
            src_y = min((self.source_w - 1), src_y)
        else:  # 几何中心对齐
            src_x = float(
                des_x * (self.source_w / self.goal_w)
                + 0.5 * (self.source_w / self.goal_w)
            )
            src_y = float(
                des_y * (self.source_h / self.goal_h)
                + 0.5 * (self.source_h / self.goal_h)
            )

            src_x = min((self.source_h - 1), src_x)
            src_y = min((self.source_w - 1), src_y)

        return src_x, src_y  # 这里返回的数值可以是小数，也可能是整数例如：23.00，但是这个数仍然是小数

    def nearest_inter(self):
        """最邻近插值法"""

        new_img = np.zeros(
            (self.goal_h, self.goal_w, self.goal_channel), dtype=np.uint8
        )
        for i in range(0, new_img.shape[0]):
            for j in range(0, new_img.shape[1]):
                src_i, src_j = self.convert2src_axes(i, j)
                new_img[i, j] = self.img[round(src_i), round(src_j)]
        return new_img

    def linear_inter(self):
        """线性插值算法"""
        new_img = np.zeros(
            (self.goal_h, self.goal_w, self.goal_channel), dtype=np.uint8
        )
        for i in range(0, new_img.shape[0]):
            for j in range(0, new_img.shape[1]):
                src_i, src_j = self.convert2src_axes(i, j)
                if (
                    ((src_j - int(src_j)) == 0 and (src_i - int(src_i)) == 0)
                    or ((src_i - int(src_i)) == 0)
                    or ((src_j - int(src_j)) == 0)
                ):  # 表明t_border_src_j是一个整数 如果是整数，直接将原图的灰度值付给目标图像即可
                    new_img[i, j] = self.img[
                        round(src_i), round(src_j)
                    ]  # 直接将原图的灰度值赋值给目标图像即可
                else:
                    """否则进行向上和向下取整,然后进行（一维）线性插值算法"""
                    src_i, src_j = self.convert2src_axes(i, j)
                    j1 = int(src_j)  # 向下取整
                    j2 = math.ceil(src_j)  # 向上取整
                    new_img[i, j] = (j2 - src_j) * self.img[round(src_i), j1] + (
                        src_j - j1
                    ) * self.img[round(src_i), j2]

        return new_img

    def double_linear_inter(self):
        new_img = np.zeros(
            (self.goal_h - 2, self.goal_w - 2, self.goal_channel), dtype=np.uint8
        )  # 这里减的目的就是为了可以进行向上向下取整
        for i in range(0, new_img.shape[0]):  # 减1的目的就是为了可以进行向上向下取整数
            for j in range(0, new_img.shape[1]):
                inner_src_i, inner_src_j = self.convert2src_axes(
                    i, j
                )  # 将取得到的变量参数坐标映射到原图中，并且返回映射到原图中的坐标
                inner_i1 = int(inner_src_i)  # 对行进行向上向下取整数
                inner_i2 = math.ceil(inner_src_i)
                inner_j1 = int(inner_src_j)  # 对列进行向上向下取整数
                inner_j2 = math.ceil(inner_src_j)

                Q1 = (inner_j2 - inner_src_j) * self.img[inner_i1, inner_j1] + (
                    inner_src_j - inner_j1
                ) * self.img[inner_i1, inner_j2]
                Q2 = (inner_j2 - inner_src_j) * self.img[inner_i2, inner_j1] + (
                    inner_src_j - inner_j1
                ) * self.img[inner_i2, inner_j2]
                new_img[i, j] = (inner_i2 - inner_src_i) * Q1 + (
                    inner_src_i - inner_i1
                ) * Q2

        return new_img

    def all_transform(self):
        # source_h, source_w, source_channel = self.img.shape  # 获得原图像的高度，宽度和通道量
        # goal_h, goal_w = round(source_h*self.h_rate), round(source_w*self.w_rate)  # 将原图像的size进行按照传入进来的rate参数等比的放大
        # goal_channel = source_channel

        """进行图像转换了"""
        new_img = np.zeros(
            (self.goal_h - 1, self.goal_w - 1, self.goal_channel), dtype=np.uint8
        )  # 得到一个空的数组用来存放转换后的值，即为新的图片

        """边界使用线性插值算法"""
        temp_row = [0, new_img.shape[0] - 1]
        # 上下两行进行线性插值
        for i in temp_row:
            # i -> h -> x
            # j -> w -> y
            for j in range(0, new_img.shape[1]):
                """边界线（除了四个角落）采用线性插值法"""
                t_border_src_i, t_border_src_j = self.convert2src_axes(i, j)
                if (
                    (
                        (t_border_src_j - int(t_border_src_j)) == 0
                        and (t_border_src_i - int(t_border_src_i)) == 0
                    )
                    or (t_border_src_i - int(t_border_src_i)) == 0
                    or (t_border_src_j - int(t_border_src_j)) == 0
                ):  # 表明t_border_src_j是一个整数 如果是整数，直接将原图的灰度值付给目标图像即可
                    new_img[i, j] = self.img[
                        round(t_border_src_i), round(t_border_src_j)
                    ]  # 直接将原图的灰度值赋值给目标图像即可
                else:
                    """否则进行向上和向下取整,然后进行（一维）线性插值算法"""
                    t_border_src_i, t_border_src_j = self.convert2src_axes(i, j)
                    j1 = int(t_border_src_j)  # 向下取整
                    j2 = math.ceil(t_border_src_j)  # 向上取整
                    new_img[i, j] = self.img[round(t_border_src_i), j1] + (
                        t_border_src_j - j1
                    ) * (
                        self.img[round(t_border_src_i), j2]
                        - self.img[round(t_border_src_i), j1]
                    )
        # 左右两列进行线性插值
        temp_col = [0, new_img.shape[1] - 1]
        for i in temp_col:
            # i -> w -> y
            # j -> h -> x
            for j in range(0, new_img.shape[0]):
                """边界线（除了四个角落）采用线性插值法"""
                t_border_src_i, t_border_src_j = self.convert2src_axes(i, j)
                if (
                    (
                        (t_border_src_j - int(t_border_src_j)) == 0
                        and (t_border_src_i - int(t_border_src_i)) == 0
                    )
                    or (t_border_src_i - int(t_border_src_i)) == 0
                    or (t_border_src_j - int(t_border_src_j)) == 0
                ):  # 表明border_src_j是一个整数 如果是整数，直接将原图的灰度值付给目标图像即可
                    new_img[j, i] = self.img[
                        round(t_border_src_i), round(t_border_src_j)
                    ]  # 直接将原图的灰度值赋值给目标图像即可
                else:
                    """否则进行向上和向下取整,然后进行（一维）线性插值算法"""
                    t_border_src_i, t_border_src_j = self.convert2src_axes(j, i)
                    j1 = int(t_border_src_i)  # 向下取整
                    j2 = math.ceil(t_border_src_i)  # 向上取整
                    new_img[j, i] = self.img[j1, round(t_border_src_j)] + (
                        t_border_src_i - j1
                    ) * (
                        self.img[j2, round(t_border_src_j)]
                        - self.img[j1, round(t_border_src_j)]
                    )

        """四个角落(顶点)使用最临近插值算法"""
        corner_low = [0, new_img.shape[0] - 1]
        corner_height = [0, new_img.shape[1] - 1]
        for i in corner_low:
            for j in corner_height:
                src_i, src_j = self.convert2src_axes(i, j)
                new_img[i, j] = self.img[round(src_i), round(src_j)]

        """中间的使用双线性插值法"""
        for i in range(1, new_img.shape[0] - 1):  # 减1的目的就是为了可以进行向上向下取整数
            for j in range(1, new_img.shape[1] - 1):
                inner_src_i, inner_src_j = self.convert2src_axes(
                    i, j
                )  # 将取得到的变量参数坐标映射到原图中，并且返回映射到原图中的坐标
                inner_i1 = int(inner_src_i)  # 对行进行向上向下取整数
                inner_i2 = math.ceil(inner_src_i)

                inner_j1 = int(inner_src_j)  # 对列进行向上向下取整数
                inner_j2 = math.ceil(inner_src_j)
                Q1 = (inner_j2 - inner_src_j) * self.img[inner_i1, inner_j1] + (
                    inner_src_j - inner_j1
                ) * self.img[inner_i1, inner_j2]
                Q2 = (inner_j2 - inner_src_j) * self.img[inner_i2, inner_j1] + (
                    inner_src_j - inner_j1
                ) * self.img[inner_i2, inner_j2]
                new_img[i, j] = (inner_i2 - inner_src_i) * Q1 + (
                    inner_src_i - inner_i1
                ) * Q2

        return new_img


if __name__ == "__main__":
    pic1 = cv2.imread(r"/image_processing/carmen.jpg")
    pic2 = cv2.imread(r"/image_processing/girl.jpg")
    pic3 = cv2.imread(r"/image_processing/architecture.jpg")
    Obj_pic1 = ImageInterLines(pic1, modify_size=(2, 2), align="center")
    new_pic1 = Obj_pic1.nearest_inter()
    cv2.imshow("origin", pic1)
    cv2.imshow("nearest_inter", new_pic1)
    new_pic2 = Obj_pic1.linear_inter()
    cv2.imshow("liner_inter", new_pic2)
    new_pic3 = Obj_pic1.all_transform()
    cv2.imshow("double_liner_inter", new_pic3)

    cv2.waitKey()
