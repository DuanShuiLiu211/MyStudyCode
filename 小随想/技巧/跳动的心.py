import random
import time
from math import cos, log, pi, sin
from tkinter import *

CANVAS_WIDTH = 1080  # 画布的宽(X)
CANVAS_HEIGHT = 720  # 画布的高(Y)
CANVAS_CENTER_X = CANVAS_WIDTH / 2  # 画布中心的X坐标
CANVAS_CENTER_Y = CANVAS_HEIGHT / 2  # 画布中心的Y坐标


def heart_function(t, cx=CANVAS_CENTER_X, cy=CANVAS_CENTER_Y, ratio=10):
    """
    爱心生成器
    :param t: 参数
    :param cx: 中心x坐标
    :param cy: 中心y坐标
    :param ratio: 放大比例
    :return: 坐标
    """
    # 心形线函数
    x = 17 * (sin(t) ** 3)
    y = -(16 * cos(t) - 5 * cos(2 * t) - 2 * cos(3 * t) - cos(3 * t))

    # 放大
    x *= ratio
    y *= ratio

    # 移到画布中央
    x += cx
    y += cy

    return int(x), int(y)


def scatter_inside(x, y, cx=CANVAS_CENTER_X, cy=CANVAS_CENTER_Y, beta=0.15):
    """
    内部扩散
    :param x: 原x
    :param y: 原y
    :param beta: 强度
    :return: 新坐标
    """
    ratio_x = -beta * log(random.random())
    ratio_y = -beta * log(random.random())

    dx = ratio_x * (x - cx)
    dy = ratio_y * (y - cy)

    return x - dx, y - dy


def rhythmic(p):
    """
    调整跳动周期
    :param p: 参数
    :return: 正弦
    """
    return 2 * (2 * sin(4 * p)) / (2 * pi)


class Heart:
    def __init__(
        self,
        w=CANVAS_WIDTH,
        h=CANVAS_HEIGHT,
        cx=CANVAS_CENTER_X,
        cy=CANVAS_CENTER_Y,
        generate_frame=60,
    ):
        self._edge_points = set()
        self._edge_diffusion_points = set()
        self._points = set()
        self.all_frame_points = {}

        self.w = w
        self.h = h
        self.cx = cx
        self.cy = cy

        self.build(3000)  # 生成基础的点坐标
        self.generate_frame = generate_frame
        for frame in range(generate_frame):
            self.get_frame(frame)

    def build(self, number):
        for _ in range(number):
            t = random.uniform(0, 2 * pi)
            x, y = heart_function(t, self.cx, self.cy, 10)
            self._edge_points.add((x, y))

        for _x, _y in list(self._edge_points):
            for _ in range(5):
                x, y = scatter_inside(_x, _y, self.cx, self.cy, 0.1)
                self._edge_diffusion_points.add((x, y))

        point_list = list(self._edge_points)
        for _ in range(10000):
            x, y = random.choice(point_list)
            x, y = scatter_inside(x, y, self.cx, self.cy, 0.3)
            self._points.add((x, y))

    @staticmethod
    def shrink_position_1st(x, y, cx=CANVAS_CENTER_X, cy=CANVAS_CENTER_Y, ratio=10):
        force = 1 / (((x - cx) ** 2 + (y - cy) ** 2) ** 0.45)
        dx = ratio * force * (x - cx) + random.randint(-1, 1)
        dy = ratio * force * (y - cy) + random.randint(-1, 1)

        return x - dx, y - dy

    @staticmethod
    def shrink_position_2st(x, y, cx=CANVAS_CENTER_X, cy=CANVAS_CENTER_Y, ratio=10):
        force = -1 / (((x - cx) ** 2 + (y - cy) ** 2) ** 0.6)
        dx = ratio * force * (x - cx)
        dy = ratio * force * (y - cy)

        return x - dx, y - dy

    def get_frame(self, generate_frame):
        all_kind_points = {
            "heart_edge": [],
            "heart_edge_diffusion": [],
            "heart": [],
            "heart_halo": [],
        }
        ratio = 15 * rhythmic(generate_frame / 10 * pi)  # 周期性的缩放系数

        # 动态变化的爱心轮廓
        for x, y in self._edge_points:
            x, y = self.shrink_position_1st(x, y, self.cx, self.cy, ratio)
            size = random.randint(1, 2)
            all_kind_points["heart_edge"].append((x, y, size))

        # 动态变化的扩散的爱心轮廓
        for x, y in self._edge_diffusion_points:
            x, y = self.shrink_position_1st(x, y, self.cx, self.cy, ratio)
            size = random.randint(1, 2)
            all_kind_points["heart_edge_diffusion"].append((x, y, size))

        # 动态变化的爱心
        for x, y in self._points:
            x, y = self.shrink_position_1st(x, y, self.cx, self.cy, ratio)
            size = random.randint(1, 2)
            all_kind_points["heart"].append((x, y, size))

        # 固定大小的爱心光晕
        halo_radius = int(4 + 6 * (1 + rhythmic(generate_frame / 10 * pi)))
        halo_number = int(4000 + 6000 * (1 + rhythmic(generate_frame / 10 * pi)))
        heart_halo_point = set()
        for _ in range(halo_number):
            t = random.uniform(0, 2 * pi)
            x, y = heart_function(t, self.cx, self.cy, ratio=10)
            x, y = self.shrink_position_2st(x, y, self.cx, self.cy, halo_radius)
            if (x, y) not in heart_halo_point:
                heart_halo_point.add((x, y))
                x += random.randint(-40, 40)
                y += random.randint(-40, 40)
                size = random.choice((1, 2, 1))
                all_kind_points["heart_halo"].append((x, y, size))
                all_kind_points["heart_halo"].append((x + 20, y + 20, size))
                all_kind_points["heart_halo"].append((x - 20, y - 20, size))
                all_kind_points["heart_halo"].append((x + 20, y - 20, size))
                all_kind_points["heart_halo"].append((x - 20, y + 20, size))

        self.all_frame_points[generate_frame] = all_kind_points

    def render(self, canvas, current_frame):
        for x, y, size in self.all_frame_points[current_frame % self.generate_frame][
            "heart_edge"
        ]:
            canvas.create_rectangle(x, y, x + size, y + size, width=0, fill="#ef7a82")
        for x, y, size in self.all_frame_points[current_frame % self.generate_frame][
            "heart_edge_diffusion"
        ]:
            canvas.create_rectangle(x, y, x + size, y + size, width=0, fill="#ef7a82")
        for x, y, size in self.all_frame_points[current_frame % self.generate_frame][
            "heart"
        ]:
            canvas.create_rectangle(x, y, x + size, y + size, width=0, fill="#ff2d51")
        for x, y, size in self.all_frame_points[current_frame % self.generate_frame][
            "heart_halo"
        ]:
            canvas.create_rectangle(x, y, x + size, y + size, width=0, fill="#ff2d51")


def draw(
    winer: Tk,
    w=CANVAS_WIDTH,
    h=CANVAS_HEIGHT,
    cx=CANVAS_CENTER_X,
    cy=CANVAS_CENTER_Y,
    current_frame=0,
):
    global canvas
    global heart

    if current_frame == 0:
        canvas = Canvas(winer, bg="black", bd=0, width=w, height=h)
        canvas.pack()
        heart = Heart(w=w, h=h, cx=cx, cy=cy, generate_frame=60)

    if current_frame != 0:
        if winer.winfo_width() != w + 6 or winer.winfo_height() != h + 6:
            w = winer.winfo_width() - 6
            h = winer.winfo_height() - 6
            cx = w / 2
            cy = h / 2
            canvas.destroy()
            winer.update()
            time.sleep(0.03)
            canvas = Canvas(winer, bg="black", bd=0, width=w, height=h)
            canvas.pack()
            heart = Heart(w=w, h=h, cx=cx, cy=cy, generate_frame=60)

    canvas.delete("all")
    heart.render(canvas, current_frame)
    winer.after(int(1000 / 60), draw, winer, w, h, cx, cy, current_frame + 1)


if __name__ == "__main__":
    # 创建窗口事件
    root = Tk()
    root.title("跳动的心")

    # 在窗口中定义绘制
    draw(
        winer=root,
        w=CANVAS_WIDTH,
        h=CANVAS_HEIGHT,
        cx=CANVAS_CENTER_X,
        cy=CANVAS_CENTER_Y,
        current_frame=0,
    )

    # 执行窗口事件
    root.mainloop()

    print("运行完成")
