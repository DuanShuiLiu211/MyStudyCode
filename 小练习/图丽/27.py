import pyautogui
import time
import math


def draw_heart(duration):
    # 获取屏幕尺寸
    screen_width, screen_height = pyautogui.size()

    # 确定爱心的中心位置
    center_x = screen_width / 2
    center_y = screen_height / 2

    # 爱心轨迹参数
    num_points = 100  # 点的数量
    scale = 10  # 爱心大小

    # 计算每个点的时间间隔
    interval = duration / num_points

    for i in range(num_points):
        # 计算爱心形状的参数 t
        t = i * 2 * math.pi / num_points

        # 爱心的数学公式
        x = scale * 16 * (math.sin(t) ** 3)
        y = -scale * (
            13 * math.cos(t)
            - 5 * math.cos(2 * t)
            - 2 * math.cos(3 * t)
            - math.cos(4 * t)
        )

        # 计算鼠标的位置
        pos_x = center_x + x
        pos_y = center_y + y

        # 移动鼠标
        pyautogui.moveTo(pos_x, pos_y)

        # 等待下一个点
        time.sleep(interval)


if __name__ == "__main__":
    while True:
        draw_heart(10)
        time.sleep(10)
